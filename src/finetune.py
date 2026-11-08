"""
finetune.py
-----------
End-to-end fine-tuning of pre-trained encoders for 13-way routing (GPU recommended).

Protocol (same splits as every other model):
  1. selection : for each learning rate, train on `train` for up to MAX_EPOCHS and pick the
                 epoch with the best validation macro-F1 (seed 0)
  2. final     : retrain on train+val with the selected (lr, epochs) for SEEDS seeds and
                 predict `test` once per seed (per-document probabilities are stored)
  3. cv        : optional 5-fold grouped CV (seed-0 folds) with the selected configuration

Efficiency: sequences are truncated at MAX_LEN tokens but padded only to the longest sequence
of each batch, and training batches are formed from documents of similar length (length-grouped
sampling, as in Hugging Face `group_by_length`).  Mixed precision uses bf16 on GPUs with native
bf16 (compute capability >= 8.0: A100, L4, H100) and fp16 with loss scaling otherwise (T4, V100).

Every run writes results/ft_runs_v2/<tag>.json and is skipped when that file exists, so the
script resumes after an interruption.

    python src/finetune.py --cv --models dbmdz/bert-base-turkish-cased xlm-roberta-base
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FT_RUNS, PREDS, RESULTS, append_results, bootstrap_ci, cv_folds, load_docs, load_variants, metrics, split_indices

FT_DIR = FT_RUNS
MAX_LEN, BATCH, EVAL_BATCH, MAX_EPOCHS, WARMUP, WD = 256, 32, 128, 6, 0.1, 0.01
SEEDS_FINAL = (0, 1, 2)


# ----------------------------------------------------------------------------- data
class DS(Dataset):
    """Holds unpadded token id lists; padding is done per batch by the collator."""
    def __init__(self, enc, ids, y):
        self.enc, self.ids, self.y = enc, np.asarray(ids), y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        d = {k: v[j] for k, v in self.enc.items()}
        d["labels"] = int(self.y[j])
        return d


class LengthGroupedBatchSampler(Sampler):
    """Shuffles documents, sorts them by length inside mega-batches of `mega` batches, cuts
    batches, and shuffles the batch order.  A new permutation is drawn every epoch from a
    seeded generator, so training is reproducible."""
    def __init__(self, lengths, batch_size, seed, mega=50):
        self.lengths = np.asarray(lengths); self.bs = batch_size; self.mega = mega
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        step = self.bs * self.mega
        batches = []
        for s in range(0, len(idx), step):
            chunk = idx[s:s + step]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            batches += [chunk[i:i + self.bs].tolist() for i in range(0, len(chunk), self.bs)]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


def sorted_batches(lengths, batch_size):
    order = np.argsort(-np.asarray(lengths), kind="stable")
    return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def tokenize(tok, texts):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)          # no padding here
    return {k: list(v) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


def amp_dtype(device):
    if device.type != "cuda":
        return torch.float32
    # is_bf16_supported() is also True on T4 through (slow) emulation, so check the hardware generation
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def predict(model, ds, collate, device):
    """Probabilities in the order of ds.ids (batches are formed by length for speed)."""
    model.eval()
    lengths = [len(ds.enc["input_ids"][j]) for j in ds.ids]
    out = np.zeros((len(ds), model.config.num_labels), dtype=np.float32)
    loader = DataLoader(ds, batch_sampler=sorted_batches(lengths, EVAL_BATCH), collate_fn=collate)
    order = [i for b in sorted_batches(lengths, EVAL_BATCH) for i in b]
    pos = 0
    with torch.no_grad():
        for b in loader:
            n = b["input_ids"].shape[0]
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == "cuda"):
                logits = model(**b).logits
            out[order[pos:pos + n]] = torch.softmax(logits.float(), -1).cpu().numpy()
            pos += n
    return out


# ----------------------------------------------------------------------------- training
def train_one(ckpt, ds_tr, collate, n_classes, lr, epochs, seed, device, eval_fn=None):
    """Trains for `epochs`; if eval_fn is given it is called after every epoch and the
    per-epoch outputs are returned (used for epoch selection)."""
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed); np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=n_classes).to(device)
    lengths = [len(ds_tr.enc["input_ids"][j]) for j in ds_tr.ids]
    sampler = LengthGroupedBatchSampler(lengths, BATCH, seed)
    loader = DataLoader(ds_tr, batch_sampler=sampler, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    total = len(sampler) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * total), total)
    dt = amp_dtype(device); scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    history = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); loss_sum = 0.0; n_tok = 0
        for b in loader:
            n_tok += int(b["attention_mask"].sum())
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=dt, enabled=device.type == "cuda"):
                loss = model(**b).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True); loss_sum += loss.item()
        secs = time.time() - t0
        rec = {"epoch": ep + 1, "train_loss": round(loss_sum / len(sampler), 4), "epoch_seconds": round(secs, 1),
               "docs_per_s": round(len(ds_tr) / secs, 1)}
        if eval_fn is not None:
            rec.update(eval_fn(model))
        history.append(rec); print("   ", rec, flush=True)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["dbmdz/bert-base-turkish-cased", "xlm-roberta-base"])
    ap.add_argument("--lrs", nargs="*", type=float, default=[2e-5, 5e-5])
    ap.add_argument("--field", default="text_clean")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer, DataCollatorWithPadding
    transformers.logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print("device:", device, "| precision:", amp_dtype(device), "| GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "-", flush=True)
    docs, labels = load_docs(); var = load_variants(); idx = split_indices(docs)
    texts = var[args.field].fillna("").values
    y = np.array([labels.index(l) for l in docs["label"]]); y_str = docs["label"].values
    label_sets = docs["label_set"].values
    FT_DIR.mkdir(parents=True, exist_ok=True); PREDS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "preds_val").mkdir(parents=True, exist_ok=True)

    for ckpt in args.models:
        short = "ft_" + ckpt.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(ckpt)
        enc_all = tokenize(tok, texts)
        collate = DataCollatorWithPadding(tok)
        L = np.array([len(x) for x in enc_all["input_ids"]])
        print(f"[{short}] tokens per document: mean {L.mean():.0f}, median {np.median(L):.0f}, "
              f"truncated at {MAX_LEN}: {(L >= MAX_LEN).mean():.1%}", flush=True)
        mk = lambda ids: DS(enc_all, ids, y)
        ds_val = mk(idx["val"])
        prec = str(amp_dtype(device)).replace("torch.", "")

        # ---- 1. selection (seed 0): lr x epoch on the validation split
        sel_file = FT_DIR / f"{short}__selection.json"
        if sel_file.exists():
            sel = json.loads(sel_file.read_text())
        else:
            sel = {"runs": [], "precision": prec, "max_len": MAX_LEN, "batch": BATCH}
            for lr in args.lrs:
                def eval_fn(model):
                    p = predict(model, ds_val, collate, device)
                    m = metrics(y_str[idx["val"]], np.array(labels)[p.argmax(1)], labels)
                    return {"val_accuracy": round(m["accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4)}
                print(f"[{short}] selection lr={lr}", flush=True)
                _, hist = train_one(ckpt, mk(idx["train"]), collate, len(labels), lr, args.max_epochs, 0, device, eval_fn)
                sel["runs"].append({"lr": lr, "history": hist})
            best = max(((r["lr"], h["epoch"], h["val_macro_f1"]) for r in sel["runs"] for h in r["history"]), key=lambda t: t[2])
            sel["best"] = {"lr": best[0], "epochs": best[1], "val_macro_f1": best[2]}
            sel_file.write_text(json.dumps(sel, indent=2))
        lr, epochs = sel["best"]["lr"], sel["best"]["epochs"]
        print(f"[{short}] selected lr={lr} epochs={epochs} (val macro-F1={sel['best']['val_macro_f1']:.4f})", flush=True)

        # ---- 1b. validation predictions of the selected configuration (train only, seed 0) for deferral thresholds
        done_1b = FT_DIR / f"{short}__val_preds.done"
        if not done_1b.exists():
            model, _ = train_one(ckpt, mk(idx["train"]), collate, len(labels), lr, epochs, 0, device)
            P = predict(model, ds_val, collate, device)
            pd.to_pickle({"doc_id": docs["doc_id"].values[idx["val"]], "y_true": y_str[idx["val"]], "y_pred": np.array(labels)[P.argmax(1)],
                          "scores": P, "labels": labels, "params": {"lr": lr, "epochs": epochs}, "family": "FT_s0", "representation": short},
                         RESULTS / "preds_val" / f"{short}__FT_s0.pkl")
            done_1b.write_text("ok"); del model; torch.cuda.empty_cache()

        # ---- 2. final: train+val -> test, several seeds
        ds_test = mk(idx["test"])
        for seed in SEEDS_FINAL:
            tag = f"{short}__final_s{seed}"
            if (FT_DIR / f"{tag}.json").exists():
                print("done:", tag); continue
            t0 = time.time()
            model, hist = train_one(ckpt, mk(idx["trainval"]), collate, len(labels), lr, epochs, seed, device)
            fit_t = time.time() - t0
            t0 = time.time(); P = predict(model, ds_test, collate, device); pred_t = (time.time() - t0) / len(idx["test"])
            yp = np.array(labels)[P.argmax(1)]
            m = metrics(y_str[idx["test"]], yp, labels, scores=P, label_sets=label_sets[idx["test"]])
            ci = bootstrap_ci(y_str[idx["test"]], yp, labels)
            row = {"representation": short, "family": f"FT_s{seed}",
                   "params": json.dumps({"lr": lr, "epochs": epochs, "seed": seed, "max_len": MAX_LEN, "batch": BATCH, "precision": prec}),
                   **m, **ci, "fit_seconds_trainval": fit_t, "predict_ms_per_doc": pred_t * 1000, "n_features": 0,
                   "checkpoint": ckpt, "model_family": "fine-tuned encoder", "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
            pd.to_pickle({"doc_id": docs["doc_id"].values[idx["test"]], "y_true": y_str[idx["test"]], "y_pred": yp, "scores": P,
                          "labels": labels, "params": row["params"], "family": row["family"], "representation": short}, PREDS / f"{short}__FT_s{seed}.pkl")
            append_results([row], file="results_ft.csv"); (FT_DIR / f"{tag}.json").write_text(json.dumps(row, indent=2))
            print(f"[{tag}] test acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}", flush=True)
            del model; torch.cuda.empty_cache()

        # ---- 3. optional grouped CV (seed-0 folds)
        if args.cv:
            accs, f1s = [], []
            for k, (tr, te) in enumerate(cv_folds(docs)[:5]):
                tag = f"{short}__cv_f{k}"
                f = FT_DIR / f"{tag}.json"
                if f.exists():
                    r = json.loads(f.read_text())
                else:
                    model, _ = train_one(ckpt, mk(tr), collate, len(labels), lr, epochs, 0, device)
                    P = predict(model, mk(te), collate, device)
                    m = metrics(y_str[te], np.array(labels)[P.argmax(1)], labels)
                    r = {"fold": k, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                    f.write_text(json.dumps(r)); del model; torch.cuda.empty_cache()
                    print(f"[{tag}] acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}", flush=True)
                accs.append(r["accuracy"]); f1s.append(r["macro_f1"])
            cvr = {"cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_sd": float(np.std(accs, ddof=1)),
                   "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_sd": float(np.std(f1s, ddof=1))}
            (FT_DIR / f"{short}__cv_summary.json").write_text(json.dumps(cvr, indent=2))
            print(f"[{short}] CV:", cvr, flush=True)


if __name__ == "__main__":
    main()

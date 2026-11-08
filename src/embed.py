"""
embed.py
--------
Extracts frozen sentence embeddings for every document and caches them as .npy files
(cache/emb_<model_key>__<text_field>.npy).  Re-running skips cached files, so the script
is resumable.  Usage:

    python src/embed.py                      # all CPU-feasible models on text_clean
    python src/embed.py --models e5_base labse --field text_clean
    python src/embed.py --list

Models are referenced by a short key; `MODELS` records the Hugging Face checkpoint,
the pooling strategy and the prefix required by instruction-tuned encoders.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
MAX_SEQ = 256

MODELS = {
    # key: (checkpoint, family, pooling, query prefix, notes)
    "minilm_multi": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "SBERT multilingual (distilled)", "mean", "", "118M params, 384-d"),
    "mpnet_multi": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "SBERT multilingual", "mean", "", "278M params, 768-d"),
    "distiluse_multi": ("sentence-transformers/distiluse-base-multilingual-cased-v2", "multilingual USE (distilled)", "mean+dense", "", "135M params, 512-d"),
    "e5_base": ("intfloat/multilingual-e5-base", "mE5", "mean", "query: ", "278M params, 768-d"),
    "e5_large": ("intfloat/multilingual-e5-large", "mE5", "mean", "query: ", "560M params, 1024-d"),
    "labse": ("sentence-transformers/LaBSE", "LaBSE", "cls+dense", "", "471M params, 768-d"),
    "sbert_tr": ("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", "Turkish SBERT (BERTurk + NLI/STS)", "mean", "", "111M params, 768-d"),
    "berturk_mean": ("dbmdz/bert-base-turkish-cased", "BERTurk (no sentence fine-tuning)", "mean", "", "111M params, 768-d"),
    "bge_m3": ("BAAI/bge-m3", "BGE-M3", "cls", "", "568M params, 1024-d"),
    "mpnet_en": ("sentence-transformers/all-mpnet-base-v2", "SBERT English", "mean", "", "110M params, 768-d; English text only"),
    "qwen3_0_6b": ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding", "last-token", "Instruct: Given a Turkish institutional document, identify the organisational unit responsible for it\nQuery: ", "0.6B params, 1024-d"),
    "qwen3_4b": ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding", "last-token", "Instruct: Given a Turkish institutional document, identify the organisational unit responsible for it\nQuery: ", "4B params, 2560-d; GPU"),
}
GPU_EXTRA = ["e5_large", "bge_m3", "qwen3_0_6b", "qwen3_4b"]
CPU_DEFAULT = ["minilm_multi", "distiluse_multi", "sbert_tr", "berturk_mean", "mpnet_multi", "e5_base", "labse"]


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(key):
    from sentence_transformers import SentenceTransformer, models
    ckpt, _, pooling, _, _ = MODELS[key]
    kw = {}
    if device() == "cuda" and key.startswith("qwen3"):
        import transformers
        dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        half = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16   # T4 only emulates bf16
        kw = {"model_kwargs": {dkey: half}, "tokenizer_kwargs": {"padding_side": "left"}}
    if key == "berturk_mean":
        word = models.Transformer(ckpt, max_seq_length=MAX_SEQ)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
        enc = SentenceTransformer(modules=[word, pool], device=device())
    else:
        enc = SentenceTransformer(ckpt, device=device(), **kw)
        enc.max_seq_length = MAX_SEQ
    return enc


def embed(key, texts, batch_size=None, log=print):
    ckpt, _, _, prefix, _ = MODELS[key]
    enc = load_encoder(key)
    batch_size = batch_size or (64 if device() == "cuda" else 32)
    t0 = time.time()
    X = enc.encode([prefix + t for t in texts], batch_size=batch_size, show_progress_bar=False,
                   convert_to_numpy=True, normalize_embeddings=False)
    dt = time.time() - t0
    if not np.isfinite(X).all():
        raise ValueError(f"{key}: {int((~np.isfinite(X)).any(axis=1).sum())} documents have non-finite embeddings "
                         "(fp16 overflow?) - not cached; run this model on a GPU with bf16 support (L4/A100)")
    log(f"{key}: {len(texts)} docs in {dt/60:.1f} min ({len(texts)/dt:.1f} docs/s), dim={X.shape[1]}")
    return X.astype(np.float32), dt


def main():
    global MAX_SEQ
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="default: CPU set, plus the large models when a GPU is present")
    ap.add_argument("--field", default="text_clean")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max_seq", type=int, default=MAX_SEQ)
    args = ap.parse_args()
    if args.list:
        for k, v in MODELS.items():
            print(f"{k:16s} {v[0]:60s} {v[4]}")
        return
    MAX_SEQ = args.max_seq
    if args.models is None:
        args.models = CPU_DEFAULT + (GPU_EXTRA if torch.cuda.is_available() else [])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    var = pd.read_pickle(ROOT / "data" / "text_variants.pkl")
    texts = var[args.field].fillna("").tolist()
    CACHE.mkdir(exist_ok=True)
    timing_file = CACHE / "embedding_times.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    failed = []
    for key in args.models:
        out = CACHE / f"emb_{key}__{args.field}.npy"
        if out.exists():
            print("cached:", out.name)
            continue
        try:
            X, dt = embed(key, texts)
        except Exception as e:          # one failing model must not stop the others
            print(f"!! FAILED {key}: {repr(e)[:400]}", flush=True)
            failed.append(key)
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        np.save(out, X)
        timing[f"{key}__{args.field}"] = {"seconds": dt, "docs": len(texts), "dim": int(X.shape[1]), "max_seq": MAX_SEQ,
                                          "device": device(), "threads": torch.get_num_threads()}
        timing_file.write_text(json.dumps(timing, indent=2))
        del X
        import gc; gc.collect()
        if device() == "cuda":
            torch.cuda.empty_cache()
    if failed:
        print("\nEmbedding extraction failed for:", failed, "(the other models are cached; re-run to retry)")


if __name__ == "__main__":
    main()

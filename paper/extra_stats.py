"""
extra_stats.py
--------------
Additional statistics quoted in the manuscript (written to results/tables/extra_stats.json):
rank agreement between estimates, the retrieved-neighbour vote that the RAG prompt contains,
extra McNemar tests (Holm-corrected as one family) and the cost of the cascade.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis import confidence, load_pred, load_results, selected_systems   # noqa: E402
from common import RESULTS, load_docs, mcnemar_test, split_indices           # noqa: E402
from llm import final_answers                                                 # noqa: E402

docs, labels = load_docs()
idx = split_indices(docs)
df = load_results()
sel = selected_systems(df, docs, labels)
out = {}

# 1. rank agreement --------------------------------------------------------------------------
loc = sel[(~sel["ablation"]) & (sel["group"] != "LLM") & sel["cv_accuracy_mean"].notna()]
rho, p = spearmanr(loc["accuracy"], loc["cv_accuracy_mean"])
out["spearman_test_vs_cv_accuracy"] = {"rho": float(rho), "p": float(p), "n": int(len(loc))}
vm = sel[(~sel["ablation"]) & sel["val_macro_f1"].notna()]
rho, p = spearmanr(vm["val_macro_f1"], vm["macro_f1"])
out["spearman_val_vs_test_macro_f1"] = {"rho": float(rho), "p": float(p), "n": int(len(vm))}
out["cv_minus_test_accuracy_points"] = {"min": float(100 * (loc["cv_accuracy_mean"] - loc["accuracy"]).min()),
                                        "max": float(100 * (loc["cv_accuracy_mean"] - loc["accuracy"]).max()),
                                        "median": float(100 * (loc["cv_accuracy_mean"] - loc["accuracy"]).median())}

# 2. vote of the ten retrieved neighbours (the labels shown in the RAG prompt) ----------------
def knn_vote(split):
    nbs = pd.read_pickle(RESULTS / "llm" / f"neighbours__rag10__{split}.pkl")
    ids = docs["doc_id"].values[idx[split]]; y = docs["label"].values[idx[split]]
    pred = []
    for d in ids:
        labs = [docs["label"].values[j] for j in nbs[int(d)]]
        counts = pd.Series(labs).value_counts()
        top = counts[counts == counts.max()].index
        pred.append(next(l for l in labs if l in top))          # tie -> label of the nearest neighbour
    pred = np.array(pred)
    return ids, y, pred
ids_t, y_t, knn_t = knn_vote("test")
_, y_v, knn_v = knn_vote("val")
out["knn10_vote"] = {"test_accuracy": float((knn_t == y_t).mean()), "test_macro_f1": float(f1_score(y_t, knn_t, labels=labels, average="macro", zero_division=0)),
                     "val_accuracy": float((knn_v == y_v).mean()), "val_macro_f1": float(f1_score(y_v, knn_v, labels=labels, average="macro", zero_division=0))}

# 3. extra McNemar tests ------------------------------------------------------------------------
def pk(contains):
    return sel.loc[sel["system"].str.contains(contains, regex=False), "pred_key"].iloc[0]
def preds(key):
    p = load_pred(key)
    return pd.Series(np.asarray(p["y_pred"]), index=p["doc_id"]).reindex(ids_t).values
P = {"BERTurk fine-tuned": preds(pk("BERTurk fine-tuned")), "Qwen3-Embedding-4B / RBF-SVM": preds(pk("Qwen3-Embedding-4B")),
     "kNN-10 vote": knn_t}
reps_tab = pd.read_csv(RESULTS / "tables" / "representatives.csv")
LEX = reps_tab.loc[reps_tab["group"] == "Lexical (TF-IDF)", "system"].iloc[0]       # lexical family representative
P[LEX] = preds(sel.loc[sel["system"] == LEX, "pred_key"].iloc[0])
out["lexical_representative"] = LEX
# retrieval-augmented LLMs that were run (results_llm.csv); the study may include a subset of these models
LLM_SHORT = {"gemini-3.8-flash": "Gemini", "gpt-6-luna": "GPT", "claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet"}
_llm_rows = pd.read_csv(RESULTS / "results_llm.csv")
LLM_RUN = [m for m in LLM_SHORT if ((_llm_rows["representation"] == f"llm_{m}") & (_llm_rows["family"] == "rag10")).any()]
for m in LLM_RUN:
    P[f"{LLM_SHORT[m]} RAG"] = preds(f"llm_{m}__rag10")

# cascade predictions (local = BERTurk fine-tuned seed 0, share chosen on validation by best accuracy)
ft_key = pk("BERTurk fine-tuned"); pt, pv = load_pred(ft_key), load_pred(ft_key, val=True)
fb = docs.loc[docs["split"] != "test", "label"].value_counts().index[0]
llm_cost = pd.read_csv(RESULTS / "results_llm.csv").set_index(["representation", "family"])
def cascade(model):
    va = final_answers(model, "rag10", "val", labels)
    llm_v = np.array([(va[d] or [fb])[0] for d in pv["doc_id"]])
    lt = load_pred(f"llm_{model}__rag10"); llm_t = pd.Series(lt["y_pred"], index=lt["doc_id"]).reindex(pt["doc_id"]).values
    cv_, ct = confidence(pv), confidence(pt)
    def run(share, conf, local, llm):
        k = int(round(share * len(conf))); low = np.argsort(conf)[:k]; p = np.asarray(local).copy(); p[low] = llm[low]; return p
    shares = np.linspace(0, 1, 101)
    accs = [(run(s, cv_, pv["y_pred"], llm_v) == np.asarray(pv["y_true"])).mean() for s in shares]
    s = float(shares[int(np.argmax(accs))])
    p_t = run(s, ct, pt["y_pred"], llm_t)
    p_t = pd.Series(p_t, index=pt["doc_id"]).reindex(ids_t).values
    usd = float(llm_cost.loc[(f"llm_{model}", "rag10"), "usd_per_1000_docs"])
    return s, p_t, usd, float(accs[0]), float(accs[-1])
casc = {}
for model in LLM_RUN:
    name = f"Cascade BERTurk -> {LLM_SHORT[model]}"
    s, p_t, usd, v_loc, v_llm = cascade(model)
    P[name] = p_t
    casc[name] = {"llm_share": s, "test_accuracy": float((p_t == y_t).mean()),
                  "test_macro_f1": float(f1_score(y_t, p_t, labels=labels, average="macro", zero_division=0)),
                  "llm_usd_per_1000_docs_alone": usd, "llm_usd_per_1000_docs_cascade": usd * s,
                  "val_accuracy_local_alone": v_loc, "val_accuracy_llm_alone": v_llm}
out["cascade"] = casc

# pairs among the LLMs that were run (each against the most accurate one), plus the fixed local comparisons
rag = sorted([n for n in P if n.endswith(" RAG")], key=lambda n: -(P[n] == y_t).mean())
pairs = [(rag[0], other) for other in rag[1:]]
pairs += [(rag[0], "BERTurk fine-tuned"), (rag[0], "Qwen3-Embedding-4B / RBF-SVM"), ("BERTurk fine-tuned", "Qwen3-Embedding-4B / RBF-SVM"),
          ("Qwen3-Embedding-4B / RBF-SVM", LEX)]
pairs += [(n, "kNN-10 vote") for n in rag[:2]]
best_casc = max([n for n in P if n.startswith("Cascade")], key=lambda n: (P[n] == y_t).mean())
pairs += [(best_casc, "BERTurk fine-tuned"), (best_casc, best_casc.split("-> ")[1] + " RAG")]
rows = []
for a, b in pairs:
    r = mcnemar_test(y_t, P[a], P[b])
    rows.append({"system_a": a, "system_b": b, "acc_a": float((P[a] == y_t).mean()), "acc_b": float((P[b] == y_t).mean()), **r})
# translation ablation on identical bodies (TF-IDF, selected classifier of each representation)
ctr, cen = load_pred(pk("TF-IDF content (TR)")), load_pred(pk("TF-IDF content (EN"))
a_ = pd.Series(ctr["y_pred"], index=ctr["doc_id"]).reindex(ids_t).values; b_ = pd.Series(cen["y_pred"], index=cen["doc_id"]).reindex(ids_t).values
rows.append({"system_a": "TF-IDF body (TR)", "system_b": "TF-IDF body (EN, translated)", "acc_a": float((a_ == y_t).mean()), "acc_b": float((b_ == y_t).mean()),
             **mcnemar_test(y_t, a_, b_)})
# the same comparison for sentence encoders: multilingual mpnet on the Turkish body vs English all-mpnet on the translated body
etr, een = load_pred(pk("paraphrase-multilingual-mpnet-base [content only")), load_pred(pk("all-mpnet-base-v2"))
a_ = pd.Series(etr["y_pred"], index=etr["doc_id"]).reindex(ids_t).values; b_ = pd.Series(een["y_pred"], index=een["doc_id"]).reindex(ids_t).values
rows.append({"system_a": "mpnet body (TR, multilingual)", "system_b": "mpnet body (EN, translated)", "acc_a": float((a_ == y_t).mean()),
             "acc_b": float((b_ == y_t).mean()), **mcnemar_test(y_t, a_, b_)})
P["mpnet body (TR, multilingual)"], P["mpnet body (EN, translated)"] = a_, b_
mc = pd.DataFrame(rows)
order = np.argsort(mc["p_value"].values); m = len(mc); adj = np.empty(m); running = 0
for rank, k in enumerate(order):
    running = max(running, (m - rank) * mc["p_value"].values[k]); adj[k] = min(1, running)
mc["p_holm"] = adj
mc.to_csv(RESULTS / "tables" / "mcnemar_extra.csv", index=False)
out["mcnemar_extra"] = mc.to_dict("records")

# 4. seed sensitivity: the tests above use BERTurk seed 0; the same tests with the other two seeds ------------------
assert ft_key.endswith("_s0"), ft_key
ft_seed_keys = [ft_key[:-1] + str(s) for s in range(3)]
seed_rows = {}
for key in ft_seed_keys:
    try:
        pb = preds(key)
    except Exception:
        continue
    seed_rows[key] = {"accuracy": float((pb == y_t).mean()),
                      **{f"p_vs_{n}": float(mcnemar_test(y_t, pb, P[n])["p_value"]) for n in [LEX, "Qwen3-Embedding-4B / RBF-SVM", rag[0]]}}
out["berturk_seed_sensitivity"] = seed_rows

# 5. paired bootstrap of macro-F1 differences (2,000 resamples of the test documents) ----------------------------
lab_id = {l: i for i, l in enumerate(labels)}
yi = np.array([lab_id[v] for v in y_t])
def as_int(p):
    return np.array([lab_id.get(v, -1) for v in p])
def macro_f1_fast(y, p, K=len(labels), keep=None):
    tp = np.bincount(y[y == p], minlength=K)[:K]; pc = np.bincount(p[p >= 0], minlength=K)[:K]; tc = np.bincount(y, minlength=K)[:K]
    den = pc + tc
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return float(f1.mean() if keep is None else f1[keep].mean())
rng = np.random.default_rng(0)
boot_idx = [rng.integers(0, len(yi), len(yi)) for _ in range(2000)]
f1_pairs = [(rag[0], n) for n in rag[1:]] + [(rag[0], "Qwen3-Embedding-4B / RBF-SVM"), (rag[0], "BERTurk fine-tuned"), (rag[0], LEX),
                                            ("BERTurk fine-tuned", LEX), ("BERTurk fine-tuned", "Qwen3-Embedding-4B / RBF-SVM"),
                                            (LEX, "Qwen3-Embedding-4B / RBF-SVM"), (best_casc, rag[0]),
                                            ("mpnet body (TR, multilingual)", "mpnet body (EN, translated)")]
boot = {}
for a, b in f1_pairs:
    pa, pb = as_int(P[a]), as_int(P[b])
    d = np.array([macro_f1_fast(yi[i], pa[i]) - macro_f1_fast(yi[i], pb[i]) for i in boot_idx])
    boot[f"{a} | {b}"] = {"diff": macro_f1_fast(yi, pa) - macro_f1_fast(yi, pb), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                          "share_le_0": float((d <= 0).mean())}
out["macro_f1_difference_bootstrap"] = boot
# the same without the unit with the fewest test documents (one document: its F1 is either 0 or driven by a single case)
smallest = labels[int(np.argmin(np.bincount(yi, minlength=len(labels))))]
keep = np.array([l != smallest for l in labels])
boot_s = {"excluded_unit": smallest, "n_test_documents_of_excluded_unit": int((yi == lab_id[smallest]).sum()),
          "macro_f1": {n: macro_f1_fast(yi, as_int(P[n]), keep=keep) for n in rag + ["BERTurk fine-tuned", "Qwen3-Embedding-4B / RBF-SVM", LEX]}}
for a, b in [(rag[0], n) for n in rag[1:]] + [(rag[0], "Qwen3-Embedding-4B / RBF-SVM"), (rag[0], "BERTurk fine-tuned"), (rag[0], LEX)]:
    pa, pb = as_int(P[a]), as_int(P[b])
    d = np.array([macro_f1_fast(yi[i], pa[i], keep=keep) - macro_f1_fast(yi[i], pb[i], keep=keep) for i in boot_idx])
    boot_s[f"{a} | {b}"] = {"diff": macro_f1_fast(yi, pa, keep=keep) - macro_f1_fast(yi, pb, keep=keep),
                            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
out["macro_f1_difference_bootstrap_without_smallest_unit"] = boot_s

# 6. retrieval-augmented prompts asked again after the retrieval index was rebuilt on the de-identified text ------------
plan_f = RESULTS / "llm" / "rerun_changed_plan.json"
if plan_f.exists():
    out["rag_prompts_asked_again"] = json.loads(plan_f.read_text())["summary"]
(RESULTS / "tables" / "extra_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k != "mcnemar_extra"}, indent=1))
print(mc.round(4).to_string(index=False))

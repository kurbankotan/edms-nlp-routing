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
     "Gemini RAG": preds("llm_gemini-3.8-flash__rag10"), "Sonnet RAG": preds("llm_claude-sonnet-5-5__rag10"),
     "GPT RAG": preds("llm_gpt-6-luna__rag10"), "TF-IDF stem / LinearSVM": preds(pk("TF-IDF word (stem)")),
     "kNN-10 vote": knn_t}

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
    return s, p_t, usd
casc = {}
for model, name in [("gemini-3.8-flash", "Cascade BERTurk -> Gemini"), ("claude-sonnet-5-5", "Cascade BERTurk -> Sonnet")]:
    s, p_t, usd = cascade(model)
    P[name] = p_t
    casc[name] = {"llm_share": s, "test_accuracy": float((p_t == y_t).mean()),
                  "test_macro_f1": float(f1_score(y_t, p_t, labels=labels, average="macro", zero_division=0)),
                  "llm_usd_per_1000_docs_alone": usd, "llm_usd_per_1000_docs_cascade": usd * s}
out["cascade"] = casc

pairs = [("Sonnet RAG", "Gemini RAG"), ("Gemini RAG", "GPT RAG"), ("Gemini RAG", "BERTurk fine-tuned"),
         ("BERTurk fine-tuned", "Qwen3-Embedding-4B / RBF-SVM"), ("Qwen3-Embedding-4B / RBF-SVM", "TF-IDF stem / LinearSVM"),
         ("Gemini RAG", "kNN-10 vote"), ("Sonnet RAG", "kNN-10 vote"),
         ("Cascade BERTurk -> Gemini", "BERTurk fine-tuned"), ("Cascade BERTurk -> Gemini", "Gemini RAG")]
rows = []
for a, b in pairs:
    r = mcnemar_test(y_t, P[a], P[b])
    rows.append({"system_a": a, "system_b": b, "acc_a": float((P[a] == y_t).mean()), "acc_b": float((P[b] == y_t).mean()), **r})
# translation ablation on identical bodies (TF-IDF, selected classifier of each representation)
ctr, cen = load_pred(pk("TF-IDF content (TR)")), load_pred(pk("TF-IDF content (EN"))
a_ = pd.Series(ctr["y_pred"], index=ctr["doc_id"]).reindex(ids_t).values; b_ = pd.Series(cen["y_pred"], index=cen["doc_id"]).reindex(ids_t).values
rows.append({"system_a": "TF-IDF body (TR)", "system_b": "TF-IDF body (EN, translated)", "acc_a": float((a_ == y_t).mean()), "acc_b": float((b_ == y_t).mean()),
             **mcnemar_test(y_t, a_, b_)})
mc = pd.DataFrame(rows)
order = np.argsort(mc["p_value"].values); m = len(mc); adj = np.empty(m); running = 0
for rank, k in enumerate(order):
    running = max(running, (m - rank) * mc["p_value"].values[k]); adj[k] = min(1, running)
mc["p_holm"] = adj
mc.to_csv(RESULTS / "tables" / "mcnemar_extra.csv", index=False)
out["mcnemar_extra"] = mc.to_dict("records")
(RESULTS / "tables" / "extra_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k != "mcnemar_extra"}, indent=1))
print(mc.round(4).to_string(index=False))

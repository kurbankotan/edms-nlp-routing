"""
analysis.py
-----------
Turns the stored predictions into the tables and figures of the paper.

    python src/analysis.py            # everything that is available so far

Outputs (results/ and figures/):
  tables.xlsx                 all tables (one sheet each) + CSV copies in results/tables/
  fig_dataset.{png,pdf}       class distribution and monthly volume with the split boundaries
  fig_main.{png,pdf}          test accuracy and macro-F1 (95% bootstrap CI) per selected system
  fig_confusion.{png,pdf}     row-normalised confusion matrix of the final system
  fig_perclass.{png,pdf}      per-unit F1 vs. training support for the main systems
  fig_deferral.{png,pdf}      coverage-accuracy curves (human deferral) and the LLM cascade
  fig_ablation.{png,pdf}      preprocessing / input-field / translation / context ablations
  fig_pipeline.{png,pdf}      framework diagram (Graphviz)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FT_RUNS, PARENT, PREDS, RESULTS, ROOT, SHORT, load_all_results, load_docs, mcnemar_test, metrics, per_class_table, split_indices

FIG = ROOT / "figures"
TAB = RESULTS / "tables"
PREDS_VAL = RESULTS / "preds_val"

# --- palette (validated with the dataviz validator; light surface)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
FAMILY_COLOR = {"Lexical (TF-IDF)": C["blue"], "Frozen encoder": C["aqua"], "Fine-tuned encoder": C["orange"], "LLM": C["violet"]}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
                     "legend.frameon": False, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})

PRETTY = {
    "tfidf_word_raw": "TF-IDF word (raw)", "tfidf_word_stem": "TF-IDF word (stem)", "tfidf_word_lemma": "TF-IDF word (lemma)",
    "tfidf_char_raw": "TF-IDF char 2-5", "tfidf_wordchar_raw": "TF-IDF word+char", "tfidf_content_tr": "TF-IDF content (TR)",
    "tfidf_content_en": "TF-IDF content (EN, translated)", "tfidf_subject_raw": "TF-IDF subject only",
    "emb_minilm_multi": "paraphrase-multilingual-MiniLM-L12", "emb_mpnet_multi": "paraphrase-multilingual-mpnet-base",
    "emb_distiluse_multi": "distiluse-base-multilingual (mUSE)", "emb_e5_base": "multilingual-e5-base", "emb_e5_large": "multilingual-e5-large",
    "emb_labse": "LaBSE", "emb_sbert_tr": "BERTurk-SBERT (NLI/STS-tr)", "emb_berturk_mean": "BERTurk mean-pooled (frozen)",
    "emb_bge_m3": "BGE-M3", "emb_mpnet_en": "all-mpnet-base-v2 (EN, translated)", "emb_qwen3_0_6b": "Qwen3-Embedding-0.6B",
    "emb_qwen3_4b": "Qwen3-Embedding-4B", "ft_bert-base-turkish-cased": "BERTurk fine-tuned", "ft_xlm-roberta-base": "XLM-R base fine-tuned",
}


SUFFIX = {"content_clean": "content only, TR", "content_en_clean": "content only, EN translated"}
LLM_NAMES = {"llm_claude-haiku-5-5": "Claude Haiku 5.5", "llm_claude-sonnet-5-5": "Claude Sonnet 5.5", "llm_gpt-6-luna": "GPT-6 Luna",
             "llm_gemini-3.8-flash": "Gemini 3.8 Flash"}
FAMILY_LABEL = {"rag10": "RAG-10", "zeroshot": "zero-shot", "FT_s0": "seed 0", "FT_s1": "seed 1", "FT_s2": "seed 2"}


def fam_label(f):
    return FAMILY_LABEL.get(str(f), str(f))


def llm_label(key):
    """'llm_gemini-3.8-flash__rag10' -> 'Gemini 3.8 Flash / RAG-10'"""
    rep, _, fam = key.partition("__")
    return LLM_NAMES.get(rep, rep) + (" / " + fam_label(fam) if fam else "")


def pretty(rep):
    extra = ""
    if "+sender" in rep:
        rep = rep.replace("+sender", ""); extra = " + sender unit"
    base, _, suffix = rep.partition("__")
    name = PRETTY.get(base, LLM_NAMES.get(base, base.replace("llm_", "LLM: ")))
    if suffix:
        name += " [" + SUFFIX.get(suffix, suffix) + "]"
    return name + extra


def family_of(rep):
    if rep.startswith("tfidf"):
        return "Lexical (TF-IDF)"
    if rep.startswith("emb"):
        return "Frozen encoder"
    if rep.startswith("ft_"):
        return "Fine-tuned encoder"
    return "LLM"


ABLATION_REPS = {"tfidf_subject_raw", "tfidf_content_tr", "tfidf_content_en"}


def is_ablation(rep):
    """Input/context ablations are reported separately from the main comparison
    (main = subject + body as input, no sender information)."""
    return ("+sender" in rep) or ("__content" in rep) or (rep in ABLATION_REPS)


def load_results():
    df = load_all_results()
    # pilot rows: linear SVM on dense embeddings (MiniLM, distiluse) before it was restricted to sparse TF-IDF
    df = df[~(df["representation"].str.startswith("emb_") & (df["family"] == "LinearSVM"))].reset_index(drop=True)
    df["system"] = df["representation"].map(pretty) + " / " + df["family"].map(fam_label)
    df["group"] = df["representation"].map(family_of)
    return df


def val_selected_family(rep):
    """Family chosen on the validation split (best val macro-F1 over the whole grid)."""
    best, fam = -1, None
    for f in (RESULTS / "val_grid").glob(f"{rep}__*.csv"):
        g = pd.read_csv(f)
        if g["macro_f1"].max() > best:
            best, fam = g["macro_f1"].max(), f.stem.split("__")[-1]
    return fam, best


def llm_val_macro_f1(model, mode, docs, labels):
    """Validation macro-F1 of an LLM configuration (None if the validation run is incomplete)."""
    from sklearn.metrics import f1_score
    from llm import final_answers
    recs = final_answers(model, mode, "val", labels)
    if recs is None:
        return None
    v = docs[docs["split"] == "val"]
    if set(v["doc_id"]) - set(recs):
        return None
    fallback = docs.loc[docs["split"] != "test", "label"].value_counts().index[0]
    pred = [(recs[d] or [fallback])[0] for d in v["doc_id"]]
    return float(f1_score(v["label"], pred, labels=labels, average="macro", zero_division=0))


def selected_systems(df, docs=None, labels=None):
    """One row per representation: the classifier family selected on validation.
    Fine-tuned encoders: seed-averaged row (predictions of seed 0 are used for tests), validation
    macro-F1 of the selected epoch, grouped CV from ft_runs/.  LLMs: one row per mode."""
    rows = []
    for rep, g in df.groupby("representation", sort=False):
        if rep.startswith("ft_"):
            r = g.iloc[0].copy()
            for c in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "top3_accuracy", "lenient_accuracy", "department_accuracy"]:
                r[c] = g[c].mean(); r[c + "_sd"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
            r["family"] = "mean of %d seeds" % len(g); r["pred_key"] = f"{rep}__FT_s0"
            sel_f = FT_RUNS / f"{rep}__selection.json"
            if sel_f.exists():
                r["val_macro_f1"] = json.loads(sel_f.read_text())["best"]["val_macro_f1"]
            cv_f = FT_RUNS / f"{rep}__cv_summary.json"
            if cv_f.exists():
                for k, v in json.loads(cv_f.read_text()).items():
                    r[k] = v
            rows.append(r)
        elif rep.startswith("llm_"):
            for _, r in g.iterrows():
                r = r.copy(); r["pred_key"] = f"{rep}__{r['family']}"
                if docs is not None:
                    r["val_macro_f1"] = llm_val_macro_f1(rep[4:], r["family"], docs, labels)
                rows.append(r)
        else:
            if "selected" in g.columns and g["selected"].fillna(False).astype(bool).any():
                r = g[g["selected"].fillna(False).astype(bool)].iloc[0].copy(); fam = r["family"]
            else:
                fam, vf1 = val_selected_family(rep)
                if fam is None or fam not in set(g["family"]):
                    fam = g.sort_values("macro_f1").iloc[-1]["family"]
                r = g[g["family"] == fam].iloc[0].copy(); r["val_macro_f1"] = vf1
            r["pred_key"] = f"{rep}__{fam}"
            rows.append(r)
    out = pd.DataFrame(rows)
    out["system"] = out["representation"].map(pretty) + " / " + out["family"].map(fam_label)
    out["group"] = out["representation"].map(family_of)
    out["ablation"] = out["representation"].map(is_ablation)
    if "val_macro_f1" not in out.columns:
        out["val_macro_f1"] = np.nan
    out["val_macro_f1"] = pd.to_numeric(out["val_macro_f1"], errors="coerce")
    return out.reset_index(drop=True)


def representatives(sel):
    """Per family, the main-comparison system with the highest VALIDATION macro-F1; the overall
    final system is the representative with the highest validation macro-F1.  No test score is
    used for these choices."""
    m = sel[~sel["ablation"]]
    reps = []
    for grp in ["Lexical (TF-IDF)", "Frozen encoder", "Fine-tuned encoder", "LLM"]:
        g = m[(m["group"] == grp) & m["val_macro_f1"].notna()]
        if len(g):
            reps.append(g.sort_values("val_macro_f1").iloc[-1])
    R = pd.DataFrame(reps).sort_values("val_macro_f1", ascending=False).reset_index(drop=True) if reps else m.head(0)
    return R


def load_pred(key, val=False):
    f = (PREDS_VAL if val else PREDS) / f"{key}.pkl"
    return pd.read_pickle(f) if f.exists() else None


def confidence(p):
    """Confidence score for deferral: max probability when available, otherwise the
    softmax of the decision values (monotone in the top-1 margin)."""
    s = np.asarray(p["scores"], dtype=float)
    if s.min() >= 0 and np.allclose(s.sum(1), 1, atol=1e-3):
        return s.max(1)
    z = s - s.max(1, keepdims=True); e = np.exp(z); return (e / e.sum(1, keepdims=True)).max(1)


def risk_coverage(conf, correct):
    order = np.argsort(-conf); c = correct[order]
    cov = np.arange(1, len(c) + 1) / len(c); acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    aurc = float(np.mean(1 - acc))
    return cov, acc, aurc


def threshold_for_target(conf_val, correct_val, target):
    """Largest coverage on validation whose accuracy on the covered subset >= target; returns the threshold."""
    order = np.argsort(-conf_val); c = correct_val[order]; acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(acc >= target)[0]
    if len(ok) == 0:
        return np.inf
    return float(conf_val[order][ok[-1]])


# ----------------------------------------------------------------------------- tables
def build_tables(docs, labels, df, sel):
    TAB.mkdir(parents=True, exist_ok=True)
    idx = split_indices(docs); y_test = docs["label"].values[idx["test"]]
    tables = {}
    # dataset
    st = json.loads((RESULTS / "dataset_stats.json").read_text())
    cc = pd.DataFrame({"unit": labels})
    cc["department"] = cc["unit"].map(PARENT); cc["short"] = cc["unit"].map(SHORT)
    for s in ["train", "val", "test"]:
        cc[s] = cc["unit"].map(st["split_class_counts"][s]).fillna(0).astype(int)
    cc["total"] = cc[["train", "val", "test"]].sum(axis=1); cc["share_%"] = (100 * cc["total"] / cc["total"].sum()).round(1)
    tables["dataset_classes"] = cc.sort_values("total", ascending=False)
    tables["dataset_stats"] = pd.DataFrame([{k: v for k, v in st.items() if not isinstance(v, dict)}]).T.reset_index().rename(columns={"index": "statistic", 0: "value"})
    # main + full
    cols = ["system", "group", "accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci", "weighted_f1", "balanced_accuracy", "top3_accuracy",
            "lenient_accuracy", "department_accuracy", "cv_accuracy_mean", "cv_accuracy_sd", "cv_macro_f1_mean", "cv_macro_f1_sd",
            "fit_seconds_trainval", "predict_ms_per_doc", "n_features", "params"]
    cols = cols[:2] + ["val_macro_f1"] + cols[2:]
    tables["main"] = sel.loc[~sel["ablation"], [c for c in cols if c in sel.columns]].sort_values("macro_f1", ascending=False)
    tables["ablations"] = sel.loc[sel["ablation"], [c for c in cols if c in sel.columns]].sort_values(["group", "system"])
    tables["all_configurations"] = df[[c for c in cols if c in df.columns]].sort_values(["group", "system"])
    # family representatives and the final system are chosen on validation only
    R = representatives(sel)
    tables["representatives"] = R[[c for c in cols if c in R.columns]] if len(R) else R
    top = R
    pcs = []
    for _, r in top.iterrows():
        p = load_pred(sel.loc[sel["system"] == r["system"], "pred_key"].iloc[0])
        if p is None:
            continue
        t = per_class_table(p["y_true"], p["y_pred"], labels); t.insert(0, "system", r["system"]); pcs.append(t)
    if pcs:
        pc = pd.concat(pcs); pc["train_support"] = pc["unit"].map(st["split_class_counts"]["train"]).fillna(0).astype(int)
        tables["per_class"] = pc
    # McNemar among the top systems
    keys = [(r["system"], sel.loc[sel["system"] == r["system"], "pred_key"].iloc[0]) for _, r in top.iterrows()]
    preds = {s: load_pred(k) for s, k in keys}
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = preds[keys[i][0]], preds[keys[j][0]]
            if a is None or b is None:
                continue
            res = mcnemar_test(a["y_true"], a["y_pred"], b["y_pred"])
            rows.append({"system_a": keys[i][0], "system_b": keys[j][0], "acc_a": float((a["y_pred"] == a["y_true"]).mean()),
                         "acc_b": float((b["y_pred"] == b["y_true"]).mean()), **res})
    if rows:
        mc = pd.DataFrame(rows)
        # Holm correction
        order = np.argsort(mc["p_value"].values); m = len(mc); adj = np.empty(m)
        running = 0
        for rank, k in enumerate(order):
            running = max(running, (m - rank) * mc["p_value"].values[k]); adj[k] = min(1, running)
        mc["p_holm"] = adj; tables["mcnemar"] = mc
    # error analysis of the final system (highest validation macro-F1)
    best_sys = R.iloc[0]["system"] if len(R) else tables["main"].iloc[0]["system"]
    bp = preds.get(best_sys) if best_sys in preds else load_pred(sel.loc[sel["system"] == best_sys, "pred_key"].iloc[0])
    if bp is not None:
        yt, yp = np.asarray(bp["y_true"]), np.asarray(bp["y_pred"])
        err = yt != yp
        same_dept = np.array([PARENT[a] == PARENT[b] for a, b in zip(yt, yp)])
        ea = {"system": best_sys, "selected_on": "validation macro-F1", "n_test": len(yt), "n_errors": int(err.sum()), "errors_within_department": int((err & same_dept).sum()),
              "errors_across_departments": int((err & ~same_dept).sum()), "department_accuracy": float(same_dept.mean())}
        pairs = pd.Series([f"{SHORT[a]} -> {SHORT[b]}" for a, b in zip(yt[err], yp[err])]).value_counts().head(12)
        tables["error_summary"] = pd.DataFrame([ea]); tables["confused_pairs"] = pairs.rename_axis("true -> predicted").reset_index(name="count")
    # deferral operating points
    rows = []
    for _, r in top.iterrows():
        key = sel.loc[sel["system"] == r["system"], "pred_key"].iloc[0]
        pt, pv = load_pred(key), load_pred(key, val=True)
        if pt is None or pv is None:
            continue
        ct, cv_ = confidence(pt), confidence(pv)
        corr_t = np.asarray(pt["y_pred"]) == np.asarray(pt["y_true"]); corr_v = np.asarray(pv["y_pred"]) == np.asarray(pv["y_true"])
        cov, acc, aurc = risk_coverage(ct, corr_t)
        rec = {"system": r["system"], "AURC": aurc, "accuracy_full": float(corr_t.mean())}
        for tgt in (0.90, 0.95):
            thr = threshold_for_target(cv_, corr_v, tgt); m = ct >= thr
            rec[f"coverage@{int(tgt*100)}"] = float(m.mean()); rec[f"accuracy@{int(tgt*100)}"] = float(corr_t[m].mean()) if m.any() else np.nan
            rec[f"threshold@{int(tgt*100)}"] = thr
        rows.append(rec)
    if rows:
        tables["deferral"] = pd.DataFrame(rows)
    # LLM cascade (local -> LLM) if LLM predictions exist
    llm_keys = sorted(p.stem for p in PREDS.glob("llm_*__rag*.pkl"))
    if llm_keys and rows:
        crows = []
        for _, r in top.iterrows():
            key = sel.loc[sel["system"] == r["system"], "pred_key"].iloc[0]
            pt, pv = load_pred(key), load_pred(key, val=True)
            if pt is None or pv is None:
                continue
            for lk in llm_keys:
                from llm import final_answers
                lt = load_pred(lk)
                model, mode = lk[4:].split("__")
                lv = final_answers(model, mode, "val", labels)
                if lt is None or lv is None:
                    continue
                if len(set(pv["doc_id"]) - set(lv)) > 0:
                    continue
                fallback = docs.loc[docs["split"] != "test", "label"].value_counts().index[0]
                llm_val_pred = np.array([(lv[d] or [fallback])[0] for d in pv["doc_id"]])
                llm_test_pred = pd.Series(lt["y_pred"], index=lt["doc_id"]).reindex(pt["doc_id"]).values
                cv_, ct = confidence(pv), confidence(pt)
                yv, ytt = np.asarray(pv["y_true"]), np.asarray(pt["y_true"])
                # validation: accuracy as a function of the LLM share
                order = np.argsort(cv_); shares = np.linspace(0, 1, 101)
                def cascade_acc(share, conf, local, llm, ytrue):
                    k = int(round(share * len(conf)))
                    low = np.argsort(conf)[:k]; pred = local.copy(); pred[low] = llm[low]
                    return float((pred == ytrue).mean())
                val_curve = [cascade_acc(s, cv_, np.asarray(pv["y_pred"]), llm_val_pred, yv) for s in shares]
                llm_alone_val = float((llm_val_pred == yv).mean())
                within = [s for s, a in zip(shares, val_curve) if a >= llm_alone_val - 0.01]
                s_min = within[0] if within else 1.0
                s_best = shares[int(np.argmax(val_curve))]
                for name_, s in (("min share within 1pt of LLM", s_min), ("best val accuracy", s_best)):
                    acc_t = cascade_acc(s, ct, np.asarray(pt["y_pred"]), llm_test_pred, ytt)
                    crows.append({"local_system": r["system"], "llm": lk, "rule": name_, "llm_share": s, "test_accuracy": acc_t,
                                  "local_alone": float((pt["y_pred"] == ytt).mean()), "llm_alone": float((llm_test_pred == ytt).mean())})
        if crows:
            tables["cascade"] = pd.DataFrame(crows)
    # lexical variants and all ablations (selected classifier per representation)
    keep = ["system", "representation", "accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "cv_accuracy_mean", "cv_accuracy_sd"]
    tables["ablation_lexical"] = sel[sel["representation"].str.startswith("tfidf")][keep]
    tables["ablation_other"] = sel[sel["ablation"] & ~sel["representation"].str.startswith("tfidf")][keep]
    # cost
    tables["cost"] = sel[["system", "group", "fit_seconds_trainval", "predict_ms_per_doc", "n_features"] + [c for c in ["embedding_dim", "usd_per_1000_docs"] if c in sel.columns]]
    with pd.ExcelWriter(RESULTS / "tables.xlsx") as xw:
        for k, t in tables.items():
            t.to_excel(xw, sheet_name=k[:31], index=False); t.to_csv(TAB / f"{k}.csv", index=False)
    return tables


# ----------------------------------------------------------------------------- figures
def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_dataset(docs, labels):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1.4]})
    cc = docs["label"].value_counts()
    names = [SHORT[u] for u in cc.index]
    a1.barh(names[::-1], cc.values[::-1], color=C["blue"], height=0.62)
    for i, v in enumerate(cc.values[::-1]):
        a1.text(v + 25, i, f"{v:,}", va="center", fontsize=7.5, color=C["ink2"])
    a1.set_xlabel("Documents"); a1.set_title("(a) Documents per target unit (n = %s)" % f"{len(docs):,}")
    a1.tick_params(axis="y", length=0); a1.set_xlim(0, cc.max() * 1.15)
    m = docs.set_index("date").resample("MS").size()
    a2.plot(m.index, m.values, color=C["blue"], lw=2)
    for d, lab in [("2022-01-01", "validation"), ("2022-07-01", "test")]:
        a2.axvline(pd.Timestamp(d), color=C["axis"], lw=1); a2.text(pd.Timestamp(d), m.max() * 1.02, " " + lab, fontsize=7.5, color=C["ink2"], va="bottom")
    a2.text(pd.Timestamp("2020-01-01"), m.max() * 1.02, " training", fontsize=7.5, color=C["ink2"], va="bottom")
    a2.set_ylim(0, m.max() * 1.15); a2.set_ylabel("Documents per month"); a2.set_title("(b) Monthly volume and chronological split")
    a2.yaxis.grid(True, color=C["grid"], lw=0.6); a2.set_axisbelow(True)
    import matplotlib.dates as mdates
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a2.tick_params(axis="x", labelsize=7.5, rotation=0)
    save(fig, "fig_dataset")


def fig_main(sel):
    d = sel.sort_values("macro_f1").reset_index(drop=True)
    h = 0.32 * len(d) + 1.2
    fig, axes = plt.subplots(1, 2, figsize=(10, h), sharey=True)
    for ax, met, title in [(axes[0], "accuracy", "(a) Accuracy"), (axes[1], "macro_f1", "(b) Macro-F1")]:
        for i, r in d.iterrows():
            lo, hi = json.loads(r[met + "_ci"]) if isinstance(r[met + "_ci"], str) else r[met + "_ci"]
            col = FAMILY_COLOR[r["group"]]
            ax.plot([lo, hi], [i, i], color=col, lw=1.5, alpha=0.8)
            ax.plot(r[met], i, "o", color=col, ms=5.5, mec="white", mew=0.8)
            if not pd.isna(r.get("cv_" + met + "_mean", np.nan)):
                ax.plot(r["cv_" + met + "_mean"], i, "|", color=C["muted"], ms=9, mew=1.2)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["system"], fontsize=7.5)
        ax.xaxis.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True); ax.set_title(title); ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", color=c, ls="", label=k) for k, c in FAMILY_COLOR.items() if k in set(d["group"])]
    handles.append(plt.Line2D([], [], marker="|", color=C["muted"], ls="", label="grouped 5x2 CV mean"))
    axes[1].legend(handles=handles, loc="lower right", fontsize=7.5)
    fig.text(0.5, -0.01, "Chronological test split (second half of 2022, n = 1,653). Dots: point estimate; bars: 95% bootstrap CI.", ha="center", fontsize=7.5, color=C["ink2"])
    save(fig, "fig_main")


def fig_confusion(labels, pred, title):
    from sklearn.metrics import confusion_matrix
    order = sorted(labels, key=lambda u: (["BİDB", "SKS", "Rektörlük"].index(PARENT[u]), SHORT[u]))
    cm = confusion_matrix(pred["y_true"], pred["y_pred"], labels=order)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", ["#ffffff"] + SEQ)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.5, color="white" if norm[i, j] > 0.55 else C["ink"])
    names = [SHORT[u] for u in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{n} ({cm[i].sum()})" for i, n in enumerate(names)], fontsize=7.5)
    ax.set_xlabel("Predicted unit"); ax.set_ylabel("True unit (test support)")
    # department bands: frames on the diagonal, names in a bracket to the right of the matrix (outside the cells)
    deps = [PARENT[u] for u in order]; start = 0
    dep_name = {"BİDB": "BİDB", "SKS": "SKS", "Rektörlük": "Rectorate"}
    x_br = len(order) - 0.5 + 0.25
    for k in range(1, len(order) + 1):
        if k == len(order) or deps[k] != deps[start]:
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), k - start, k - start, fill=False, ec=C["orange"], lw=1.2))
            ax.plot([x_br, x_br], [start - 0.4, k - 0.6], color=C["orange"], lw=1.2, clip_on=False, solid_capstyle="butt")
            ax.text(x_br + 0.2, (start + k - 1) / 2, dep_name.get(deps[start], deps[start]), ha="left", va="center", fontsize=7,
                    color=C["orange"], fontweight="bold", clip_on=False)
            start = k
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_title(title, fontsize=9)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    save(fig, "fig_confusion")


def fig_perclass(tables, labels):
    pc = tables.get("per_class")
    if pc is None:
        return
    systems = list(dict.fromkeys(pc["system"]))[:4]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    grp = dict(zip(tables["representatives"]["system"], tables["representatives"]["group"])) if "representatives" in tables else {}
    cols = [FAMILY_COLOR.get(grp.get(s_), c_) for s_, c_ in zip(systems, [C["blue"], C["orange"], C["aqua"], C["violet"]])]
    w = 0.8 / len(systems); offs = [(i - (len(systems) - 1) / 2) * w for i in range(len(systems))]
    for (s, col), off in zip(zip(systems, cols), offs):
        t = pc[pc["system"] == s].set_index("unit").loc[sorted(labels, key=lambda u: -pc[(pc.system == s) & (pc.unit == u)]["train_support"].iloc[0])]
        x = np.arange(len(t)) + off
        ax.bar(x, t["f1"], width=w, color=col, label=s.replace("mean of 3 seeds", "seed 0"))
    t0 = pc[pc["system"] == systems[0]].set_index("unit").loc[sorted(labels, key=lambda u: -pc[(pc.system == systems[0]) & (pc.unit == u)]["train_support"].iloc[0])]
    ax.set_xticks(range(len(t0))); ax.set_xticklabels([f"{SHORT[u]}\n(n={n})" for u, n in zip(t0.index, t0["train_support"])], fontsize=6.8, rotation=35, ha="right")
    ax.set_ylabel("F1 (test)"); ax.set_ylim(0, 1); ax.yaxis.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper right"); ax.set_title("Per-unit F1 ordered by training support (n = training documents)")
    save(fig, "fig_perclass")


def fig_deferral(sel, tables, labels):
    top = tables["representatives"]
    top = top[top["group"] != "LLM"] if len(top) else top
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.28})
    ax = axes[0]; cols = [C["blue"], C["orange"], C["aqua"], C["violet"]]
    for (_, r), col in zip(top.iterrows(), cols):
        key = sel.loc[sel["system"] == r["system"], "pred_key"].iloc[0]
        pt = load_pred(key)
        if pt is None:
            continue
        cov, acc, aurc = risk_coverage(confidence(pt), np.asarray(pt["y_pred"]) == np.asarray(pt["y_true"]))
        ax.plot(cov * 100, acc * 100, color=col, lw=2, label=f"{r['system'].replace('mean of 3 seeds', 'seed 0')} (AURC {aurc:.3f})")   # curve of one seed
    for tgt in (90, 95):
        ax.axhline(tgt, color=C["axis"], lw=0.8); ax.text(99, tgt + 0.6, f"{tgt}% target", fontsize=7, color=C["ink2"], ha="right")
    ax.set_xlabel("Coverage: documents routed automatically (%)"); ax.set_ylabel("Accuracy on routed documents (%)")
    ax.set_title("(a) Confidence-based deferral to staff"); ax.legend(fontsize=6.8, loc="lower left", bbox_to_anchor=(0.0, 0.0)); ax.set_ylim(30, 101); ax.yaxis.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, 100)
    ax2 = axes[1]
    casc = tables.get("cascade")
    if casc is not None and len(casc):
        best_local = top.iloc[0]["system"]
        key = sel.loc[sel["system"] == best_local, "pred_key"].iloc[0]; pt = load_pred(key); ct = confidence(pt)
        shares = np.linspace(0, 1, 101); all_accs = []
        for lk, col in zip(sorted(casc["llm"].unique()), cols):
            lt = load_pred(lk); llm_test = pd.Series(lt["y_pred"], index=lt["doc_id"]).reindex(pt["doc_id"]).values
            accs = []
            for s in shares:
                k = int(round(s * len(ct))); low = np.argsort(ct)[:k]; pred = np.asarray(pt["y_pred"]).copy(); pred[low] = llm_test[low]
                accs.append((pred == np.asarray(pt["y_true"])).mean() * 100)
            ax2.plot(shares * 100, accs, color=col, lw=2, label=llm_label(lk)); all_accs += accs
        lo, hi = min(all_accs), max(all_accs)
        ax2.set_ylim(lo - 0.5 * (hi - lo), hi + 0.08 * (hi - lo))      # room below the curves for the legend
        ax2.set_xlabel("Share of documents sent to the LLM (%)"); ax2.set_ylabel("Cascade accuracy (%)")
        ax2.set_title(f"(b) Cascade: {best_local.split(' / ')[0]} \u2192 LLM"); ax2.legend(fontsize=6.8, loc="lower right"); ax2.yaxis.grid(True, color=C["grid"], lw=0.6); ax2.set_axisbelow(True)
    else:
        ax2.text(0.5, 0.5, "LLM cascade: run llm.py first", ha="center", va="center", color=C["muted"]); ax2.set_axis_off()
    save(fig, "fig_deferral")


def fig_ablation(sel):
    rows = [("Preprocessing", ["tfidf_word_raw", "tfidf_word_stem", "tfidf_word_lemma", "tfidf_char_raw", "tfidf_wordchar_raw"]),
            ("Input field", ["tfidf_subject_raw", "tfidf_content_tr", "tfidf_word_raw"]),
            ("Translation (content only)", ["tfidf_content_tr", "tfidf_content_en", "emb_mpnet_multi__content_clean", "emb_mpnet_en__content_en_clean"]),
            ("Sender-unit context", [r for r in sel["representation"] if r.endswith("+sender")] + [r.replace("+sender", "") for r in sel["representation"] if r.endswith("+sender")])]
    rows = [(t, [r for r in reps if r in set(sel["representation"])]) for t, reps in rows]
    rows = [(t, reps) for t, reps in rows if reps]
    if not rows:
        return
    import textwrap
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows) + 1, 3.4), squeeze=False)
    fig.subplots_adjust(wspace=1.1)
    for ax, (title, reps) in zip(axes[0], rows):
        d = sel.set_index("representation").loc[reps]
        y = np.arange(len(d))
        ax.barh(y + 0.18, d["accuracy"] * 100, height=0.34, color=C["blue"], label="Accuracy")
        ax.barh(y - 0.18, d["macro_f1"] * 100, height=0.34, color=C["aqua"], label="Macro-F1")
        for yy, (a, f) in zip(y, zip(d["accuracy"], d["macro_f1"])):
            ax.text(a * 100 + 0.5, yy + 0.18, f"{a*100:.1f}", va="center", fontsize=6.5, color=C["ink2"])
            ax.text(f * 100 + 0.5, yy - 0.18, f"{f*100:.1f}", va="center", fontsize=6.5, color=C["ink2"])
        ax.set_yticks(y); ax.set_yticklabels([textwrap.fill(pretty(r), 22) for r in reps], fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 100)
        ax.set_title(title); ax.tick_params(axis="y", length=0); ax.xaxis.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    axes[0][0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig_ablation")


def fig_pipeline():
    dot = r"""
digraph G {
  rankdir=TB; nodesep=0.3; ranksep=0.45; bgcolor="white"; newrank=true; compound=true;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#c3c2b7", penwidth=1, margin="0.12,0.06"];
  edge [color="#52514e", arrowsize=0.7, fontname="Helvetica", fontsize=9];
  subgraph cluster_data { label="Data layer"; fontname="Helvetica-Bold"; fontsize=11; color="#e1e0d9"; style="rounded"; labeljust="l";
    { rank=same; raw [label="EDMS dispatch log\n33,828 records (2020-2022)", fillcolor="#eef4fc"];
      doc [label="Document table\n9,345 documents, 13 units\n(first dispatch = label)", fillcolor="#eef4fc"]; }
    { rank=same; deid [label="De-identification\nnames, e-mail, phone, ID numbers\n-> placeholders (NER + patterns)", fillcolor="#eef4fc"];
      split [label="Chronological partitions\ntrain / validation / test\n+ grouped 5x2 CV", fillcolor="#eef4fc"]; }
    raw -> doc; doc -> deid; deid -> split; }
  subgraph cluster_rep { label="Representation layer"; fontname="Helvetica-Bold"; fontsize=11; color="#e1e0d9"; style="rounded"; rank=same; labeljust="l";
    lex [label="Lexical\nTF-IDF word / character n-grams\n(raw, stem, lemma)", fillcolor="#e8f6f0"];
    emb [label="Frozen sentence encoders\nmultilingual and Turkish", fillcolor="#e8f6f0"];
    ft  [label="Fine-tuned encoders\nBERTurk, XLM-R", fillcolor="#e8f6f0"];
  }
  subgraph cluster_cls { label="Decision layer"; fontname="Helvetica-Bold"; fontsize=11; color="#e1e0d9"; style="rounded"; rank=same; labeljust="r";
    clf [label="Classifier grid\nLR, SVM, KNN, RF, NB, DT\n(selected on validation)", fillcolor="#fdeee8"];
    conf [label="Ranked units + confidence\n(top-1 probability)", fillcolor="#fdeee8"];
  }
  subgraph cluster_route { label="Routing layer"; fontname="Helvetica-Bold"; fontsize=11; color="#e1e0d9"; style="rounded"; rank=same; labeljust="l";
    auto [label="Automatic routing\n(top-1; top-3 as suggestion)", fillcolor="#f3f0fb"];
    llm [label="LLM second stage\n(retrieval-augmented,\nexternal API)", fillcolor="#f3f0fb"];
    human [label="Staff review\n(top-3 suggestion)", fillcolor="#f3f0fb"];
  }
  split -> lex; split -> emb; split -> ft;
  lex -> clf; emb -> clf; clf -> conf; ft -> conf [label="softmax"];
  conf -> auto [label="confidence >= threshold"]; conf -> llm [label="low confidence"]; conf -> human [style=dashed, label="very low"];
  llm -> human [label="residual"];
}"""
    FIG.mkdir(exist_ok=True)
    (FIG / "fig_pipeline.dot").write_text(dot)
    for fmt in ("png", "pdf"):
        try:
            subprocess.run(["dot", f"-T{fmt}", "-Gdpi=250" if fmt == "png" else "-Gdpi=72", str(FIG / "fig_pipeline.dot"), "-o", str(FIG / f"fig_pipeline.{fmt}")], check=False)
        except FileNotFoundError:
            print("graphviz 'dot' not found - fig_pipeline skipped"); return


def main():
    docs, labels = load_docs()
    df = load_results(); sel = selected_systems(df, docs, labels)
    tables = build_tables(docs, labels, df, sel)
    fig_dataset(docs, labels); fig_main(sel[~sel["ablation"]]); fig_perclass(tables, labels); fig_deferral(sel, tables, labels)
    fig_ablation(sel); fig_pipeline()
    R = tables["representatives"]
    final = R.iloc[0]["system"] if len(R) else tables["main"].iloc[0]["system"]
    bp = load_pred(sel.loc[sel["system"] == final, "pred_key"].iloc[0])
    if bp is not None:
        fig_confusion(labels, bp, f"Confusion matrix on the test split: {final}")
    print(tables["main"][["system", "val_macro_f1", "accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "cv_accuracy_mean"]].round(4).to_string())
    if len(R):
        print("\nfamily representatives (chosen on validation):")
        print(R[["system", "val_macro_f1", "accuracy", "macro_f1"]].round(4).to_string())
    print("\nfigures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()

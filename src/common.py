"""
common.py
---------
Shared evaluation protocol for every representation (TF-IDF, frozen embeddings,
fine-tuned transformers, LLM predictions).

Protocol
  * Model selection: fit on `train`, select hyper-parameters on `val` (chronological
    validation split) by macro-F1.
  * Final estimate: refit the selected configuration on train+val, predict `test`
    (second half of 2022) exactly once.  Per-document predictions and class scores
    are stored so that top-k routing, confidence-based deferral, McNemar tests and
    error analyses can be computed afterwards without re-training.
  * Robustness: 5-fold StratifiedGroupKFold (groups = identical texts), two repeats,
    with the selected configuration -> mean +/- sd over 10 folds.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
FT_RUNS = RESULTS / "ft_runs_v2"       # fine-tuning runs (v2: dynamic padding, hardware-appropriate precision)
SEEDS = (0, 1)

PARENT = {
    "Bilgi İşlem Daire Başkanlığı": "BİDB", "Yazılım Şube Müdürlüğü": "BİDB",
    "Bilgi İşlem Şube Müdürlüğü": "BİDB", "İnternet Hizmetleri Şube Müdürlüğü": "BİDB",
    "Haberleşme Hizmetleri Şube Müdürlüğü": "BİDB",
    "Sağlık Kültür ve Spor Daire Başkanlığı": "SKS", "Kültür Hizmetleri Şube Müdürlüğü": "SKS",
    "Satın Alma Şube Müdürlüğü": "SKS", "Sosyal Tesisler Şube Müdürlüğü": "SKS",
    "Beslenme Hizmetleri Şube Müdürlüğü": "SKS", "Spor Tesisleri Şube Müdürlüğü": "SKS",
    "Sağlık Hizmetleri Şube Müdürlüğü": "SKS", "Rektörlük": "Rektörlük",
}
SHORT = {
    "Bilgi İşlem Daire Başkanlığı": "BİDB (dept.)", "Yazılım Şube Müdürlüğü": "Software",
    "Bilgi İşlem Şube Müdürlüğü": "IT Services", "İnternet Hizmetleri Şube Müdürlüğü": "Internet Services",
    "Haberleşme Hizmetleri Şube Müdürlüğü": "Communications",
    "Sağlık Kültür ve Spor Daire Başkanlığı": "SKS (dept.)", "Kültür Hizmetleri Şube Müdürlüğü": "Cultural Services",
    "Satın Alma Şube Müdürlüğü": "Procurement", "Sosyal Tesisler Şube Müdürlüğü": "Social Facilities",
    "Beslenme Hizmetleri Şube Müdürlüğü": "Catering", "Spor Tesisleri Şube Müdürlüğü": "Sports Facilities",
    "Sağlık Hizmetleri Şube Müdürlüğü": "Health Services", "Rektörlük": "Rectorate",
}


# ----------------------------------------------------------------------------- data
def load_docs():
    docs = pd.read_pickle(ROOT / "data" / "documents.pkl")
    labels = sorted(docs["label"].unique())
    return docs, labels


def load_variants():
    return pd.read_pickle(ROOT / "data" / "text_variants.pkl")


def split_indices(docs):
    idx = {s: np.where(docs["split"].values == s)[0] for s in ("train", "val", "test")}
    idx["trainval"] = np.concatenate([idx["train"], idx["val"]])
    return idx


def cv_folds(docs):
    """List of (train_idx, test_idx) for 5-fold x 2 repeats grouped stratified CV."""
    folds = []
    for seed in SEEDS:
        f = docs[f"fold_s{seed}"].values
        for k in range(5):
            folds.append((np.where(f != k)[0], np.where(f == k)[0]))
    return folds


# ----------------------------------------------------------------------------- metrics
def topk_hits(scores, y_idx, k):
    top = np.argsort(-scores, axis=1)[:, :k]
    return (top == y_idx[:, None]).any(axis=1)


def metrics(y_true, y_pred, labels, scores=None, label_sets=None):
    """y_true/y_pred: label strings.  scores: (n, n_classes) in `labels` order."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)[0],
        "macro_recall": precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)[1],
    }
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    dep_t = np.array([PARENT[x] for x in yt]); dep_p = np.array([PARENT[x] for x in yp])
    m["department_accuracy"] = float((dep_t == dep_p).mean())
    if scores is not None:
        y_idx = np.array([labels.index(x) for x in yt])
        m["top2_accuracy"] = float(topk_hits(scores, y_idx, 2).mean())
        m["top3_accuracy"] = float(topk_hits(scores, y_idx, 3).mean())
    if label_sets is not None:
        m["lenient_accuracy"] = float(np.mean([p in s for p, s in zip(yp, label_sets)]))
    return {k: float(v) for k, v in m.items()}


def per_class_table(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"unit": labels, "precision": p, "recall": r, "f1": f, "support": s})


def bootstrap_ci(y_true, y_pred, labels, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true); yp = np.asarray(y_pred); n = len(yt)
    accs, f1s = [], []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        accs.append(accuracy_score(yt[b], yp[b]))
        f1s.append(f1_score(yt[b], yp[b], labels=labels, average="macro", zero_division=0))
    return {"accuracy_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "macro_f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def mcnemar_test(y_true, pred_a, pred_b):
    """Exact McNemar test on paired correctness (Dietterich, 1998)."""
    from statsmodels.stats.contingency_tables import mcnemar
    a = np.asarray(pred_a) == np.asarray(y_true); b = np.asarray(pred_b) == np.asarray(y_true)
    tbl = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    res = mcnemar(tbl, exact=True)
    return {"n_a_only": tbl[0][1], "n_b_only": tbl[1][0], "statistic": float(res.statistic), "p_value": float(res.pvalue)}


# ----------------------------------------------------------------------------- classifiers
def classifier_grid(dense):
    """Hyper-parameter grid per classifier family.  Dense features (embeddings) are
    standardised for the linear / kernel models; sparse TF-IDF rows are already L2-normalised."""
    def lin(est):
        return make_pipeline(StandardScaler(), est) if dense else est
    grid = {
        "LR": [(dict(C=c, class_weight=cw), lin(LogisticRegression(C=c, class_weight=cw, max_iter=5000)))
               for c in (0.1, 1, 10) for cw in (None, "balanced")],
        "LinearSVM": [(dict(C=c, class_weight=cw), lin(LinearSVC(C=c, class_weight=cw, max_iter=20000)))
                      for c in (0.1, 0.5, 1, 5) for cw in (None, "balanced")],
        "KNN": [(dict(k=k, metric="cosine", weights="distance"), KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance"))
                for k in (1, 5, 15)],
        "RF": [(dict(n_estimators=200, max_features="sqrt", min_samples_leaf=l, class_weight=cw),
                RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=l, class_weight=cw, n_jobs=-1, random_state=0))
               for l in (1, 3) for cw in (None, "balanced_subsample")],
        "DT": [(dict(min_samples_leaf=l), DecisionTreeClassifier(min_samples_leaf=l, random_state=0)) for l in (1, 5)],
    }
    if dense:
        # linear SVM on sparse TF-IDF only: on standardised dense embeddings liblinear needed up to
        # ~20 min per fit (C=5) and never beat LR or the RBF-SVM on the first two encoders (pilot)
        grid.pop("LinearSVM")
        # the RBF kernel is evaluated on dense embeddings only; on sparse n-gram spaces the
        # linear kernel is the standard (and here much cheaper) choice
        grid["RBF-SVM"] = [(dict(C=c, gamma="scale", class_weight=cw), lin(SVC(C=c, gamma="scale", class_weight=cw)))
                           for c in (1, 10) for cw in (None, "balanced")]
        grid["NB"] = [(dict(kind="GaussianNB"), GaussianNB())]
    else:
        grid["NB"] = [(dict(kind="ComplementNB", alpha=a), ComplementNB(alpha=a)) for a in (0.1, 0.5, 1.0)]
    return grid


def class_scores(est, X, labels):
    """(n, n_classes) scores aligned with `labels` (probabilities when available)."""
    final = est.steps[-1][1] if hasattr(est, "steps") else est
    if hasattr(final, "predict_proba"):
        s = est.predict_proba(X)
    else:
        s = est.decision_function(X)
        if s.ndim == 1:
            s = np.stack([-s, s], axis=1)
    cls = list(final.classes_)
    out = np.full((X.shape[0], len(labels)), -1e9 if not hasattr(final, "predict_proba") else 0.0)
    for j, c in enumerate(cls):
        out[:, labels.index(c)] = s[:, j]
    return out


# ----------------------------------------------------------------------------- features
class PrecomputedFeatures:
    """Rows of a precomputed matrix (e.g. frozen embeddings); nothing is fitted."""
    def __init__(self, X):
        self.X = X
        self.n_features = X.shape[1]

    def fit(self, idx):
        return self

    def transform(self, idx):
        return self.X[idx]


class TfidfFeatures:
    """TF-IDF fitted on the training rows only (vocabulary and IDF never see held-out text)."""
    def __init__(self, texts, char=False, **kw):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.texts = np.asarray(texts, dtype=object)
        params = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000) if char \
            else dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        params.update(kw)
        self.vec = TfidfVectorizer(**params)
        self.n_features = None

    def fit(self, idx):
        self.vec.fit(self.texts[idx])
        self.n_features = len(self.vec.vocabulary_)
        return self

    def transform(self, idx):
        return self.vec.transform(self.texts[idx])


class UnionFeatures:
    def __init__(self, *feats):
        self.feats = feats
        self.n_features = None

    def fit(self, idx):
        for f in self.feats:
            f.fit(idx)
        self.n_features = sum(f.n_features for f in self.feats)
        return self

    def transform(self, idx):
        return sparse.hstack([f.transform(idx) for f in self.feats]).tocsr()


# ----------------------------------------------------------------------------- protocol
def run_representation(name, feat, docs, labels, dense, families=None, log=print, cv=True, extra_meta=None, results_file="results_main.csv"):
    """Full protocol for one feature extractor `feat` (PrecomputedFeatures / TfidfFeatures).

    1. every classifier family: hyper-parameters selected on `val` (macro-F1), model trained on `train`
    2. every family: refit on train+val, one evaluation on `test` (predictions + class scores stored)
    3. the family with the best validation macro-F1 is the representation's *selected system*;
       the grouped 5x2 CV is run for that family only (variance estimate of the reported system)
    Returns the result rows (one per family); rows are also appended to results/results_main.csv."""
    PREDS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "val_grid").mkdir(parents=True, exist_ok=True)
    (RESULTS / "preds_val").mkdir(parents=True, exist_ok=True)
    idx = split_indices(docs)
    y = docs["label"].values
    label_sets = docs["label_set"].values
    grid = classifier_grid(dense)
    if families:
        grid = {k: v for k, v in grid.items() if k in families}
    # resumable: a representation whose families are all in results_main.csv is skipped
    main_csv = RESULTS / results_file
    if main_csv.exists():
        old = pd.read_csv(main_csv)
        done = set(old.loc[old["representation"] == name, "family"])
        if all(k in done and (PREDS / f"{name}__{k}.pkl").exists() for k in grid):
            log(f"[{name}] already done - skipping")
            return []
    feat.fit(idx["train"]); Xtr, Xva = feat.transform(idx["train"]), feat.transform(idx["val"])
    feat.fit(idx["trainval"]); Xtv, Xte = feat.transform(idx["trainval"]), feat.transform(idx["test"])
    n_features = feat.n_features
    selected = {}
    # --- 1. selection on the chronological validation split
    for fam, configs in grid.items():
        best, val_rows = None, []
        for params, est in configs:
            e = clone(est)
            t0 = time.time(); e.fit(Xtr, y[idx["train"]]); fit_t = time.time() - t0
            yp = e.predict(Xva)
            m = metrics(y[idx["val"]], yp, labels)
            val_rows.append({"family": fam, **params, **m, "fit_seconds": fit_t})
            if best is None or m["macro_f1"] > best[0]:
                best = (m["macro_f1"], params, est, yp, class_scores(e, Xva, labels))
        selected[fam] = best
        pd.DataFrame(val_rows).to_csv(RESULTS / "val_grid" / f"{name}__{fam}.csv", index=False)
        pd.to_pickle({"doc_id": docs["doc_id"].values[idx["val"]], "y_true": y[idx["val"]], "y_pred": best[3], "scores": best[4],
                      "labels": labels, "params": best[1], "family": fam, "representation": name},
                     RESULTS / "preds_val" / f"{name}__{fam}.pkl")
    chosen = max(selected, key=lambda f: selected[f][0])
    # --- 2. final fit on train+val, single evaluation on test (every family) ; 3. CV for the chosen family
    results = []
    for fam, (vf1, params, est, _, _) in selected.items():
        e = clone(est)
        t0 = time.time(); e.fit(Xtv, y[idx["trainval"]]); fit_t = time.time() - t0
        t0 = time.time(); yp = e.predict(Xte); pred_t = (time.time() - t0) / len(idx["test"])
        sc = class_scores(e, Xte, labels)
        m = metrics(y[idx["test"]], yp, labels, scores=sc, label_sets=label_sets[idx["test"]])
        ci = bootstrap_ci(y[idx["test"]], yp, labels)
        row = {"representation": name, "family": fam, "params": json.dumps(params), "val_macro_f1": vf1, "selected": fam == chosen,
               **m, **ci, "fit_seconds_trainval": fit_t, "predict_ms_per_doc": pred_t * 1000, "n_features": int(n_features), **(extra_meta or {})}
        pd.to_pickle({"doc_id": docs["doc_id"].values[idx["test"]], "y_true": y[idx["test"]], "y_pred": yp, "scores": sc, "labels": labels,
                      "params": params, "family": fam, "representation": name}, PREDS / f"{name}__{fam}.pkl")
        if cv and fam == chosen:
            accs, f1s = [], []
            for tr, te in cv_folds(docs):
                feat.fit(tr); Xa, Xb = feat.transform(tr), feat.transform(te)
                e = clone(est); e.fit(Xa, y[tr]); p = e.predict(Xb)
                accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, labels=labels, average="macro", zero_division=0))
            row.update({"cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_sd": float(np.std(accs, ddof=1)),
                        "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_sd": float(np.std(f1s, ddof=1)),
                        "cv_folds": json.dumps({"acc": accs, "macro_f1": f1s})})
        results.append(row)
        append_results([row], file=results_file)   # written immediately so an interrupted run resumes
        log(f"[{name}] {fam:10s} {'*' if fam == chosen else ' '} {params}  val mF1={vf1:.4f} | test acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} wF1={m['weighted_f1']:.4f}"
            + (f"  CV acc={row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_sd']:.4f}" if "cv_accuracy_mean" in row else ""))
    return results


def append_results(rows, file="results_main.csv"):
    """Upserts result rows (key: representation + family).  Every stage writes its own file
    (results_tfidf.csv, results_emb.csv, results_ft.csv, results_llm.csv) so that stages can run
    in parallel processes; the file is replaced atomically (write to a temp file, then rename)."""
    if not rows:
        return None
    f = RESULTS / file
    df = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f)
        old = old[~(old["representation"] + "__" + old["family"]).isin(df["representation"] + "__" + df["family"])]
        df = pd.concat([old, df], ignore_index=True)
    tmp = f.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(f)
    return df


def load_all_results():
    """All result files merged (later files win on duplicate representation + family)."""
    files = sorted(RESULTS.glob("results_*.csv"))
    files = [f for f in files if not f.name.endswith(".tmp.csv")]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=["representation", "family"], keep="last").reset_index(drop=True)

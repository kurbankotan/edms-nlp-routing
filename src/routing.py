"""
routing.py
----------
Routing mechanisms evaluated on the stored validation and test predictions of the local family
representatives (fine-tuned BERTurk: class probabilities; TF-IDF + linear SVM: decision values).

1. Calibration.  A single temperature T is fitted on the validation predictions by minimising the
   negative log-likelihood (temperature scaling; Guo et al., 2017) and applied to the test predictions.
   Expected calibration error (ECE, 15 equal-width confidence bins; Naeini et al., 2015) and reliability
   curves are reported before and after scaling.  Note: the test predictions come from the model refitted
   on train+validation, so T is transferred from the train-only model that produced the validation scores.
2. Deferral rules on the temperature-scaled confidences.  (a) empirical threshold: the largest validation
   coverage whose accuracy meets the target (as in analysis.py, which applies it to the raw confidences);
   (b) calibrated rule: route when the calibrated confidence >= target;
   (c) recency rule: the empirical threshold chosen on the last quarter of the validation half-year only.
3. Hierarchical back-off.  A document is routed to the predicted unit if the unit confidence reaches t_u;
   otherwise to the office of the predicted department (whose staff forward it inside the department) if
   the department confidence (sum of the unit probabilities of that department) reaches t_d; otherwise
   it is left to staff.  t_u and t_d are chosen on validation for the same target accuracy at their level.

    python src/routing.py        # -> results/tables/{calibration,hierarchical}.csv, routing_stats.json, figures/fig_routing.*
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import C, confidence, load_pred, plt, save, threshold_for_target   # noqa: E402  (palette, rcParams)
from common import PARENT, RESULTS, load_docs                        # noqa: E402

TAB = RESULTS / "tables"
SYSTEMS = {"BERTurk fine-tuned": ("ft_bert-base-turkish-cased__FT_s0", "prob"),
           "TF-IDF (lemma) + linear SVM": ("tfidf_word_lemma__LinearSVM", "margin")}     # lexical family representative (validation)
DEPTS = ["BİDB", "SKS", "Rektörlük"]
OFFICES = ["Bilgi İşlem Daire Başkanlığı", "Sağlık Kültür ve Spor Daire Başkanlığı", "Rektörlük"]   # office of each department
TARGETS = (0.90, 0.95)
RECENT_FROM = "2022-04-01"          # last quarter of the validation half-year


def logits(p, kind):
    s = np.asarray(p["scores"], dtype=float)
    return np.log(np.clip(s, 1e-12, 1.0)) if kind == "prob" else s


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y_idx):
    def nll(log_t):
        P = softmax(z / np.exp(log_t))
        return -float(np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))))
    r = minimize_scalar(nll, bounds=(np.log(0.02), np.log(50.0)), method="bounded")
    return float(np.exp(r.x))


def nll(P, y_idx):
    return -float(np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))))


def ece(conf, correct, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1); total = 0.0; curve = []
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
            curve.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(total), curve


def dept_matrix(labels):
    return np.array([[1.0 if PARENT[l] == d else 0.0 for d in DEPTS] for l in labels])


def main():
    docs, labels = load_docs()
    lab = np.array(labels)
    date = dict(zip(docs["doc_id"].astype(int), pd.to_datetime(docs["date"])))
    M = dept_matrix(labels)
    cal_rows, hier_rows, stats, curves = [], [], {}, {}
    for name, (key, kind) in SYSTEMS.items():
        pv, pt = load_pred(key, val=True), load_pred(key)
        assert list(pv["labels"]) == labels and list(pt["labels"]) == labels
        yv, yt = np.asarray(pv["y_true"]), np.asarray(pt["y_true"])
        iv, it = np.array([labels.index(y) for y in yv]), np.array([labels.index(y) for y in yt])
        zv, zt = logits(pv, kind), logits(pt, kind)
        T = fit_temperature(zv, iv)
        Pv0, Pt0, Pv1, Pt1 = softmax(zv), softmax(zt), softmax(zv / T), softmax(zt / T)
        corr_v, corr_t = Pv0.argmax(1) == iv, Pt0.argmax(1) == it          # argmax is unchanged by T
        recent = np.array([date[int(d)] >= pd.Timestamp(RECENT_FROM) for d in pv["doc_id"]])
        row = {"system": name, "temperature": T}
        for tag, Pa, Pb in (("before", Pv0, Pt0), ("after", Pv1, Pt1)):
            row[f"ece_val_{tag}"] = ece(Pa.max(1), corr_v)[0]
            e_t, curve = ece(Pb.max(1), corr_t)
            row[f"ece_test_{tag}"] = e_t; row[f"nll_test_{tag}"] = nll(Pb, it)
            curves[(name, tag)] = curve
        T_recent = fit_temperature(zv[recent], iv[recent])
        row["temperature_recent"] = T_recent
        row["ece_test_after_recent"] = ece(softmax(zt / T_recent).max(1), corr_t)[0]
        # temperature scaling acts on the whole logit vector, so it can reorder the documents' top probabilities slightly
        row["spearman_raw_vs_scaled_test"] = float(spearmanr(confidence(pt), Pt1.max(1))[0])
        for tgt in TARGETS:
            t = int(tgt * 100)
            thr = threshold_for_target(Pv1.max(1), corr_v, tgt); m = Pt1.max(1) >= thr
            row[f"emp_cov@{t}"], row[f"emp_acc@{t}"] = float(m.mean()), float(corr_t[m].mean()) if m.any() else np.nan
            row[f"emp_val_cov@{t}"] = float((Pv1.max(1) >= thr).mean())
            raw_v = confidence(pv); row[f"raw_val_cov@{t}"] = float((raw_v >= threshold_for_target(raw_v, corr_v, tgt)).mean())
            m = Pt1.max(1) >= tgt
            row[f"cal_cov@{t}"], row[f"cal_acc@{t}"] = float(m.mean()), float(corr_t[m].mean()) if m.any() else np.nan
            thr = threshold_for_target(Pv1[recent].max(1), corr_v[recent], tgt); m = Pt1.max(1) >= thr
            row[f"rec_cov@{t}"], row[f"rec_acc@{t}"] = float(m.mean()), float(corr_t[m].mean()) if m.any() else np.nan
        cal_rows.append(row)

        # hierarchical back-off on the calibrated probabilities
        Dv, Dt = Pv1 @ M, Pt1 @ M
        dep = np.array(DEPTS)
        yv_d, yt_d = np.array([PARENT[y] for y in yv]), np.array([PARENT[y] for y in yt])
        ok_dv, ok_dt = dep[Dv.argmax(1)] == yv_d, dep[Dt.argmax(1)] == yt_d
        for tgt in TARGETS:
            t_u = threshold_for_target(Pv1.max(1), corr_v, tgt)
            rest = Pv1.max(1) < t_u
            t_d = threshold_for_target(Dv.max(1)[rest], ok_dv[rest], tgt) if rest.any() else np.inf
            r = {"system": name, "target": tgt, "t_unit": t_u, "t_dept": t_d}
            for split, P, D, corr, okd, ytrue in (("val", Pv1, Dv, corr_v, ok_dv, yv), ("test", Pt1, Dt, corr_t, ok_dt, yt)):
                unit = P.max(1) >= t_u; dpt = (~unit) & (D.max(1) >= t_d); staff = ~(unit | dpt)
                office_true = np.isin(ytrue, OFFICES)          # the office itself was the target: no forwarding needed
                r.update({f"{split}_unit_share": float(unit.mean()), f"{split}_unit_acc": float(corr[unit].mean()) if unit.any() else np.nan,
                          f"{split}_dept_share": float(dpt.mean()), f"{split}_dept_acc": float(okd[dpt].mean()) if dpt.any() else np.nan,
                          f"{split}_staff_share": float(staff.mean()), f"{split}_resolved_share": float((unit | dpt).mean()),
                          f"{split}_dept_needs_forwarding": float((dpt & okd & ~office_true).sum() / max(dpt.sum(), 1)),
                          f"{split}_dept_office_is_target": float((dpt & okd & office_true).sum() / max(dpt.sum(), 1))})
            hier_rows.append(r)
        # top-down decoding (department first, then the unit inside it) versus flat argmax
        dsel = Dt.argmax(1)
        td = np.where(M[:, dsel].T > 0, Pt1, -1.0).argmax(1)
        stats[name] = {"flat_accuracy": float(corr_t.mean()), "top_down_accuracy": float((td == it).mean()),
                       "department_accuracy_flat": float((dep[(Pt1 @ M).argmax(1)] == yt_d).mean()),
                       "department_accuracy_from_unit": float((np.array([PARENT[l] for l in lab[Pt1.argmax(1)]]) == yt_d).mean()),
                       "n_val_recent": int(recent.sum())}

    cal = pd.DataFrame(cal_rows); hier = pd.DataFrame(hier_rows)
    TAB.mkdir(parents=True, exist_ok=True)
    cal.to_csv(TAB / "calibration.csv", index=False); hier.to_csv(TAB / "hierarchical.csv", index=False)
    (TAB / "routing_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---- figure: (a) reliability of BERTurk on the test half-year; (b) where the documents go
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=C["axis"], lw=1, ls="--", zorder=1)
    for tag, col, lab_ in (("before", C["blue"], "uncalibrated"), ("after", C["orange"], f"temperature-scaled (T = {cal.iloc[0]['temperature']:.2f})")):
        cv = curves[("BERTurk fine-tuned", tag)]
        x, yy, n = zip(*[c for c in cv if c[2] >= 10])
        ax.plot(x, yy, color=col, lw=1.6, marker="o", ms=4.5, mec="white", mew=1.0, label=lab_, zorder=3)
    e0, e1 = cal.iloc[0]["ece_test_before"], cal.iloc[0]["ece_test_after"]
    ax.set_xlim(0.15, 1.0); ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("Confidence (mean of bin)"); ax.set_ylabel("Accuracy in bin")
    ax.set_title(f"(a) Reliability, BERTurk, test (ECE {e0:.3f} → {e1:.3f})")
    ax.legend(fontsize=7, loc="upper left"); ax.yaxis.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    ax2 = axes[1]
    rows = []
    for sysname in SYSTEMS:
        for tgt in TARGETS:
            h = hier[(hier["system"] == sysname) & (hier["target"] == tgt)].iloc[0]
            short = "BERTurk" if sysname.startswith("BERTurk") else "TF-IDF"
            rows.append((f"{short}\n{int(tgt*100)}% target", h["test_unit_share"], h["test_dept_share"], h["test_staff_share"]))
    ylab = [r[0] for r in rows][::-1]; vals = np.array([r[1:] for r in rows])[::-1]
    left = np.zeros(len(rows))
    for j, (col, lab_) in enumerate(((C["blue"], "unit (automatic)"), (C["orange"], "department office (back-off)"), (C["violet"], "staff"))):
        ax2.barh(ylab, vals[:, j] * 100, left=left, color=col, edgecolor="white", linewidth=1.2, height=0.62, label=lab_)
        for i, v in enumerate(vals[:, j]):
            if v >= 0.05:
                ax2.text(left[i] + v * 50, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        left += vals[:, j] * 100
    ax2.set_xlim(0, 100); ax2.set_xlabel("Share of test documents (%)")
    ax2.set_title("(b) Hierarchical back-off on the test half-year")
    ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    save(fig, "fig_routing")
    pd.set_option("display.width", 250)
    print(cal.round(4).T.to_string())
    print(hier.round(4).to_string(index=False))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

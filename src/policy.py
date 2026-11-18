"""
policy.py
---------
LLM-informed hierarchical routing: one policy that combines the calibrated local classifier, the ranked answer
of a retrieval-augmented LLM and the organisational hierarchy, evaluated on the stored validation and test
predictions (no new model or API calls).

1. Local posterior.  P_L(y|x): temperature-scaled probabilities of the local model (as in routing.py).
2. Cost-aware querying.  The LLM is asked only when the local model cannot route the document on its own,
   i.e. when max_y P_L(y|x) < t_q, the unit threshold of the local back-off for the same target accuracy.
3. Evidence fusion.  The LLM's answer is treated as noisy evidence about the unit.  With r(y) in {1, 2, 3, 0}
   the rank of unit y in the LLM's answer (0: not listed) and pi_r the probability that the true unit is at
   rank r divided by the number of units at that rank (estimated on the queried validation documents),
        P_F(y|x)  is proportional to  exp([log P_L(y|x) + g * log pi_r(y)] / T_F),
   where the evidence weight g and the temperature T_F are fitted on the queried validation documents by
   minimising the negative log-likelihood (g < 1 discounts the dependence between the two sources).
4. Hierarchical decision on the fused posterior.  Route to the unit argmax P_F if max P_F >= t_u; otherwise to
   the office of the department with the largest probability mass if that mass >= t_d; otherwise to staff.
   t_u and t_d are chosen on validation for the target accuracy at their level, exactly as for the local
   back-off, which is the policy without step 3.
5. Test evaluation with the validation thresholds, LLM share and cost, and paired bootstrap intervals for the
   differences to the local back-off (staff share and share of documents routed to a wrong unit or department).

    python src/policy.py        # -> results/tables/policy.csv, policy_stats.json, figures/fig_policy.*
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import C, load_pred, plt, save, threshold_for_target    # noqa: E402
from common import PARENT, RESULTS, load_docs                          # noqa: E402
from llm import final_answers                                          # noqa: E402
from routing import DEPTS, OFFICES, SYSTEMS, TARGETS, dept_matrix, fit_temperature, logits, softmax   # noqa: E402

TAB = RESULTS / "tables"
LLMS = {"gemini-3.8-flash": "Gemini 3.8 Flash", "gpt-6-luna": "GPT-6 Luna", "claude-haiku-5-5": "Claude Haiku 5.5"}
N_BOOT = 2000


def llm_ranks(model, split, ids, labels, fallback):
    """(n, K) rank of every unit in the LLM's answer: 0, 1, 2 for the three listed units, 3 if not listed."""
    ans = final_answers(model, "rag10", split, labels)
    R = np.full((len(ids), len(labels)), 3)
    for n, d in enumerate(ids):
        for r, u in enumerate((ans.get(int(d)) or [fallback])[:3]):
            j = labels.index(u); R[n, j] = min(R[n, j], r)
    return R


def rank_likelihood(R, y, K):
    """pi_r: P(true unit at rank r) / number of units at rank r (Laplace-smoothed)."""
    n = np.bincount(R[np.arange(len(y)), y], minlength=4).astype(float) + 1.0
    return (n / n.sum()) / np.array([1.0, 1.0, 1.0, K - 3.0])


def fuse(P, R, pi, g, T):
    return softmax((np.log(np.clip(P, 1e-12, 1.0)) + g * np.log(pi[R])) / T)


def nll(P, y):
    return -float(np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def thresholds(P, y, M, dep_of, tgt):
    """Unit threshold on max P, then department threshold on the remaining documents (as routing.py)."""
    t_u = threshold_for_target(P.max(1), P.argmax(1) == y, tgt)
    rest = P.max(1) < t_u
    D = P @ M
    t_d = threshold_for_target(D.max(1)[rest], (D.argmax(1) == dep_of[y])[rest], tgt) if rest.any() else np.inf
    return t_u, t_d


def decide(P, M, t_u, t_d):
    D = P @ M
    unit = P.max(1) >= t_u
    office = (~unit) & (D.max(1) >= t_d)
    return unit, office, ~(unit | office), P.argmax(1), D.argmax(1)


def outcome(P, y, M, dep_of, office_idx, t_u, t_d):
    """Per-document indicators: unit / office / staff, wrong automatic decision, forwarding step needed."""
    unit, office, staff, u_hat, d_hat = decide(P, M, t_u, t_d)
    ok_u, ok_d = u_hat == y, d_hat == dep_of[y]
    wrong = (unit & ~ok_u) | (office & ~ok_d)
    fwd = office & ok_d & ~np.isin(y, office_idx)
    return dict(unit=unit, office=office, staff=staff, wrong=wrong, fwd=fwd, ok_u=ok_u, ok_d=ok_d)


def summary(o, prefix):
    u, of = o["unit"], o["office"]
    return {f"{prefix}unit_share": float(u.mean()), f"{prefix}unit_acc": float(o["ok_u"][u].mean()) if u.any() else np.nan,
            f"{prefix}office_share": float(of.mean()), f"{prefix}office_dept_acc": float(o["ok_d"][of].mean()) if of.any() else np.nan,
            f"{prefix}staff_share": float(o["staff"].mean()), f"{prefix}wrong_share": float(o["wrong"].mean()),
            f"{prefix}forwarding_share": float(o["fwd"].mean())}


def main():
    docs, labels = load_docs()
    K = len(labels); M = dept_matrix(labels)
    dep_of = np.array([DEPTS.index(PARENT[l]) for l in labels])
    office_idx = [labels.index(o) for o in OFFICES]
    fallback = docs.loc[docs["split"] != "test", "label"].value_counts().index[0]
    usd = pd.read_csv(RESULTS / "results_llm.csv").set_index(["representation", "family"])["usd_per_1000_docs"]
    rng = np.random.default_rng(0)
    rows, stats, plot = [], {}, {}
    for name, (key, kind) in SYSTEMS.items():
        pv, pt = load_pred(key, val=True), load_pred(key)
        yv = np.array([labels.index(v) for v in pv["y_true"]]); yt = np.array([labels.index(v) for v in pt["y_true"]])
        zv, zt = logits(pv, kind), logits(pt, kind)
        T = fit_temperature(zv, yv)
        Pv, Pt = softmax(zv / T), softmax(zt / T)
        boot = [rng.integers(0, len(yt), len(yt)) for _ in range(N_BOOT)]
        for tgt in TARGETS:
            t_q, t_d0 = thresholds(Pv, yv, M, dep_of, tgt)                     # local back-off = policy without the LLM
            base = outcome(Pt, yt, M, dep_of, office_idx, t_q, t_d0)
            rows.append({"local_system": name, "llm": "none (local back-off)", "target": tgt, "t_unit": t_q, "t_dept": t_d0,
                         "llm_share": 0.0, "llm_usd_per_1000_docs": 0.0, **summary(base, "test_")})
            plot[(name, tgt, "none")] = base
            for model, label in LLMS.items():
                qv, qt = Pv.max(1) < t_q, Pt.max(1) < t_q                    # documents the local model cannot route alone
                Rv = llm_ranks(model, "val", pv["doc_id"], labels, fallback)
                Rt = llm_ranks(model, "test", pt["doc_id"], labels, fallback)
                pi = rank_likelihood(Rv[qv], yv[qv], K)
                res = minimize(lambda x: nll(fuse(Pv[qv], Rv[qv], pi, x[0], np.exp(x[1])), yv[qv]), x0=[1.0, 0.0], method="Nelder-Mead")
                g, T_F = float(res.x[0]), float(np.exp(res.x[1]))
                Fv, Ft = Pv.copy(), Pt.copy()
                Fv[qv] = fuse(Pv[qv], Rv[qv], pi, g, T_F); Ft[qt] = fuse(Pt[qt], Rt[qt], pi, g, T_F)
                t_u, t_d = thresholds(Fv, yv, M, dep_of, tgt)
                o = outcome(Ft, yt, M, dep_of, office_idx, t_u, t_d)
                d_staff = [o["staff"][b].mean() - base["staff"][b].mean() for b in boot]
                d_wrong = [o["wrong"][b].mean() - base["wrong"][b].mean() for b in boot]
                share = float(qt.mean())
                rows.append({"local_system": name, "llm": label, "target": tgt, "t_unit": t_u, "t_dept": t_d, "t_query": t_q,
                             "evidence_weight": g, "temperature_fused": T_F, "pi": json.dumps([float(v) for v in pi]),
                             "llm_share": share, "llm_usd_per_1000_docs": share * float(usd.loc[(f"llm_{model}", "rag10")]),
                             **summary(o, "test_"),
                             "diff_staff": float(o["staff"].mean() - base["staff"].mean()),
                             "diff_staff_ci": json.dumps([float(np.percentile(d_staff, 2.5)), float(np.percentile(d_staff, 97.5))]),
                             "diff_wrong": float(o["wrong"].mean() - base["wrong"].mean()),
                             "diff_wrong_ci": json.dumps([float(np.percentile(d_wrong, 2.5)), float(np.percentile(d_wrong, 97.5))]),
                             "fused_accuracy_queried": float((Ft[qt].argmax(1) == yt[qt]).mean()),
                             "local_accuracy_queried": float((Pt[qt].argmax(1) == yt[qt]).mean()),
                             "llm_accuracy_queried": float((Rt[qt][np.arange(qt.sum()), yt[qt]] == 0).mean())})
                plot[(name, tgt, label)] = o
    pol = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    pol.to_csv(TAB / "policy.csv", index=False)

    # ---- figure: where the documents go at the 95% target, local back-off vs LLM-informed policy
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    short = {"BERTurk fine-tuned": "BERTurk", "TF-IDF (lemma) + linear SVM": "TF-IDF"}
    for ax, name in zip(axes, SYSTEMS):
        cases = ["none"] + list(LLMS.values())
        names = ["local back-off"] + [f"+ {c}" for c in LLMS.values()]
        left = np.zeros(len(cases))
        for j, (part, col, lab_) in enumerate((("unit", C["blue"], "unit (automatic)"), ("office", C["orange"], "department office"),
                                              ("staff", C["violet"], "staff"))):
            v = np.array([plot[(name, 0.95, c)][part].mean() for c in cases])
            ax.barh(range(len(cases)), v * 100, left=left * 100, color=col, label=lab_, height=0.62, edgecolor="white", linewidth=1)
            for i in range(len(cases)):
                if v[i] >= 0.05:
                    ax.text((left[i] + v[i] / 2) * 100, i, f"{v[i]*100:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            left += v
        for i, c in enumerate(cases):
            ax.text(101.5, i, f"wrong {plot[(name, 0.95, c)]['wrong'].mean()*100:.1f}%", va="center", fontsize=7, color=C["ink2"])
        ax.set_yticks(range(len(cases))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 122); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("Share of test documents (%)")
        ax.set_title(f"({'ab'[list(SYSTEMS).index(name)]}) {short[name]} as first stage, 95% target", fontsize=9, loc="left")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.04, -0.22), ncol=3, frameon=False, fontsize=8)
    save(fig, "fig_policy")
    stats["note"] = "thresholds chosen on validation; bootstrap over test documents with fixed thresholds"
    (TAB / "policy_stats.json").write_text(json.dumps(stats, indent=2))
    cols = ["local_system", "llm", "target", "llm_share", "test_unit_share", "test_unit_acc", "test_office_share", "test_office_dept_acc",
            "test_staff_share", "test_wrong_share", "diff_staff", "diff_staff_ci", "diff_wrong", "diff_wrong_ci"]
    print(pol[[c for c in cols if c in pol.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

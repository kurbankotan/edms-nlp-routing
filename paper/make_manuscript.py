"""
make_manuscript.py
------------------
Builds paper/manuscript.json (content blocks) from the result files, then renders it to Word
with paper/render_docx.js.  Every number in the text is read from results/, so re-running
after new experiments regenerates a consistent manuscript.

    python paper/make_manuscript.py            # -> paper/Manuscript.docx
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import PARENT, SHORT   # noqa: E402
from analysis import LLM_NAMES     # noqa: E402

RES = ROOT / "results"
TAB = RES / "tables"
FIG = ROOT / "figures"
OUT = ROOT / "paper"

PENDING = "[pending]"
ANON = "--anon" in sys.argv     # copy for double-anonymized review: no author details, institution name or repository owner
sys.path.insert(0, str(OUT))
import statements as S          # noqa: E402  (declarations shared with the title page and the cover letter)


# ----------------------------------------------------------------------------- data access
def load():
    R = {"st": json.loads((RES / "dataset_stats.json").read_text())}
    R["prep"] = json.loads((RES / "preprocessing_stats.json").read_text()) if (RES / "preprocessing_stats.json").exists() else {}
    for name in ["main", "all_configurations", "per_class", "mcnemar", "deferral", "cascade", "error_summary", "confused_pairs",
                 "ablation_lexical", "ablations", "cost", "dataset_classes", "representatives", "mcnemar_extra", "calibration", "hierarchical"]:
        f = TAB / f"{name}.csv"
        R[name] = pd.read_csv(f) if f.exists() else None
    R["ex"] = json.loads((TAB / "extra_stats.json").read_text()) if (TAB / "extra_stats.json").exists() else {}
    R["rs"] = json.loads((TAB / "routing_stats.json").read_text()) if (TAB / "routing_stats.json").exists() else {}
    R["llm"] = pd.read_csv(RES / "results_llm.csv") if (RES / "results_llm.csv").exists() else None
    R["deid"] = json.loads((RES / "deidentification_stats.json").read_text()) if (RES / "deidentification_stats.json").exists() else {}
    return R


def pct(x, d=1):
    return PENDING if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def f3(x):
    return PENDING if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(s):
    try:
        lo, hi = json.loads(s) if isinstance(s, str) else s
        return f"[{100*lo:.1f}, {100*hi:.1f}]"
    except Exception:
        return PENDING


def row_of(df, contains):
    if df is None:
        return None
    m = df[df["system"].str.contains(contains, regex=False)]
    return m.iloc[0] if len(m) else None


def fig_block(name, caption, width_in=6.3):
    p = FIG / f"{name}.png"
    if not p.exists():
        return {"type": "figure_missing", "caption": caption, "text": f"[{caption} — not available yet]"}
    w, h = Image.open(p).size
    return {"type": "figure", "path": f"figures/{name}.png", "caption": caption, "width_in": width_in, "aspect": w / h}


def small(n):
    """Integers below ten in words (journal style)."""
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n] if 0 <= n < 10 else f"{n:,}"


def H(level, text):
    return {"type": "heading", "level": level, "text": text}


def P(text):
    return {"type": "paragraph", "text": text}


# ----------------------------------------------------------------------------- content
def build(R):
    st = R["st"]; main = R["main"]; ex = R["ex"]; reps = R["representatives"]
    n_docs, n_cls = st["n_documents"], st["n_classes"]
    sp = st["split_sizes"]
    best = main.iloc[0] if main is not None else None                      # highest test macro-F1 (reported, not selected)
    final = reps.iloc[0] if reps is not None and len(reps) else best      # chosen on validation macro-F1
    def rep_of(group):
        r = reps[reps["group"] == group] if reps is not None else None
        return r.iloc[0] if r is not None and len(r) else None
    lex, rep_emb, ft, rep_llm = (rep_of(g) for g in ["Lexical (TF-IDF)", "Frozen encoder", "Fine-tuned encoder", "LLM"])
    LLM3 = ["gemini-3.8-flash", "gpt-6-luna", "claude-haiku-5-5"]          # the LLMs of the study (Section 4.5)
    gem, gpt, hai = (row_of(main, f"{LLM_NAMES['llm_' + m]} / RAG-10") for m in LLM3)
    gem0, gpt0, hai0 = (row_of(main, f"{LLM_NAMES['llm_' + m]} / zero-shot") for m in LLM3)
    zs, rg = [gem0, gpt0, hai0], [gem, gpt, hai]
    qwen = row_of(main, "Qwen3-Embedding-4B"); xlmr = row_of(main, "XLM-R base fine-tuned"); muse = row_of(main, "distiluse")
    e5b = row_of(main, "multilingual-e5-base /")
    llm_rows = main[main["group"] == "LLM"] if main is not None else None
    emb_rows = main[main["group"] == "Frozen encoder"] if main is not None else None
    defer = R["deferral"]
    d_best = row_of(defer, "BERTurk fine-tuned") if defer is not None else None
    d_lex = row_of(defer, "TF-IDF") if defer is not None else None
    d_emb = row_of(defer, rep_emb["system"]) if defer is not None and rep_emb is not None else None
    casc = R["cascade"]
    cg = ex.get("cascade", {}).get("Cascade BERTurk -> Gemini", {})
    cgpt = ex.get("cascade", {}).get("Cascade BERTurk -> GPT", {}); chai = ex.get("cascade", {}).get("Cascade BERTurk -> Haiku", {})
    knn = ex.get("knn10_vote", {})
    seeds = ex.get("berturk_seed_sensitivity", {})
    boot = ex.get("macro_f1_difference_bootstrap", {})
    def bd(a, b):
        """Paired-bootstrap macro-F1 difference a - b in points with its 95% interval."""
        r = boot.get(f"{a} | {b}")
        if r is None:
            return PENDING
        sg = lambda v: f"{100*v:+.1f}".replace("-", "−")
        return f"{sg(r['diff'])} points, 95% CI [{sg(r['ci'][0])}, {sg(r['ci'][1])}]"
    def pfmt(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.1 else f"p = {p:.3f}"
    deid = R["deid"]
    LEXN = ex.get("lexical_representative", lex["system"] if lex is not None else "")     # name of the lexical representative in extra_stats
    _plan = ex.get("rag_prompts_asked_again", {}).get("rag10__test")     # RAG prompts asked again after the retrieval moved to de-identified text
    rr_share = _plan["changed"] / _plan["documents"] if _plan else None
    mce = R["mcnemar_extra"]
    def mcp(a, b):
        """p-value (uncorrected) of an extra McNemar test, formatted."""
        if mce is None:
            return PENDING
        r = mce[((mce["system_a"] == a) & (mce["system_b"] == b)) | ((mce["system_a"] == b) & (mce["system_b"] == a))]
        if not len(r):
            return PENDING
        return pfmt(float(r.iloc[0]["p_value"]))
    mc = R["mcnemar"]
    def mcp_rep(a, b):
        r = mc[(mc["system_a"].str.contains(a, regex=False) & mc["system_b"].str.contains(b, regex=False)) |
               (mc["system_a"].str.contains(b, regex=False) & mc["system_b"].str.contains(a, regex=False))] if mc is not None else []
        if not len(r):
            return PENDING
        p = float(r.iloc[0]["p_value"])
        return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.1 else f"p = {p:.3f}"
    llm_res = R["llm"].set_index(["representation", "family"]) if R["llm"] is not None else None
    def usd(model, mode="rag10"):
        try:
            return f"{float(llm_res.loc[(f'llm_{model}', mode), 'usd_per_1000_docs']):.2f}"
        except Exception:
            return PENDING
    from llm import LLM_DIR, model_info, read_records       # noqa: E402
    usd_total = {}                                           # every API call of the study (test, validation, re-requests)
    for f in LLM_DIR.glob("*.jsonl"):
        m = f.stem.split("__")[0]
        if m not in LLM3:                                    # responses of models that are not part of the study
            continue
        _, pin, pout = model_info(m)
        usd_total[m] = usd_total.get(m, 0.0) + sum(r["in_tokens"] * pin + r["out_tokens"] * pout for r in read_records(f)) / 1e6
    abl_all = R["ablations"]
    cal, hie, rs = R["calibration"], R["hierarchical"], R["rs"]
    cb_ = cal[cal["system"].str.startswith("BERTurk")].iloc[0] if cal is not None else None
    ct_ = cal[cal["system"].str.startswith("TF-IDF")].iloc[0] if cal is not None else None
    def H_(system_prefix, tgt):
        r = hie[hie["system"].str.startswith(system_prefix) & (abs(hie["target"] - tgt) < 1e-9)] if hie is not None else []
        return r.iloc[0] if len(r) else None
    hb95, hb90, ht95, ht90 = H_("BERTurk", 0.95), H_("BERTurk", 0.90), H_("TF-IDF", 0.95), H_("TF-IDF", 0.90)
    classes = st["class_counts"]
    maj = max(classes.values()) / n_docs
    meta = {
        "title": S.TITLE,
        "authors": "" if ANON else ", ".join(a["name"] + "^{" + a["aff"] + (",*" if a["corresponding"] else "") + "}" for a in S.AUTHORS),
        "affiliations": [] if ANON else [f"^{{{k}}} {v}" for k, v in S.AFFILIATIONS.items()],
        "corresponding": "" if ANON else f"* Corresponding author. E-mail: {S.EMAIL}",
        "creator": "Anonymous" if ANON else "; ".join(a["name"] for a in S.AUTHORS),
    }
    B = []

    # ---------------------------------------------------------------- highlights & abstract
    abl0 = R["ablation_lexical"]
    def acc_of(rep):
        r = abl0[abl0["representation"] == rep] if abl0 is not None else []
        return float(r.iloc[0]["accuracy"]) if len(r) else np.nan
    tr_drop = 100 * (acc_of("tfidf_content_tr") - acc_of("tfidf_content_en"))           # translation cost of the lexical model (points)
    local_mf1_max = float(main.loc[main["group"] != "LLM", "macro_f1"].max())
    zs_lo, zs_hi = min(r["accuracy"] for r in zs), max(r["accuracy"] for r in zs)
    rg_lo, rg_hi = min(r["accuracy"] for r in rg), max(r["accuracy"] for r in rg)
    highlights = [
        f"Routing of {n_docs:,} de-identified Turkish EDMS documents on a chronological test",
        f"TF-IDF on native Turkish text rivals fine-tuned BERTurk; translation costs {tr_drop:.0f} points",
        f"Retrieved past documents lift LLM routing accuracy from {100*zs_lo:.0f}–{100*zs_hi:.0f}% to {100*rg_lo:.0f}–{100*rg_hi:.0f}%",
        f"A BERTurk-to-LLM cascade reaches {pct(cg.get('test_accuracy'))}% with LLM calls for {100*cg.get('llm_share', 0):.0f}% of documents",
        f"Back-off to department offices leaves only {100*hb95['test_staff_share']:.0f}% of documents to staff",
    ]
    assert all(len(h) <= 85 for h in highlights), [len(h) for h in highlights]     # Elsevier limit
    B.append({"type": "highlights", "items": highlights})
    B.append({"type": "abstract", "keywords": "Electronic document management; document routing; Turkish text classification; large language models; selective prediction; sentence embeddings",
              "text": (
        f"Routing incoming documents to the organisational unit that must act on them remains manual in many electronic document "
        f"management systems (EDMS). This study treats routing as decision support and evaluates it on {n_docs:,} documents of a Turkish "
        f"university with {n_cls} target units. The evaluation is chronological (training up to 2021, validation and test in "
        f"2022), all choices are made on validation data, and macro-averaged metrics with bootstrap intervals and McNemar tests are reported. "
        f"Term frequency–inverse document frequency (TF-IDF) models on Turkish text, eleven frozen sentence encoders, "
        f"fine-tuned BERTurk and XLM-RoBERTa, and three large language models (LLMs) prompted zero-shot or with ten retrieved past documents "
        f"are compared, all on de-identified text. Retrieval raised LLM accuracy from {pct(zs_lo)}–{pct(zs_hi)}% to {pct(rg_lo)}–{pct(rg_hi)}%; "
        f"the best LLM had the highest macro-F1 ({f3(gem['macro_f1'])} against at most {f3(local_mf1_max)} for local models), but its accuracy "
        f"({pct(gem['accuracy'])}%) did not differ significantly from a frozen Qwen3 embedding with a support vector machine "
        f"({pct(qwen['accuracy'])}%). Fine-tuned BERTurk ({pct(ft['accuracy'])}%) and the TF-IDF baseline ({pct(lex['accuracy'])}%) had practically "
        f"equal macro-F1. Grouped cross-validation gave up to {ex['cv_minus_test_accuracy_points']['max']:.1f} points more than the chronological test, "
        f"and machine translation to English lowered TF-IDF accuracy by {tr_drop:.1f} points. A cascade passing the {100*cg['llm_share']:.0f}% "
        f"least confident documents from BERTurk to the LLM reached {pct(cg['test_accuracy'])}% at {100*cg['llm_share']:.0f}% of the LLM "
        f"cost. Temperature scaling halved the calibration error, and backing off to the department office left {100*hb95['test_staff_share']:.0f}% "
        f"of documents to staff at a 95% target, although validation thresholds lost up to three points on the test half-year. Code, splits and "
        f"predictions are released.")})

    # ---------------------------------------------------------------- 1 introduction
    B.append(H(1, "1. Introduction"))
    B.append(P(
        "Public institutions produce and receive large volumes of official correspondence: petitions, requests, notifications, approvals and reports that "
        "are registered in an electronic document management system (EDMS) and then dispatched to the unit that has to act on them (Sprague, 1995). "
        "Although the EDMS has digitised the document life cycle, the dispatch decision itself usually remains manual: a registrar or a department head "
        "reads the subject line and the body and chooses a target unit from the organisational chart. In a university with dozens of departments and "
        "sub-units this step is repeated thousands of times a year, it delays processing when the responsible person is unavailable, and misrouted "
        "documents have to be returned and re-dispatched. Automating the decision, or at least proposing the most likely units with a measure of "
        "confidence, is therefore a natural application of text classification."))
    B.append(P(
        "Text classification itself is a mature field, from bag-of-words models with linear classifiers (Joachims, 1998; Sebastiani, 2002) to "
        "contextual encoders (Devlin et al., 2019) and, recently, large language models (LLMs) used as zero- or few-shot classifiers (Sun et al., 2023). "
        "Institutional document routing, however, differs from the benchmark setting in ways that matter for a deployable system. The texts are short, "
        "formulaic and highly templated; identical or near-identical documents recur over time; the target classes are organisational units whose "
        "boundaries are defined by internal practice rather than by topic; the class distribution is strongly skewed; and the system will be used on "
        "future correspondence, so its accuracy has to be estimated on documents written after those used for training. Several of these properties "
        "invalidate the random train-test split that is still common in applied studies: with recurring documents, a random split leaks test documents "
        "into training and yields optimistic estimates (Kapoor & Narayanan, 2023; Søgaard et al., 2021), and selecting among many model configurations on "
        "the same test split inflates the reported score further. Finally, an EDMS contains personal data, which constrains the use of external LLM "
        "services and requires de-identification before any text leaves the institution."))
    B.append(P(
        "This paper addresses these issues for Turkish institutional correspondence. It builds on a preliminary comparison of language representation "
        "models and classifiers on the same EDMS data, in which documents were machine-translated into English and lemmatised with English tools, and "
        "126 model combinations were ranked on a random split. The present study replaces that design with a leakage-controlled, chronological protocol, "
        "adds the baselines whose absence makes earlier comparisons inconclusive (TF-IDF with linear models, modern multilingual sentence encoders, "
        "fine-tuned Turkish encoders and LLMs), processes Turkish natively, and extends the classifier into a routing mechanism with top-k suggestions, "
        "confidence-based deferral to staff and an optional LLM second stage. The main contributions are as follows."))
    B.append({"type": "numbered", "id": "", "items": [
        f"A documented construction of a document-level routing dataset from an EDMS dispatch log ({st['n_dispatch_records_raw']:,} records, "
        f"{n_docs:,} documents, {n_cls} units), with an explicit treatment of broadcast circulars, multi-unit dispatch chains, duplicate texts and the label-consistency ceiling of the data, "
        "and a de-identification step after which every model sees only de-identified text.",
        "A leakage-controlled evaluation protocol: chronological train/validation/test partitions, validation-only model selection, grouped cross-validation "
        "in which identical texts never cross folds, bootstrap confidence intervals, McNemar tests and macro-averaged, per-unit and hierarchical (department-level) metrics.",
        "A controlled comparison of four classifier families under identical conditions (TF-IDF, eleven frozen sentence encoders, two fine-tuned "
        "encoders and three LLMs), including the lexical baseline that earlier work omitted, with ablations of Turkish preprocessing (raw, stemming, "
        "Zeyrek lemmatisation), input fields, machine translation and sender-unit context.",
        "A confidence-aware routing framework that turns the classifier into decision support: top-k suggestions, selective automation with thresholds "
        "chosen on validation data, and a cost- and latency-aware cascade whose low-confidence documents are passed to a retrieval-augmented LLM on de-identified text.",
        "A fully reproducible release (code, splits, seeds, hyper-parameters and prediction files) and a critical discussion of what the results do and do not support.",
    ]})
    B.append(P(
        "The remainder of the paper is organised as follows. Section 2 reviews related work on document routing, Turkish text classification, "
        "sentence encoders, LLM-based classification and evaluation practice. Section 3 defines the routing problem and describes the dataset. "
        "Section 4 presents the proposed framework and the evaluation protocol. Section 5 reports the results, Section 6 discusses them together with "
        "the limitations of the study, and Section 7 concludes."))

    # ---------------------------------------------------------------- 2 related work
    B.append(H(1, "2. Related work"))
    B.append(H(2, "2.1. Document classification and routing in organisations"))
    B.append(P(
        "Assigning documents to organisational categories has been studied in several institutional settings. Binici (2019) assigned standard file-plan "
        "numbers to Turkish e-records with a support vector machine trained on a small manually labelled sample and reported 88% agreement with manual "
        "classification. Alzoubi et al. (2023) classified about 225,000 Turkish customer inquiries of an institution into 14 classes and found an LSTM "
        "(84% accuracy) ahead of SVM (78%) after data preparation. Schwaar et al. (2025) routed messages of a German public administration to 45 offices and "
        "compared normalisation and segmentation strategies for documents of varying length, reaching cross-validated accuracies above 80% with "
        "classical and neural classifiers. Closely related is the routing of IT service tickets to support groups, where recent work compares "
        "classical models, fine-tuned encoders and LLMs (Alsaç et al., 2025) and studies cost-aware abstention, i.e. when a ticket should not be routed "
        "automatically (Gupta, 2026). A recurring finding of this line of work is that a well-tuned TF-IDF model with a linear classifier remains a strong "
        "baseline for short institutional texts (Gupta, 2026; Wang & Manning, 2012), which is why the present study treats it as the reference system "
        "rather than omitting it."))
    B.append(H(2, "2.2. Turkish text classification"))
    B.append(P(
        "Turkish is agglutinative: a stem carries chains of inflectional and derivational suffixes, so the surface vocabulary is large and sparse. "
        "Preprocessing therefore matters more than in English (Uysal & Gunal, 2014), and stemming or lemmatisation with a morphological analyser such as "
        "Zemberek (Akın & Akın, 2007) is standard practice. Benchmark datasets such as TTC-3600 (Kılınç et al., 2017) established that feature-selected "
        "bag-of-words models are competitive on Turkish news categorisation, and later studies on Turkish legal texts (Akça et al., 2022) and sentiment "
        "(Karakaş & Şimşek, 2026) found domain-adapted or fine-tuned BERTurk models (Schweter, 2020) ahead of both classical models and prompted LLMs. "
        "Whether these conclusions transfer to short, templated administrative correspondence with organisational units as classes has not been examined, "
        "and the earlier version of this study side-stepped the morphology question by translating the documents to English, a design whose cost and "
        "benefit are quantified here."))
    B.append(H(2, "2.3. Sentence encoders and fine-tuned transformers"))
    B.append(P(
        "Static word embeddings (Mikolov et al., 2013; Pennington et al., 2014; Bojanowski et al., 2017) represent words independently of context and are "
        "usually averaged to obtain a document vector. Transformer encoders (Vaswani et al., 2017; Devlin et al., 2019; Liu et al., 2019; Lan et al., 2020) "
        "produce contextual representations, but their raw token embeddings are poor sentence representations unless the encoder is trained for that purpose. "
        "Sentence-BERT (Reimers & Gurevych, 2019) and the Universal Sentence Encoder (Cer et al., 2018; Yang et al., 2020) made sentence-level similarity "
        "training the norm, and multilingual encoders obtained by knowledge distillation (Reimers & Gurevych, 2020), by translation-pair training (Feng et al., "
        "2022; Conneau et al., 2020) or by large-scale contrastive pre-training (Wang et al., 2024; Chen et al., 2024; Zhang et al., 2025) now cover Turkish "
        "directly. The MTEB benchmark shows that no encoder dominates across tasks (Muennighoff et al., 2023), which argues for evaluating several on the "
        "target task. Frozen encoders with a light classifier and end-to-end fine-tuning (Sun et al., 2019) are two distinct operating points in cost and "
        "accuracy, and both are included here. Between the two lies efficient few-shot fine-tuning such as SetFit, which adapts a sentence encoder "
        "contrastively on a few labelled pairs and trains a light classification head (Tunstall et al., 2022); it targets settings with a handful of "
        "examples per class, whereas the archive studied here provides hundreds to thousands of labelled documents for most units (Section 3.3)."))
    B.append(H(2, "2.4. Large language models for classification and cost-aware cascades"))
    B.append(P(
        "LLMs classify text from an instruction and, optionally, a few examples (Brown et al., 2020; Sun et al., 2023), and they can match crowd annotators "
        "on some labelling tasks (Gilardi et al., 2023). Their accuracy on institution-specific label sets is, however, limited by the fact that the "
        "classes are defined by internal practice: a model cannot know which of three IT sub-units handles a password request unless it is told, either "
        "through descriptions or through retrieved examples. Retrieval-augmented prompting (Lewis et al., 2020; Liu et al., 2022) addresses this by showing "
        "the model similar past documents and their labels. Because LLM calls are slow and billed per token, cascades that resolve easy inputs with a cheap "
        "model and forward the rest to a stronger one (Chen et al., 2023) and abstention policies tuned under explicit error costs (Gupta, 2026) have become "
        "the practical way to use LLMs in classification pipelines. The present study follows this pattern: the LLM is a second stage for documents on "
        "which the local classifier is uncertain, and it only receives de-identified text."))
    B.append(H(2, "2.5. Evaluation practice: leakage, temporal validity and selective prediction"))
    B.append(P(
        "Random splits over-estimate the performance of text classifiers when documents are duplicated or temporally dependent (Gorman & Bedrick, 2019; "
        "Søgaard et al., 2021), and choosing a configuration on the test split is a form of leakage that becomes severe when many configurations are "
        "compared (Kapoor & Narayanan, 2023). Accuracy and weighted F1 hide the behaviour of small classes, so macro-averaged scores and per-class reports "
        "are recommended for imbalanced problems (Sokolova & Lapalme, 2009; Grandini et al., 2020), and differences between classifiers should be tested "
        "statistically (Dietterich, 1998; Demšar, 2006). For deployment, a classifier should also be able to abstain: selective prediction with a "
        "confidence threshold (Chow, 1970; El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017), learning to defer to an expert (Madras et al., 2018; "
        "Mozannar & Sontag, 2020) and the evaluation of failure detection by risk-coverage curves (Jaeger et al., 2023) provide the tools used in Section 4.5. "
        "Thresholds on confidence presuppose calibrated scores: neural classifiers, pre-trained transformers included, tend to be over-confident and are "
        "commonly corrected by temperature scaling (Guo et al., 2017; Desai & Durrett, 2020); calibration is measured by the expected calibration error "
        "(Naeini et al., 2015) and is known to deteriorate under dataset shift (Ovadia et al., 2019). Where the classes form a hierarchy (Silla & Freitas, "
        "2011), an uncertain fine-grained decision can be replaced by a coarser but more reliable one, trading specificity for accuracy (Deng et al., "
        "2012); in routing, the coarser decision is the department instead of the unit. Finally, decision support in public administration should remain "
        "auditable: interpretable models are preferable where they are competitive (Rudin, 2019), and showing similar past cases is an established form "
        "of explanation (Kenny et al., 2021). The protocol of this study was designed to satisfy these recommendations simultaneously."))

    # ---------------------------------------------------------------- 3 problem and data
    B.append(H(1, "3. Problem definition and dataset"))
    B.append(H(2, "3.1. Routing in the EDMS of " + ("a Turkish public university" if ANON else "Düzce University")))
    B.append(P(
        ("The university" if ANON else "Düzce University") + " operates a nationwide-standard EDMS in which every official document is registered with a subject line (*EvrakKonusu*) and a body "
        "(*EvrakIcerigi*) and then dispatched by a user to one or more recipients, each dispatch being recorded with its sender unit, recipient unit, "
        "recipient role (*for action* or *for information*) and timestamp. The data used here are the dispatch records of the Information Technologies "
        "Department (BİDB) and the Health, Culture and Sports Department (SKS), their sub-units (şube müdürlükleri) and the Rectorate, exported for "
        f"this study and covering the dispatches recorded from {pd.Timestamp(st['dispatch_time_min']):%B %Y} to "
        f"{pd.Timestamp(st['dispatch_time_max']):%B %Y} ({small(st['n_documents_registered_before_dispatch_period'])} of the documents had been registered "
        "in the last months of 2019). The two departments form a two-level hierarchy "
        "(department office and four to six sub-units each), and the Rectorate appears both as a sender that forwards documents downwards and as the "
        "addressee of documents sent upwards."))
    B.append(P(
        "The routing task is defined as follows. Given the subject and body of a document, predict the unit to which the document is dispatched *for action* "
        "in the first recorded dispatch, i.e. the first routing decision that a human made for that document. Documents that pass through a chain "
        "(for example Rectorate → BİDB office → Software Unit) are labelled with the first target unit, and the set of all units that received the document "
        "for action is kept for a lenient evaluation in which any of them is accepted. This is a single-label formulation with a set-valued reference, which "
        "matches the decision-support use case: the system proposes units to the person who dispatches, and the first dispatch is the decision it supports."))
    B.append(H(2, "3.2. Dataset construction"))
    rem = st["n_dispatch_records_raw"] - st["n_records_with_target_unit"]
    B.append(P(
        f"⟦T:construction⟧ lists every step from the raw export to the document table; all counts are produced by the released script. The export contains "
        f"{st['n_dispatch_records_raw']:,} dispatch records of {st['n_documents_raw']:,} documents. Records without a recipient unit ({rem:,}, mostly "
        f"dispatches for information only) are removed. Subject and body are concatenated; {st['n_records_empty_content']:,} records have an empty body "
        f"and are represented by their subject line only. Identical texts that were dispatched to more than {st['broadcast_threshold_units']} distinct "
        f"units are broadcasts rather than routing decisions: {st['n_broadcast_texts']} such texts (median {st['broadcast_units_per_text_median']:.0f} "
        f"recipient units each) account for {st['n_broadcast_records_removed']:,} records and {st['n_broadcast_documents_removed']:,} documents "
        f"({pct(st['n_broadcast_documents_removed']/st['n_documents_with_target_unit'])}% of the {st['n_documents_with_target_unit']:,} documents with a recipient unit). "
        f"They are of two kinds: {st['n_broadcast_documents_empty_body']:,} subject-only records, {st['n_broadcast_documents_bare_petition']:,} of them bare "
        f"*DİLEKÇE* (petition) entries whose content exists only as a scanned attachment, and {st['n_broadcast_documents_with_body']:,} circulars with a body "
        f"(COVID-19 measures, satisfaction surveys, maintenance notices, Senate decisions, working-hour announcements) that were sent to all units alike. "
        f"No single routing decision exists for them, so they are excluded; this is the main reason why the document table is smaller than the export. "
        f"One row per document is then formed from the chronologically first dispatch, giving {n_docs:,} "
        f"documents; {st['n_documents_multi_unit']:,} of them ({pct(st['n_documents_multi_unit']/n_docs)}%) were later forwarded to at least one further "
        f"unit, and {st['n_documents_self_dispatch']:,} first dispatches are internal to the sender's own unit."))
    B.append({"type": "table", "caption": "⟦T:construction⟧. Construction of the document-level dataset from the EDMS dispatch log.",
              "columns": ["Step", "Records", "Documents"], "col_widths": [0.6, 0.2, 0.2], "font_size": 18, "rows": [
                  ["Raw export (one row per dispatch)", f"{st['n_dispatch_records_raw']:,}", f"{st['n_documents_raw']:,}"],
                  ["Dispatched for action to a unit (recipient unit present)", f"{st['n_records_with_target_unit']:,}", f"{st.get('n_documents_with_target_unit', 0):,}"],
                  [f"Broadcast filter: identical text sent to > {st['broadcast_threshold_units']} units removed ({st['n_broadcast_texts']} texts; "
                   f"{st.get('n_broadcast_documents_empty_body', 0):,} subject-only + {st.get('n_broadcast_documents_with_body', 0):,} circulars)",
                   f"{st['n_records_after_broadcast_filter']:,}", f"{n_docs:,}"],
                  ["One row per document (label = unit of the first dispatch; all acting units kept as a set)", "–", f"{n_docs:,}"],
                  ["Chronological partitions: train (≤ 2021) / validation (2022 H1) / test (2022 H2)", "–", f"{sp['train']:,} / {sp['val']:,} / {sp['test']:,}"],
              ]})
    B.append(H(2, "3.3. Characteristics of the data"))
    cc = R["dataset_classes"]
    B.append(P(
        f"⟦F:dataset⟧ and ⟦T:classes⟧ show the class distribution and the chronological partitions. The {n_cls} units are strongly imbalanced: the Software Unit "
        f"of BİDB receives {pct(maj)}% of all documents, whereas the Health Services Unit has {min(classes.values())} documents in total and only one in the "
        f"test split, so its per-unit scores are not interpretable and it is reported for completeness. Documents are short (mean {st['words_mean']:.1f} words, "
        f"median {st['words_median']:.0f}, 90th percentile {st['words_p90']:.0f}, maximum {st['words_max']}); the subject line has {st['subject_words_mean']:.1f} "
        f"words on average. The correspondence is highly templated: after normalisation only {st['n_unique_texts']:,} distinct texts remain among the {n_docs:,} "
        f"documents, and {st['n_documents_with_identical_twin']:,} documents ({pct(st['n_documents_with_identical_twin']/n_docs)}%) have at least one textually "
        f"identical twin with a different document identifier (standard forms such as leave requests, proxy assignments or vehicle allocations). Identical "
        f"texts are not always routed to the same unit: assigning every text its majority unit would reach {pct(st['label_consistency_ceiling'])}% accuracy, "
        f"which is the ceiling for any deterministic function of the text and an estimate of the irreducible label noise of the task. In the chronological "
        f"split, {st['n_test_docs_with_identical_text_in_train']} test documents ({pct(st['n_test_docs_with_identical_text_in_train']/sp['test'])}%) have an "
        f"identical text in the training partition; they are kept because recurring forms are part of the real routing workload, and the grouped "
        f"cross-validation of Section 4.6 provides the complementary estimate in which no identical text crosses a fold boundary."))
    if cc is not None:
        rows = [[SHORT[r["unit"]] + (f" ({r['department']})" if r["department"] != "Rektörlük" else ""), f"{r['train']:,}", f"{r['val']:,}", f"{r['test']:,}", f"{r['total']:,}", f"{r['share_%']:.1f}"] for _, r in cc.iterrows()]
        B.append({"type": "table", "caption": "⟦T:classes⟧. Documents per target unit and chronological partition (BİDB: Information Technologies Department; SKS: Health, Culture and Sports Department).",
                  "columns": ["Target unit", "Train", "Validation", "Test", "Total", "Share (%)"], "col_widths": [0.4, 0.12, 0.12, 0.12, 0.12, 0.12], "font_size": 16, "rows": rows})
    B.append(fig_block("fig_dataset", "⟦F:dataset⟧. (a) Documents per target unit; (b) monthly document volume with the boundaries of the chronological partitions."))
    B.append(H(2, "3.4. Ethics, privacy and availability"))
    B.append(P(
        "The records contain names of staff and of third parties in subject lines and bodies, so the dataset cannot be released publicly; the released "
        "code contains the construction script, the document identifiers of every partition and fold, all prediction files, and the de-identification "
        "procedure, so that the experiments can be reproduced by anyone with access to the export and the evaluation can be audited without it. The "
        "export was provided by the university for anonymised use in academic work (see the Ethics statement). For this study it was stored and "
        "processed on cloud services: Google Drive and Google Colab for the experiments, and the cloud workspace of the AI assistant named in the "
        "declaration on generative AI. To limit the disclosure of personal data (Republic of Türkiye, 2016), all reported models were trained and "
        "queried on de-identified text (Section 4.2), and only de-identified text was sent to the external LLM services; the English variant of "
        "Section 4.2 re-uses machine translations of the document bodies produced with an online translation service in the earlier study."))

    # ---------------------------------------------------------------- 4 framework
    B.append(H(1, "4. Proposed framework"))
    B.append(H(2, "4.1. Overview"))
    B.append(P(
        "⟦F:pipeline⟧ shows the framework. The data layer produces the de-identified document table and the partitions of Section 3. The representation layer maps a document "
        "to a vector with one of three families: lexical TF-IDF features on Turkish-native preprocessing variants, frozen sentence encoders, or an encoder "
        "fine-tuned end to end. The decision layer trains a classifier (or uses the fine-tuned head) and outputs a ranked list of units with a confidence "
        "score. The routing layer uses the confidence: documents above a threshold are routed automatically (with the top-k list as suggestion), documents "
        "below it are either shown to staff or, in the cascade variant, passed to an LLM that sees the de-identified document together with similar past "
        "documents and their units. All thresholds are chosen on the validation partition."))
    B.append(fig_block("fig_pipeline", "⟦F:pipeline⟧. The confidence-aware routing framework. Solid arrows: data flow; the dashed arrow marks documents whose confidence is too low for any automatic decision."))
    B.append(H(2, "4.2. De-identification and Turkish-native preprocessing"))
    prep = R["prep"]
    dt_ = deid.get("text", {})
    B.append(P(
        "*De-identification.* All text fields are de-identified before any representation is computed. Person names in the subject and body are "
        "replaced by the placeholder [KİŞİ] (person) with a Turkish named-entity recogniser (savasy/bert-base-turkish-ner-cased), and e-mail "
        "addresses, telephone numbers and 11-digit national identity numbers by [E-POSTA], [TELEFON] and [TCKN] with regular expressions. In the "
        f"concatenated subjects and bodies, {dt_.get('[KİŞİ]', 0):,} names were replaced, and {deid.get('documents_with_a_person_name', 0):,} of the "
        f"{n_docs:,} documents ({pct(deid.get('documents_with_a_person_name', 0) / n_docs)}%) contained at least one; {dt_.get('[E-POSTA]', 0)} e-mail "
        f"addresses, {dt_.get('[TELEFON]', 0)} telephone numbers and {dt_.get('[TCKN]', 0)} identity numbers were masked. The English translations "
        "used below are masked with the names found in the Turkish original and, in addition, by an English recogniser (dslim/bert-base-NER; "
        "placeholder [PERSON]). The de-identified text is the only input of every reported model, of the retrieval of examples for the LLM prompts "
        "and of the LLMs themselves. For the lexical models each placeholder is mapped to a reserved token, so that the presence of a name remains "
        "available as a feature but not its identity."))
    B.append(P(
        "Three lexical variants are produced from the concatenated subject and body. *Raw*: Turkish-aware lower-casing (İ→i, I→ı), removal of all "
        "non-letter characters (dates, numbers and identifiers carry no routing information), and removal of the 53 NLTK Turkish stop words extended by "
        "closing formulas of official correspondence (*arz ederim*, *rica ederim*, *bilgilerinize*). *Stem*: the raw tokens stripped by the Snowball "
        "Turkish stemmer, a suffix-stripping algorithm without a lexicon. *Lemma*: the raw tokens lemmatised with Zeyrek, a Python port of the Zemberek "
        "morphological analyser (Akın & Akın, 2007); because Zeyrek performs no contextual disambiguation, the longest stem among the candidate analyses is "
        f"taken. The variants differ in vocabulary size (raw {prep.get('raw', 0):,}, stem {prep.get('stem', 0):,}, lemma {prep.get('lemma', 0):,} types on the "
        "full corpus), which quantifies the sparsity that morphology introduces. Stemming and lemmatisation are thus distinct operations: the former truncates "
        "surface forms by rules, the latter returns dictionary forms through morphological analysis. A fourth variant, *English*, re-uses the Google-Translate "
        "output of the earlier study for the document body, de-identified as described above; it is evaluated on the body only, against the Turkish body "
        "processed identically, so that the effect of translation is isolated. Contextual encoders receive the cleaned text with its original casing and punctuation, because their sub-word "
        "tokenisers were trained on natural text and stemming would destroy the forms they expect."))
    B.append(H(2, "4.3. Representations"))
    B.append(P(
        "*Lexical.* TF-IDF vectors of word unigrams and bigrams (minimum document frequency 2, sub-linear term frequency, L2 normalisation) and, separately, "
        "of character 2–5-grams within word boundaries, which are robust to spelling variation and suffixes without any morphological tool; the union of both "
        "is also evaluated. The vocabulary and inverse document frequencies are always fitted on the training rows of the partition or fold in use."))
    B.append(P(
        "*Frozen sentence encoders.* Each document is encoded once with a pre-trained encoder and the resulting vector is used as a fixed feature. ⟦T:encoders⟧ "
        "lists the encoders: multilingual Sentence-BERT models obtained by distillation (paraphrase-multilingual-MiniLM-L12-v2 and -mpnet-base-v2; Reimers & "
        "Gurevych, 2020), the distilled multilingual Universal Sentence Encoder (distiluse-base-multilingual-cased-v2), LaBSE (Feng et al., 2022), "
        "multilingual-E5 base and large (Wang et al., 2024), BGE-M3 (Chen et al., 2024), Qwen3-Embedding 0.6B and 4B (Zhang et al., 2025), a Turkish "
        "Sentence-BERT fine-tuned on Turkish NLI and STS data (BERTurk-SBERT), and BERTurk itself with mean pooling and no sentence-level training, which "
        "isolates the effect of sentence training. Texts are truncated to 256 sub-word tokens (longer than the 90th percentile of document length); "
        "instruction-tuned encoders receive their recommended query prefix. The English Sentence-BERT model all-mpnet-base-v2 is applied to the translated "
        "bodies for the translation ablation."))
    B.append(P(
        "*Fine-tuned encoders.* BERTurk (dbmdz/bert-base-turkish-cased; Schweter, 2020) and XLM-RoBERTa base (Conneau et al., 2020) are fine-tuned end to "
        "end with a linear classification head on the [CLS] representation, cross-entropy loss, AdamW with weight decay 0.01, linear warm-up over 10% of the "
        "steps, batch size 32 and maximum length 256. The learning rate (2×10^{-5} or 5×10^{-5}) and the number of epochs (1–6) are selected on the "
        "validation partition with seed 0; the selected configuration is then retrained on training plus validation documents with three seeds and "
        "evaluated once on the test partition, so that seed variance is reported."))
    B.append({"type": "table", "caption": "⟦T:encoders⟧. Pre-trained encoders. Dim.: embedding dimension; pooling as implemented in the released checkpoints.",
              "columns": ["Model (checkpoint)", "Training signal", "Languages", "Dim.", "Params"], "col_widths": [0.36, 0.3, 0.14, 0.08, 0.12], "font_size": 16, "rows": [
                  ["paraphrase-multilingual-MiniLM-L12-v2", "distillation of paraphrase SBERT", "50+", "384", "118M"],
                  ["paraphrase-multilingual-mpnet-base-v2", "distillation of paraphrase SBERT", "50+", "768", "278M"],
                  ["distiluse-base-multilingual-cased-v2", "distillation of multilingual USE", "50+", "512", "135M"],
                  ["LaBSE", "translation ranking", "109", "768", "471M"],
                  ["multilingual-e5-base / -large", "contrastive pre-training + supervised fine-tuning", "100", "768 / 1024", "278M / 560M"],
                  ["BGE-M3", "self-knowledge distillation, multi-granularity", "100+", "1024", "568M"],
                  ["Qwen3-Embedding-0.6B / -4B", "instruction-tuned contrastive training", "100+", "1024 / 2560", "0.6B / 4B"],
                  ["BERTurk-SBERT (emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)", "Turkish NLI + STS fine-tuning of BERTurk", "tr", "768", "111M"],
                  ["BERTurk mean-pooled (dbmdz/bert-base-turkish-cased)", "masked language modelling only", "tr", "768", "111M"],
                  ["all-mpnet-base-v2 (translation ablation)", "contrastive training on 1B English pairs", "en", "768", "110M"],
              ]})
    B.append(H(2, "4.4. Classifiers and hyper-parameter selection"))
    B.append(P(
        "Every lexical or frozen representation is combined with the classifier families of the earlier study so that the comparison remains "
        "interpretable: logistic regression (LR), support vector machines (Cortes & Vapnik, 1995) with a linear kernel on the sparse TF-IDF features and "
        "an RBF kernel on the dense embeddings, k-nearest neighbours with cosine distance (KNN), random forest (Breiman, 2001) (RF), decision tree (DT) "
        "and naive Bayes (complement NB for sparse, Gaussian NB for dense features). Dense features are standardised before the linear and kernel models. "
        "In a pilot on the first two encoders, a linear SVM on dense features needed up to 20 minutes per fit for the larger regularisation values "
        "and reached a validation macro-F1 within about 0.01 of LR and clearly below the RBF SVM, so on dense features the linear family is "
        "represented by LR. ⟦T:grid⟧ gives the search grid; the "
        "regularisation strength and the class weighting (none or inverse frequency) are the hyper-parameters that matter most. For every representation "
        "and family the configuration with the highest validation macro-F1 is kept, and the family with the highest validation macro-F1 is the "
        "representation's *selected system*. Test results are reported for all families, but every claim about a representation refers to its selected "
        "system, so that no choice is informed by the test partition."))
    B.append({"type": "table", "caption": "⟦T:grid⟧. Hyper-parameter grid (selected on the validation partition by macro-F1).",
              "columns": ["Family", "Grid"], "col_widths": [0.22, 0.78], "font_size": 16, "rows": [
                  ["LR", "C ∈ {0.1, 1, 10}; class weight ∈ {none, balanced}; L-BFGS, up to 5,000 iterations"],
                  ["Linear SVM (sparse)", "C ∈ {0.1, 0.5, 1, 5}; class weight ∈ {none, balanced}"],
                  ["RBF SVM (dense)", "C ∈ {1, 10}; γ = scale; class weight ∈ {none, balanced}"],
                  ["KNN", "k ∈ {1, 5, 15}; cosine distance; distance weighting"],
                  ["RF", "200 trees; √p features per split; minimum leaf ∈ {1, 3}; class weight ∈ {none, balanced subsample}"],
                  ["DT", "minimum leaf ∈ {1, 5}"],
                  ["NB", "complement NB α ∈ {0.1, 0.5, 1} (sparse); Gaussian NB (dense)"],
                  ["Fine-tuned encoders", "learning rate ∈ {2×10^{-5}, 5×10^{-5}}; epochs 1–6 (best validation epoch); 3 seeds for the final model"],
              ]})
    B.append(H(2, "4.5. Confidence-aware routing"))
    B.append(P(
        "*Top-k suggestion.* The decision layer returns the units ranked by class score; top-1 is the automatic decision and the top-3 list is what a "
        "dispatcher would see as a suggestion. Top-3 accuracy and the lenient accuracy (top-1 contained in the set of all units that acted on the document) "
        "quantify the value of the suggestion list."))
    B.append(P(
        "*Deferral to staff.* The confidence of a decision is the probability of the top-1 unit (LR, RF, NB, KNN and the fine-tuned encoders output "
        "probabilities, which need not be calibrated (Guo et al., 2017) because only the ranking enters the threshold; for SVMs the decision values are passed through a softmax, which preserves the ranking of documents by margin). Sorting test "
        "documents by confidence yields the risk-coverage curve of selective prediction (Geifman & El-Yaniv, 2017); its area (AURC; Jaeger et al., 2023) "
        "summarises how well confidence separates correct from incorrect decisions, and the operating point for a target accuracy (90% or 95% on the "
        "automatically routed documents) is the largest validation coverage that meets the target. The corresponding threshold is then applied unchanged "
        "to the test partition, which gives an honest estimate of the automation rate and of the accuracy on the automated share. For the fine-tuned "
        "BERTurk, the deferral, calibration, back-off and cascade analyses and the McNemar tests use the predictions of the first of its three seeds "
        "(seed 0); Section 5.4 reports how the tests change with the other two seeds."))
    B.append(P(
        "*Calibration.* A threshold expressed as a probability is meaningful only if confidences are calibrated. A single temperature T is therefore "
        "fitted on the validation predictions by minimising the negative log-likelihood (temperature scaling; Guo et al., 2017) and applied to the test "
        "predictions; for the linear SVM, the decision values play the role of logits. Because the test predictions come from the model refitted on "
        "training plus validation documents, T is transferred from the model that produced the validation scores. Calibration is measured by the expected "
        "calibration error (ECE) over 15 equal-width confidence bins (Naeini et al., 2015). Three deferral rules are compared on the calibrated scores: "
        "the empirical rule above; a *calibrated rule* that routes a document automatically when its calibrated confidence reaches the target accuracy; "
        "and a *recency rule* that chooses the empirical threshold on the last quarter of the validation half-year only, as a deployment that "
        "recalibrates on recent documents would do."))
    B.append(P(
        "*Hierarchical back-off.* In the EDMS, the office of a department (BİDB or SKS) receives documents for the department and forwards them to its "
        "units. A document whose unit is uncertain but whose department is clear can therefore still be routed automatically, to the office of that "
        "department (Deng et al., 2012). The department confidence is the sum of the calibrated probabilities of the department's units. A document is "
        "routed to the predicted unit if the unit confidence reaches t_{u}; otherwise to the office of the predicted department if the department "
        "confidence reaches t_{d}; otherwise it is left to staff. The two thresholds are chosen on validation as the largest coverage that meets the "
        "target accuracy at their level, unit or department. A back-off decision counts as correct when the document reaches the right department; "
        "it then needs one forwarding step inside the department, unless the office itself was the target. The Rectorate forms a department of its "
        "own, so for it the two levels coincide."))
    B.append(P(
        "*LLM second stage.* Documents below the threshold can be passed to an LLM instead of to staff. The prompt (Appendix B) contains the list of the "
        f"{n_cls} units with a one-sentence description of their duties, written from the units' functions and the document types observed in the training "
        "partition, and asks for the three most likely units as JSON. In the *zero-shot* mode the model sees only the document; in the "
        "*retrieval-augmented* mode (rag10) it also sees the ten most similar documents of the training pool (TF-IDF cosine similarity; the pool is the "
        "training partition for validation documents and training plus validation for test documents) together with the units they were dispatched to. "
        "Both the document and the retrieved examples are given in their de-identified form (Section 4.2), and the retrieval itself is computed on the "
        "de-identified texts."))
    B.append(P(
        "Three models of three providers were queried through their public APIs in November 2026: Claude Haiku 5.5 (Anthropic), GPT-6 Luna (OpenAI) "
        "and Gemini 3.8 Flash (Google). Extended reasoning was switched off for all of them: thinking budget 0 and temperature 0 for Gemini; reasoning "
        "effort *none* and a strict JSON schema whose items are restricted to the unit names for GPT-6 Luna; and thinking disabled for Claude Haiku. "
        "Claude and GPT used the provider-default sampling temperature, and outputs were limited to 300 tokens. "
        "The first JSON object in the answer is parsed; a returned name that is not one of the units (accent-insensitive string similarity below 0.85, "
        "e.g. a personnel department outside the list) is discarded, and a document whose answer contains no valid unit (a truncated output, a refusal, "
        "or only units outside the list, mostly for documents whose de-identified text is little more than a name placeholder) is requested again, up "
        "to three times; if no valid unit is obtained, the majority unit of the training data is predicted. In a first run, the examples of the "
        "retrieval-augmented prompts were selected by a retrieval index built on the original text, while the prompts themselves already contained "
        "only de-identified text; after the index had been rebuilt on the de-identified text, every prompt whose examples or their order changed "
        f"({pct(rr_share)}% of the test prompts) was asked again, and only the answers to the final prompts are reported. A fourth model, Claude Sonnet 5.5, took part in the first run only; it could not be "
        "queried again within the budget of the study and is therefore not reported. All calls, including repeated ones, are counted in the costs, "
        "and the effective setting is stored with every response."))
    B.append(P(
        "*Cascade.* The cascade is evaluated as in Chen et al. (2023): documents are sent to the LLM in increasing order of local confidence, the share "
        "sent is chosen on the validation partition (the smallest share whose validation accuracy is within one point of the LLM alone, and the share "
        "with the highest validation accuracy), and the resulting accuracy, token cost and latency are reported on the test partition."))
    B.append(H(2, "4.6. Evaluation protocol and metrics"))
    B.append(P(
        f"*Partitions.* Documents dated up to 31 December 2021 form the training partition ({sp['train']:,} documents), the first half of 2022 the validation "
        f"partition ({sp['val']:,}) and the second half of 2022 the test partition ({sp['test']:,}). Every model is selected on the validation partition, "
        "refitted on training plus validation with the selected configuration, and evaluated exactly once on the test partition. The chronological design "
        "estimates what an institution would observe after deploying a model trained on its archive; it also exposes drift, since the composition of "
        "correspondence changes over the semesters. As a complementary estimate, a stratified group 5-fold cross-validation with two repeats is run over "
        "all documents with the selected configuration, where the groups are the normalised texts, so that identical documents never appear on both sides "
        "of a fold; its mean and standard deviation over the ten folds are reported next to the test results."))
    B.append(P(
        "*Metrics.* For a document with true unit y and predicted unit ŷ, accuracy is the share of documents with ŷ = y. Precision, recall and F1 are "
        "computed per unit from the one-versus-rest confusion counts of that unit (Eqs. 1–3) and aggregated by macro-averaging (unweighted mean over the "
        f"{n_cls} units, Eq. 4), which gives every unit the same weight, and by support-weighting (weighted F1), which is reported for comparability with "
        "the earlier study only; weighted recall equals accuracy in single-label classification and is therefore not reported. Balanced accuracy is the "
        "macro-averaged recall. Department-level accuracy counts a prediction as correct when it falls in the same department (BİDB, SKS or Rectorate) as "
        "the true unit and separates within-department from cross-department errors."))
    B.append({"type": "equation", "text": "Precision_{c} = TP_{c} / (TP_{c} + FP_{c})", "label": "(1)"})
    B.append({"type": "equation", "text": "Recall_{c} = TP_{c} / (TP_{c} + FN_{c})", "label": "(2)"})
    B.append({"type": "equation", "text": "F1_{c} = 2 · Precision_{c} · Recall_{c} / (Precision_{c} + Recall_{c})", "label": "(3)"})
    B.append({"type": "equation", "text": f"Macro-F1 = (1/{n_cls}) Σ_{{c}} F1_{{c}}", "label": "(4)"})
    B.append(P(
        "Here TP_{c} is the number of documents of unit c predicted as c, FP_{c} the number of documents of other units predicted as c, and FN_{c} the number "
        "of documents of unit c predicted as another unit. *Uncertainty and tests.* Test-partition accuracy and macro-F1 are accompanied by 95% percentile "
        "bootstrap intervals over documents (2,000 resamples; Efron & Tibshirani, 1994). Pairs of systems are compared on the test partition with the exact "
        "McNemar test on the discordant documents (Dietterich, 1998) with Holm correction for multiple comparisons (Holm, 1979). *Cost.* Training time on "
        "the training-plus-validation documents and inference latency per document are measured for every system on the hardware stated in Section 5; LLM "
        "cost is computed from the token usage reported by the APIs at list prices."))

    # ---------------------------------------------------------------- 5 results
    B.append(H(1, "5. Results"))
    env = json.loads((RES / "environment.json").read_text()) if (RES / "environment.json").exists() else {}
    B.append(P(
        "*Experimental setup.* Models were trained and evaluated on Google Colab: lexical models and classifier grids on CPU"
        + (f" ({env.get('cpu_count', '?')} cores)" if env else "")
        + ", embedding extraction, fine-tuning and de-identification on "
        + (f"a {env.get('gpu')} GPU" if env.get("gpu") else "a GPU")
        + "; LLMs were accessed through their public APIs"
        + (f" in {env.get('date', 'November 2026')}" if env else "")
        + ". Fine-tuning used 16-bit mixed precision; the classifier grids of the frozen encoders ran as four parallel jobs with two threads each. "
        + "Software: Python " + env.get("python", "3.x") + ", scikit-learn " + env.get("sklearn", "1.x") + ", PyTorch " + env.get("torch", "2.x")
        + ", Transformers " + env.get("transformers", "4.x/5.x") + ", sentence-transformers " + env.get("sentence_transformers", "3.x")
        + ". All random seeds are fixed and listed in the released code, which was written with the assistance of an AI coding assistant (Claude, "
        + "Anthropic; see the declaration on generative AI) under the authors' direction. The LLM experiments with the three reported models, "
        + "including validation runs, the first run of Section 4.5 and repeated requests, cost "
        + f"USD {sum(usd_total.values()):.2f} at list prices."))
    B.append(H(2, "5.1. Main comparison"))
    if main is not None:
        rows = []
        for r in main.itertuples():
            cvs = f"{100*r.cv_accuracy_mean:.1f} ± {100*r.cv_accuracy_sd:.1f}" if not pd.isna(getattr(r, "cv_accuracy_mean", np.nan)) else "–"
            rows.append([r.system, f"{100*r.accuracy:.1f} {ci(r.accuracy_ci)}", f"{100*r.macro_f1:.1f} {ci(r.macro_f1_ci)}", f"{100*r.weighted_f1:.1f}",
                         f"{100*r.balanced_accuracy:.1f}", f"{100*r.top3_accuracy:.1f}" if not pd.isna(r.top3_accuracy) else "–",
                         f"{100*r.lenient_accuracy:.1f}" if not pd.isna(r.lenient_accuracy) else "–", f"{100*r.department_accuracy:.1f}", cvs])
        B.append({"type": "table", "caption": "⟦T:main⟧. Test-partition results of the system selected on validation for every representation (sorted by macro-F1). "
                  "Accuracy and macro-F1 with 95% bootstrap intervals; Top-3: true unit among the three highest-scored units; Lenient: top-1 among all units that acted on the document; "
                  "Dept.: department-level accuracy; CV: grouped 5×2 cross-validation accuracy, mean ± SD over ten folds (same configuration). All values in %.",
                  "columns": ["System", "Accuracy", "Macro-F1", "wF1", "Bal. acc.", "Top-3", "Lenient", "Dept.", "CV acc."],
                  "col_widths": [0.3, 0.13, 0.13, 0.07, 0.08, 0.07, 0.07, 0.07, 0.08], "font_size": 14, "rows": rows})
        sp_tc = ex.get("spearman_test_vs_cv_accuracy", {}); sp_vt = ex.get("spearman_val_vs_test_macro_f1", {}); gap = ex.get("cv_minus_test_accuracy_points", {})
        ftr = pd.read_csv(RES / "results_ft.csv")
        ft_sd = 100 * ftr[ftr["representation"] == "ft_bert-base-turkish-cased"]["accuracy"].std(ddof=1)
        n_emb_below = int((emb_rows["accuracy"] < lex["accuracy"]).sum()) if emb_rows is not None else 0
        top5 = main.loc[main["group"] != "LLM", "macro_f1"].nlargest(5)
        drop = (main["val_macro_f1"] - main["macro_f1"]); big = main.loc[drop.idxmax()]
        B.append(P(
            f"⟦T:main⟧ and ⟦F:main⟧ summarise the main comparison; the full grid with every classifier family is given in Appendix A. The "
            f"retrieval-augmented Gemini 3.8 Flash had the highest validation macro-F1 ({f3(gem['val_macro_f1'])}) and is therefore the final system "
            f"under the protocol; it also had the highest test accuracy, {pct(gem['accuracy'])}% (95% CI {ci(gem['accuracy_ci'])}), and macro-F1, "
            f"{pct(gem['macro_f1'])}% (CI {ci(gem['macro_f1_ci'])}). GPT-6 Luna and Claude Haiku 5.5 with retrieval reached {pct(gpt['accuracy'])}% and "
            f"{pct(hai['accuracy'])}% ({pct(gpt['macro_f1'])}% and {pct(hai['macro_f1'])}% macro-F1). Among the local models, the fine-tuned BERTurk had "
            f"the highest validation and test macro-F1 ({pct(ft['accuracy'])}% accuracy, {pct(ft['macro_f1'])}% macro-F1, mean of three seeds with a "
            f"standard deviation of {ft_sd:.1f} points in accuracy), and the largest frozen encoder, Qwen3-Embedding-4B with an RBF SVM, the highest "
            f"test accuracy ({pct(qwen['accuracy'])}%, macro-F1 {pct(qwen['macro_f1'])}%), only {100*(gem['accuracy']-qwen['accuracy']):.1f} points "
            f"below Gemini. The lexical system selected on validation, {lex['system']}, reached {pct(lex['accuracy'])}% and {pct(lex['macro_f1'])}%, "
            f"i.e. {100*(ft['accuracy']-lex['accuracy']):.1f}–{100*(qwen['accuracy']-lex['accuracy']):.1f} accuracy points below the best local "
            f"encoders and with practically the same macro-F1 as BERTurk. Fine-tuned XLM-R ({pct(xlmr['accuracy'])}%, {pct(xlmr['macro_f1'])}%) "
            f"stayed below BERTurk, and {small(n_emb_below)} of the {len(emb_rows)} frozen encoders, among them the distilled multilingual USE that was the "
            f"best model of the earlier study ({pct(muse['accuracy'])}%, {pct(muse['macro_f1'])}%), were less accurate than TF-IDF. The gap between "
            f"accuracy and macro-F1 is large for every system, which reflects the skewed class distribution: the three smallest units contribute "
            f"little to accuracy but a quarter of the macro average. The five best local models lie within {100*(top5.max()-top5.min()):.1f} "
            f"macro-F1 points of each other ({f3(top5.min())}–{f3(top5.max())}), whereas Gemini is {100*(gem['macro_f1']-local_mf1_max):.1f} points "
            f"above the best of them; with an accuracy close to that of Qwen3-Embedding-4B, its advantage therefore lies mainly in the smaller units "
            f"(Section 5.3)."))
        B.append(P(
            f"Grouped cross-validation accuracies exceed the chronological test accuracies by {gap.get('min', 0):.1f}–{gap.get('max', 0):.1f} points "
            f"(median {gap.get('median', 0):.1f}) for the local systems, which quantifies the temporal drift that a random split would hide; the ranking "
            f"of the systems is nevertheless very similar under both estimates (Spearman ρ = {sp_tc.get('rho', 0):.2f} over {sp_tc.get('n', 0)} systems). "
            f"Validation and test macro-F1 agree less closely (ρ = {sp_vt.get('rho', 0):.2f} over {sp_vt.get('n', 0)} systems). The clearest case is "
            f"{big['system'].replace(' / RAG-10', ' with retrieval')}, whose validation macro-F1 ({f3(big['val_macro_f1'])}) was almost that of Gemini "
            f"({f3(gem['val_macro_f1'])}) but which fell to {f3(big['macro_f1'])} on the test half-year, the largest drop of any system. The frozen "
            f"encoder selected on validation, {rep_emb['system']} ({f3(rep_emb['val_macro_f1'])} on validation, {f3(rep_emb['macro_f1'])} on test), "
            f"kept its macro-F1 but is {100*(qwen['accuracy']-rep_emb['accuracy']):.1f} accuracy points below Qwen3-Embedding-4B; it is reported as "
            f"the representative of its family nonetheless, because replacing it by a configuration that looks better on the test partition would "
            f"reintroduce the selection bias the protocol avoids."))
    B.append(fig_block("fig_main", "⟦F:main⟧. Test-partition accuracy (a) and macro-F1 (b) of the selected systems with 95% bootstrap intervals; vertical ticks mark the grouped cross-validation mean."))
    B.append(H(2, "5.2. Preprocessing, input fields and translation"))
    abl = R["ablation_lexical"]
    if abl is not None and len(abl):
        rows = [[r.system, f"{100*r.accuracy:.1f}", f"{100*r.macro_f1:.1f}", f"{100*r.weighted_f1:.1f}",
                 f"{100*r.cv_accuracy_mean:.1f} ± {100*r.cv_accuracy_sd:.1f}" if not pd.isna(r.cv_accuracy_mean) else "–"] for r in abl.itertuples()]
        B.append({"type": "table", "caption": "⟦T:ablation⟧. Lexical ablations on the test partition (selected classifier per representation; values in %). "
                  "Content-only rows isolate the effect of machine translation on identical documents; '+ sender unit' adds the dispatching unit as a feature.",
                  "columns": ["Representation", "Accuracy", "Macro-F1", "wF1", "CV accuracy"], "col_widths": [0.44, 0.14, 0.14, 0.14, 0.14], "font_size": 16, "rows": rows})
    def A(rep):
        r = abl[abl["representation"] == rep] if abl is not None else None
        return r.iloc[0] if r is not None and len(r) else None
    a_raw, a_stem, a_lem, a_chr, a_sub, a_ctr, a_cen = (A(k) for k in ["tfidf_word_raw", "tfidf_word_stem", "tfidf_word_lemma", "tfidf_char_raw", "tfidf_subject_raw", "tfidf_content_tr", "tfidf_content_en"])
    def acc_mf(r):
        return f"{pct(r['accuracy'])}% / {pct(r['macro_f1'])}" if r is not None else PENDING
    def cvs(r):
        return f"{pct(r['cv_accuracy_mean'])} ± {pct(r['cv_accuracy_sd'])}" if r is not None and not pd.isna(r['cv_accuracy_mean']) else PENDING
    def E(contains):
        """Row of the (lexical and encoder) ablation table whose system name contains the string."""
        r = abl_all[abl_all["system"].str.contains(contains, regex=False)] if abl_all is not None else None
        return r.iloc[0] if r is not None and len(r) else None
    allc0 = R["all_configurations"]
    raw_svm = allc0[allc0["system"] == "TF-IDF word (raw) / LinearSVM"] if allc0 is not None else []
    raw_svm_acc = pct(raw_svm.iloc[0]["accuracy"]) if len(raw_svm) else PENDING
    e_tr, e_en = E("paraphrase-multilingual-mpnet-base [content only"), E("all-mpnet-base-v2")
    dc = deid.get("content", {}); de = deid.get("content_en", {})
    B.append(P(
        f"⟦T:ablation⟧ and ⟦F:ablation⟧ report the lexical ablations (accuracy / macro-F1 on the test partition). Stemming and lemmatisation reduce the "
        f"vocabulary by {100*(1-R['prep']['stem']/R['prep']['raw']):.0f}% and {100*(1-R['prep']['lemma']/R['prep']['raw']):.0f}%. On the chronological "
        f"test, lemmatisation gave the highest lexical accuracy and macro-F1 ({acc_mf(a_lem)}), closely followed by stemming ({acc_mf(a_stem)}), "
        f"against {acc_mf(a_raw)} for the raw tokens (validation selected logistic regression for this variant; its linear-SVM configuration reached "
        f"{raw_svm_acc}%, close to the stemmed one). Under grouped cross-validation, stemming and lemmatisation are indistinguishable ({cvs(a_lem)} "
        f"and {cvs(a_stem)}) and about two points above the raw tokens ({cvs(a_raw)}), so morphological normalisation is worth about two points on "
        f"new correspondence, whereas the choice between the rule-based stemmer and the morphological analyser does not matter. Character n-grams, "
        f"which need no linguistic resource, reach {acc_mf(a_chr)} and are thus a practical alternative to a morphological analyser. The subject line "
        f"alone yields {acc_mf(a_sub)}, because many subjects are form names (*Vekalet*, *Görevlendirme*, *Araç Tahsisi*), and the body alone "
        f"{acc_mf(a_ctr)}; both together are needed for the best result, as the body carries the terms that distinguish sub-units of the same "
        f"department. The translation ablation on identical bodies is clear for the lexical model: the Turkish body processed natively reaches "
        f"{acc_mf(a_ctr)} (CV {cvs(a_ctr)}), the machine-translated English body {acc_mf(a_cen)} (CV {cvs(a_cen)}; McNemar "
        f"{mcp('TF-IDF body (TR)', 'TF-IDF body (EN, translated)')}). For sentence encoders the result is mixed and not significant: the English "
        f"all-mpnet-base-v2 on the translated body was {100*(e_en['accuracy']-e_tr['accuracy']):.1f} points more accurate than the multilingual "
        f"mpnet encoder on the Turkish body ({pct(e_en['accuracy'])}% against {pct(e_tr['accuracy'])}%; McNemar "
        f"{mcp('mpnet body (TR, multilingual)', 'mpnet body (EN, translated)')}) but {100*(e_tr['macro_f1']-e_en['macro_f1']):.1f} points lower "
        f"in macro-F1 ({pct(e_en['macro_f1'])} against {pct(e_tr['macro_f1'])}; paired bootstrap, Turkish minus "
        f"English: {bd('mpnet body (TR, multilingual)', 'mpnet body (EN, translated)')}), and this comparison changes the encoder together with the language. "
        f"The English recogniser also inserted more placeholders than the Turkish one ({de.get('[PERSON]', 0):,} against {dc.get('[KİŞİ]', 0):,} "
        f"names in the bodies), so the translations lose somewhat more words to de-identification, which may account for part of the difference. "
        f"Translation therefore does not help: it lowers the accuracy of the lexical model, presumably because institutional terms, unit names "
        f"and form titles are translated inconsistently, it does not improve an encoder consistently, and it adds cost and an external dependency "
        f"to the pipeline."))
    sr = A("tfidf_word_lemma+sender"); base = A("tfidf_word_lemma")
    if sr is not None:
        e5s, e5n = E("multilingual-e5-base + sender"), e5b
        mps, mpn = E("paraphrase-multilingual-mpnet-base + sender"), row_of(main, "paraphrase-multilingual-mpnet-base /")
        B.append(P(
            f"*Sender-unit context.* In the dispatch workflow the unit that forwards a document is always known, and it narrows the candidate recipients "
            f"(a document in the SKS office's inbox will go to an SKS sub-unit or back to the Rectorate). Appending a one-hot encoding of the sender unit to "
            f"the lemmatised lexical features raises the test accuracy from {pct(base['accuracy'])}% to {pct(sr['accuracy'])}% and the macro-F1 from "
            f"{f3(base['macro_f1'])} to {f3(sr['macro_f1'])} (grouped CV {cvs(base)} → {cvs(sr)}; top-3 accuracy {pct(sr['top3_accuracy'])}%), more "
            f"than any change of representation or model in this study; the same feature raises multilingual-E5 base from {pct(e5n['accuracy'])}% to "
            f"{pct(e5s['accuracy'])}% and multilingual mpnet from {pct(mpn['accuracy'])}% to {pct(mps['accuracy'])}%. This variant is reported outside "
            f"the main comparison because it uses information that is unavailable when a document is routed at its point of entry; where the assistant "
            f"runs inside a department's inbox, however, it is the configuration to deploy, and it shows that most of the remaining errors of the "
            f"text-only systems are between units of the same department."))
    B.append(fig_block("fig_ablation", "⟦F:ablation⟧. Ablations of preprocessing, input field, translation and sender context (test partition; accuracy and macro-F1 in %)."))
    B.append(H(2, "5.3. Per-unit behaviour and confusion structure"))
    es = R["error_summary"]; cp = R["confused_pairs"]
    pc0 = R["per_class"]; gain_sentence = ""
    if pc0 is not None and len(pc0):                         # where the final system gains over the best local representative
        fin = pc0[pc0["system"] == final["system"]].set_index("unit")
        loc_best = pc0[pc0["system"] != final["system"]].groupby("unit")["f1"].max()
        g = (fin["f1"] - loc_best).sort_values(ascending=False)
        ok = [u for u in g.index if fin.loc[u, "support"] >= 10]
        top, low = ok[:3], ok[::-1][:3]
        def lst(us):
            it = [f"{SHORT[u]} ({f3(fin.loc[u, 'f1'])} against {f3(loc_best[u])})" for u in us]
            return ", ".join(it[:-1]) + " and " + it[-1] if len(it) > 1 else "".join(it)
        gain_sentence = (" Compared with the best of the three local representatives on each unit, the final system gains most in F1 on "
                         + lst(top) + ", and loses most on " + lst(low) + ".")
    if es is not None and len(es):
        e = es.iloc[0]
        B.append(P(
            f"⟦F:confusion⟧ shows the confusion matrix of the final system, {e['system']}, and ⟦F:perclass⟧ the per-unit F1 of the family representatives ordered by training support. Of the "
            f"{int(e['n_errors'])} errors on the {int(e['n_test']):,} test documents, {int(e['errors_within_department'])} "
            f"({pct(e['errors_within_department']/e['n_errors'])}%) stay inside the correct department and {int(e['errors_across_departments'])} cross "
            f"departments; department-level accuracy is {pct(e['department_accuracy'])}%. The most frequent confusions are "
            + "; ".join(f"{r['true -> predicted'].replace(' -> ', ' → ')} ({r['count']})" for r in cp.head(5).to_dict('records')) + ". "
            "Two structural sources of error stand out. First, the department offices (BİDB office, SKS office) receive documents that are forwarded to a "
            "sub-unit afterwards, so office and sub-unit labels describe the same kind of document at different points of the chain; the lenient accuracy of "
            "⟦T:main⟧, which accepts any unit that acted on the document, is consistently higher than the strict accuracy for this reason. Second, documents "
            "addressed to the Rectorate are mostly written *by* the Software Unit (reports and replies), so their text resembles Software Unit documents; this "
            "confusion is inherent to labelling by recipient and would require the direction of the correspondence as an additional feature."
            + gain_sentence))
    B.append(fig_block("fig_confusion", f"⟦F:confusion⟧. Confusion matrix of the final system (retrieval-augmented {final['system'].split(' / ')[0]}, selected on validation) on the test partition (rows: true unit with test support; colour: row-normalised share; orange frames and brackets: departments).", 5.6))
    pc = R["per_class"]
    if pc is not None and len(pc):
        first = pc[pc["system"] == pc["system"].iloc[0]].sort_values("train_support", ascending=False)
        rows = [[SHORT[r.unit], f"{int(r.train_support):,}", f"{int(r.support)}", f"{100*r.precision:.1f}", f"{100*r.recall:.1f}", f"{100*r.f1:.1f}"] for r in first.itertuples()]
        B.append({"type": "table", "caption": f"⟦T:perclass⟧. Per-unit precision, recall and F1 of {pc['system'].iloc[0]} on the test partition (values in %; support: training / test documents).",
                  "columns": ["Unit", "Train n", "Test n", "Precision", "Recall", "F1"], "col_widths": [0.34, 0.12, 0.12, 0.14, 0.14, 0.14], "font_size": 16, "rows": rows})
    B.append(fig_block("fig_perclass", "⟦F:perclass⟧. Per-unit F1 of the family representatives (selected on validation; fine-tuned BERTurk: seed 0) on the test partition, ordered by training support."))
    B.append(H(2, "5.4. Statistical comparison"))
    mc = R["mcnemar"]
    if mc is not None and len(mc):
        def fp(p):
            return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
        s0 = lambda n: n.replace("mean of 3 seeds", "seed 0")
        rows = [[s0(r.system_a), s0(r.system_b), f"{100*r.acc_a:.1f} / {100*r.acc_b:.1f}", f"{int(r.n_a_only)} / {int(r.n_b_only)}", fp(r.p_value), fp(r.p_holm)] for r in mc.itertuples()]
        B.append({"type": "table", "caption": "⟦T:mcnemar⟧. Exact McNemar tests between the family representatives (selected on validation) on the test partition (the fine-tuned BERTurk with its seed-0 predictions). Discordant: documents correct only for A / only for B; p-values without and with Holm correction.",
                  "columns": ["System A", "System B", "Acc. A / B (%)", "Discordant A / B", "p", "p (Holm)"], "col_widths": [0.26, 0.26, 0.14, 0.14, 0.1, 0.1], "font_size": 14, "rows": rows})
        if mce is not None and len(mce):
            ar = lambda n: n.replace(" -> ", " → ")
            rows = [[ar(r.system_a), ar(r.system_b), f"{100*r.acc_a:.1f} / {100*r.acc_b:.1f}", f"{int(r.n_a_only)} / {int(r.n_b_only)}", fp(r.p_value), fp(r.p_holm)] for r in mce.itertuples()]
            B.append({"type": "table", "caption": "⟦T:mcnemar_extra⟧. Further exact McNemar tests quoted in the text (test partition; Holm correction within this table). "
                      "BERTurk fine-tuned: seed 0; kNN-10 vote: majority unit of the ten retrieved documents shown in the RAG prompt; cascades as in ⟦T:cascade⟧; "
                      "body-only TF-IDF and mpnet encoders for the translation tests.",
                      "columns": ["System A", "System B", "Acc. A / B (%)", "Discordant A / B", "p", "p (Holm)"], "col_widths": [0.26, 0.26, 0.14, 0.14, 0.1, 0.1], "font_size": 14, "rows": rows})
        sig = mc[mc["p_holm"] < 0.05]
        nsig = mc[mc["p_holm"] >= 0.05]
        s_keys = sorted(seeds)
        sv = lambda k, n: pfmt(seeds[k][f"p_vs_{n}"]) if k in seeds else PENDING
        sacc = lambda k: pct(seeds[k]["accuracy"]) if k in seeds else PENDING
        bs12 = ex.get("macro_f1_difference_bootstrap_without_smallest_unit", {})
        def bd12(a, b):
            r = bs12.get(f"{a} | {b}")
            if r is None:
                return PENDING
            sg = lambda v: f"{100*v:+.1f}".replace("-", "−")
            return f"{sg(r['diff'])} points, 95% CI [{sg(r['ci'][0])}, {sg(r['ci'][1])}]"
        hs_f1 = pc0.loc[(pc0["system"] == final["system"]) & (pc0["unit"] == bs12.get("excluded_unit")), "f1"] if pc0 is not None else []
        B.append(P(
            f"⟦T:mcnemar⟧ lists the tests between the family representatives and ⟦T:mcnemar_extra⟧ the further comparisons quoted in the text. "
            f"{'All ' + small(len(mc)) if len(sig) == len(mc) else small(len(sig)).capitalize() + ' of the ' + small(len(mc))} representative comparisons are significant "
            f"after Holm correction: the retrieval-augmented Gemini is more accurate than each local representative, and the fine-tuned BERTurk and "
            f"the TF-IDF model are both more accurate than the multilingual-E5 representative"
            + (f", whereas BERTurk and TF-IDF do not differ ({mcp_rep('BERTurk', 'TF-IDF')})" if len(nsig) else "") + ". "
            f"Among the LLMs, Gemini was also more accurate than GPT-6 Luna ({mcp('Gemini RAG', 'GPT RAG')}) and Claude Haiku "
            f"({mcp('Gemini RAG', 'Haiku RAG')}). Three qualifications apply. First, in accuracy Gemini did not differ from the best frozen encoder, "
            f"Qwen3-Embedding-4B ({mcp('Gemini RAG', 'Qwen3-Embedding-4B / RBF-SVM')}), which is not a family representative because multilingual-E5 "
            f"large had the higher validation macro-F1; the advantage of Gemini lies in macro-F1, where a paired bootstrap over the test documents "
            f"gives {bd('Gemini RAG', 'Qwen3-Embedding-4B / RBF-SVM')} against Qwen3-Embedding-4B, {bd('Gemini RAG', 'BERTurk fine-tuned')} against "
            f"BERTurk and {bd('Gemini RAG', LEXN)} against TF-IDF, while BERTurk and TF-IDF differ by {bd('BERTurk fine-tuned', LEXN)}. Second, part of "
            f"this macro-F1 advantage comes from the Health Services Unit, whose single test document Gemini routed correctly (F1 "
            f"{f3(float(hs_f1.iloc[0])) if len(hs_f1) else PENDING}) and the local models did not; without this unit, the advantage is "
            f"{bd12('Gemini RAG', 'Qwen3-Embedding-4B / RBF-SVM')} against Qwen3-Embedding-4B and {bd12('Gemini RAG', 'BERTurk fine-tuned')} against "
            f"BERTurk, smaller but still significant. Third, the comparisons with BERTurk depend on the seed: with seed 1 "
            f"({sacc(s_keys[1]) if len(s_keys) > 1 else PENDING}% accuracy) the conclusions are those of seed 0, but with seed 2 "
            f"({sacc(s_keys[2]) if len(s_keys) > 2 else PENDING}%) BERTurk was more accurate than TF-IDF "
            f"({sv(s_keys[2], LEXN) if len(s_keys) > 2 else PENDING}) and no longer differed significantly from Qwen3-Embedding-4B "
            f"({sv(s_keys[2], 'Qwen3-Embedding-4B / RBF-SVM') if len(s_keys) > 2 else PENDING}) or Gemini "
            f"({sv(s_keys[2], 'Gemini RAG') if len(s_keys) > 2 else PENDING}, both before correction). Differences of one to two points between "
            f"the best local models are therefore not robust on a test partition of {sp['test']:,} documents. These tests, together with the "
            "overlapping bootstrap intervals in ⟦F:main⟧, are the basis on which the conclusions of Section 6 are restricted to differences that the "
            "data actually support."))
    B.append(H(2, "5.5. Top-k suggestion and deferral to staff"))
    if defer is not None and len(defer):
        rows = [[r.system.replace("mean of 3 seeds", "seed 0"), f"{r.AURC:.3f}", f"{100*r.accuracy_full:.1f}", f"{100*r['coverage@90']:.1f}", f"{100*r['accuracy@90']:.1f}", f"{100*r['coverage@95']:.1f}", f"{100*r['accuracy@95']:.1f}"]
                for _, r in defer.iterrows()]
        B.append({"type": "table", "caption": "⟦T:deferral⟧. Selective routing on the test partition. AURC: area under the risk-coverage curve (lower is better). "
                  "Coverage@90/95: share of test documents routed automatically with the threshold chosen on validation for a 90% / 95% target accuracy; "
                  "Accuracy@90/95: accuracy actually obtained on that share.",
                  "columns": ["System", "AURC", "Acc. (all)", "Cov.@90", "Acc.@90", "Cov.@95", "Acc.@95"], "col_widths": [0.34, 0.1, 0.11, 0.11, 0.11, 0.11, 0.12], "font_size": 15, "rows": rows})
        a_ft, a_lx = f"{d_best['AURC']:.3f}", f"{d_lex['AURC']:.3f}"
        aurc_txt = f"both {a_ft}" if a_ft == a_lx else f"{a_ft} and {a_lx}"
        B.append(P(
            f"⟦T:deferral⟧ and ⟦F:deferral⟧a show the local family representatives (LLM answers carry no comparable confidence score). Confidence is "
            f"informative for the fine-tuned and the lexical model: errors concentrate in the low-confidence tail, and the two reach practically the "
            f"same AURC ({aurc_txt}). The RBF SVM on multilingual-E5 large ranks the documents much worse "
            f"({d_emb['AURC']:.3f}): its one-versus-rest scores are derived from pairwise votes, and for most documents the top unit leads the "
            f"second by about one vote, so the scores separate correct from incorrect decisions poorly, and the threshold chosen on validation for a "
            f"90% target routed only {pct(d_emb['coverage@90'])}% of the test documents, at {pct(d_emb['accuracy@90'])}%. With the threshold chosen on "
            f"validation for a 95% target, the fine-tuned BERTurk routed "
            f"{pct(d_best['coverage@95'])}% of the test documents automatically at {pct(d_best['accuracy@95'])}% accuracy, and {pct(d_best['coverage@90'])}% "
            f"at {pct(d_best['accuracy@90'])}% for a 90% target; the TF-IDF model routed {pct(d_lex['coverage@95'])}% at {pct(d_lex['accuracy@95'])}% and "
            f"{pct(d_lex['coverage@90'])}% at {pct(d_lex['accuracy@90'])}%. The thresholds of the lexical model thus held within one point on the next half-year, whereas "
            f"those of the fine-tuned model fell about three points short of their targets: confidence still ranks the new documents well, but the absolute "
            f"level of a fine-tuned model's confidence shifted across the half-year boundary, so a deployment should recalibrate thresholds on recent "
            f"documents or apply a safety margin. The documents below the threshold would be shown to staff with the top-3 list, which contains the correct "
            f"unit for {pct(ft['top3_accuracy'])}% of all documents with BERTurk, or passed to the LLM second stage."))
    B.append(fig_block("fig_deferral", "⟦F:deferral⟧. (a) Risk-coverage curves of the local family representatives on the test partition (accuracy on the automatically routed share as a function of coverage); (b) test accuracy of the cascade from the fine-tuned BERTurk to each retrieval-augmented LLM as a function of the share of documents sent to the LLM (the shares chosen on validation are given in ⟦T:cascade⟧)."))
    B.append(H(2, "5.6. Calibration and hierarchical back-off"))
    if cal is not None and hie is not None:
        def cc(c, a):
            return f"{100*c:.1f} / {100*a:.1f}" if not pd.isna(a) else f"{100*c:.1f} / –"
        rows = []
        for _, r in cal.iterrows():
            nm = "BERTurk fine-tuned (seed 0)" if r["system"].startswith("BERTurk") else r["system"]
            for i, t in enumerate((90, 95)):
                rows.append([nm if i == 0 else "", f"{r['temperature']:.2f}" if i == 0 else "",
                             f"{r['ece_test_before']:.3f} → {r['ece_test_after']:.3f}" if i == 0 else "", f"{t}%",
                             cc(r[f"emp_cov@{t}"], r[f"emp_acc@{t}"]), cc(r[f"cal_cov@{t}"], r[f"cal_acc@{t}"]), cc(r[f"rec_cov@{t}"], r[f"rec_acc@{t}"])])
        B.append({"type": "table", "caption": "⟦T:calib⟧. Calibration and deferral rules on the test partition, all applied to the temperature-scaled confidences (the "
                  "empirical rule on the raw confidences is reported in ⟦T:deferral⟧). T: temperature fitted on validation; ECE: expected "
                  "calibration error before → after temperature scaling; rule cells: coverage / accuracy (%) of the automatically routed documents. Empirical: "
                  "largest validation coverage meeting the target; calibrated: calibrated confidence ≥ target; recent quarter: empirical threshold chosen on "
                  "April–June 2022 only.",
                  "columns": ["System", "T", "ECE (test)", "Target", "Empirical", "Calibrated", "Recent quarter"],
                  "col_widths": [0.25, 0.07, 0.15, 0.08, 0.15, 0.15, 0.15], "font_size": 15, "rows": rows})
        def reach(c, t):
            a = 100 * c[f"rec_acc@{t}"]
            return f"to the {t}% target" if a >= t else f"to within {t - a:.1f} points of the {t}% target"
        B.append(P(
            f"⟦T:calib⟧ and ⟦F:routing⟧a summarise calibration. The fine-tuned BERTurk was over-confident (T = {cb_['temperature']:.2f}): temperature "
            f"scaling roughly halved its calibration error, from {cb_['ece_test_before']:.3f} to {cb_['ece_test_after']:.3f} on the test half-year "
            f"(validation {cb_['ece_val_before']:.3f} → {cb_['ece_val_after']:.3f}), and the negative log-likelihood fell from {cb_['nll_test_before']:.2f} "
            f"to {cb_['nll_test_after']:.2f}. The softmax-normalised margins of the TF-IDF model were far too flat (T = {ct_['temperature']:.2f}) and "
            f"became usable probabilities after scaling (ECE {ct_['ece_test_before']:.3f} → {ct_['ece_test_after']:.3f}). Calibration made the "
            f"confidence interpretable as a probability but did not remove the effect of drift. Because it acts on all logits, temperature scaling "
            f"also reorders the documents' top probabilities slightly (Spearman ρ ≥ {cal['spearman_raw_vs_scaled_test'].min():.3f} between raw and "
            f"scaled confidence on test), and the largest coverage that meets a target is sensitive to such small changes: for the 95% target the "
            f"empirical rule covered {pct(cb_['emp_val_cov@95'])}% of the BERTurk validation documents instead of {pct(cb_['raw_val_cov@95'])}% and "
            f"routed {pct(cb_['emp_cov@95'])}% of the test documents at {pct(cb_['emp_acc@95'])}%, against {pct(d_best['coverage@95'])}% at "
            f"{pct(d_best['accuracy@95'])}% with the raw scores (⟦T:deferral⟧). The calibrated rule held the 90% target for BERTurk "
            f"({pct(cb_['cal_cov@90'])}% of the documents routed at {pct(cb_['cal_acc@90'])}%) but not the 95% target ({pct(cb_['cal_cov@95'])}% at "
            f"{pct(cb_['cal_acc@95'])}%), so the highest confidences remained optimistic on the new half-year. Choosing the threshold on the most recent "
            f"validation quarter brought the lexical model {reach(ct_, 90)} and {reach(ct_, 95)} ({pct(ct_['rec_cov@90'])}% of the documents at "
            f"{pct(ct_['rec_acc@90'])}% and {pct(ct_['rec_cov@95'])}% at {pct(ct_['rec_acc@95'])}%) at the price of coverage, whereas for BERTurk the "
            f"{rs['BERTurk fine-tuned']['n_val_recent']:,} documents of that quarter were too few to support a 95% threshold. Thresholds should "
            f"therefore be re-estimated regularly on recent documents, and with a margin."))
        rows = []
        for r in hie.itertuples():
            nm = "BERTurk fine-tuned (seed 0)" if r.system.startswith("BERTurk") else r.system
            rows.append([nm, f"{100*r.target:.0f}%", cc(r.test_unit_share, r.test_unit_acc), cc(r.test_dept_share, r.test_dept_acc),
                         f"{100*r.test_staff_share:.1f}", f"{100*r.test_dept_needs_forwarding:.1f}"])
        B.append({"type": "table", "caption": "⟦T:hier⟧. Hierarchical back-off on the test partition (calibrated probabilities; thresholds chosen on validation "
                  "for the target accuracy at each level). Unit: documents routed to the predicted unit (share / accuracy, %); department office: documents "
                  "routed to the office of the predicted department (share / department-level accuracy, %); staff: documents left for manual dispatch (%); "
                  "forwarding: share of the back-off documents (%) whose department was right and whose target was one of its units.",
                  "columns": ["System", "Target", "Unit", "Department office", "Staff", "Forwarding"],
                  "col_widths": [0.3, 0.1, 0.16, 0.2, 0.1, 0.14], "font_size": 15, "rows": rows})
        short = max(100 * (r.target - min(r.test_unit_acc, r.test_dept_acc)) for r in hie.itertuples())
        tdb = rs.get("BERTurk fine-tuned", {}); tdt = rs.get(ct_["system"], {})
        B.append(P(
            f"The hierarchical back-off changes how much of the routing can be automated (⟦T:hier⟧, ⟦F:routing⟧b). With the thresholds chosen for "
            f"a 95% target, BERTurk routed {pct(hb95['test_unit_share'])}% of the test documents to their unit ({pct(hb95['test_unit_acc'])}% correct) "
            f"and a further {pct(hb95['test_dept_share'])}% to the office of the predicted department, which was the right department for "
            f"{pct(hb95['test_dept_acc'])}% of them; {pct(hb95['test_staff_share'])}% of the documents were left to staff, against "
            f"{pct(1 - hb95['test_unit_share'])}% with unit-level deferral alone. For the 90% target only {pct(hb90['test_staff_share'])}% remained "
            f"for staff. About four fifths of the back-off documents ({pct(hb95['test_dept_needs_forwarding'], 0)}%) then need one forwarding step "
            f"inside the department, the step the offices perform today, and {pct(hb95['test_dept_office_is_target'], 0)}% had the office itself as "
            f"target. The lexical model behaves similarly ({pct(ht95['test_staff_share'])}% left to staff at the 95% target). As with the flat rules, "
            f"the accuracies fall short of the validation targets on the test half-year, by up to {short:.1f} points. Decoding the department first "
            f"and the unit inside it did not change the unit-level accuracy (BERTurk {pct(tdb.get('flat_accuracy'))}% flat and "
            f"{pct(tdb.get('top_down_accuracy'))}% top-down; TF-IDF {pct(tdt.get('flat_accuracy'))}% and {pct(tdt.get('top_down_accuracy'))}%), so "
            f"the gain comes from backing off, not from a different prediction."))
    B.append(fig_block("fig_routing", "⟦F:routing⟧. (a) Reliability curves of the fine-tuned BERTurk on the test partition before and after temperature "
                       "scaling (bins with at least ten documents; dashed line: perfect calibration); (b) share of test documents routed to the unit, backed "
                       "off to the department office, or left to staff, with thresholds chosen on validation for 90% and 95% target accuracy."))
    B.append(H(2, "5.7. LLM second stage and cascade"))
    if llm_rows is not None and len(llm_rows):
        names = {"gemini-3.8-flash": "Gemini 3.8 Flash", "gpt-6-luna": "GPT-6 Luna", "claude-haiku-5-5": "Claude Haiku 5.5"}
        rows = []
        for mode, mlab in [("zeroshot", "zero-shot"), ("rag10", "RAG-10")]:
            for m, lab in names.items():
                r = row_of(main, f"{lab} / {mlab}")
                if r is None:
                    continue
                rows.append([f"{lab}, {mlab}", f"{pct(r['accuracy'])} {ci(r['accuracy_ci'])}", pct(r["macro_f1"]), pct(r["top3_accuracy"]),
                             f"{r['predict_ms_per_doc']:.0f}", usd(m, mode)])
        rows.append(["Vote of the 10 retrieved documents (no LLM)", f"{pct(knn.get('test_accuracy'))}", pct(knn.get("test_macro_f1")), "–", "–", "0"])
        prices = "; ".join(f"{names[m].replace('Claude ', '').replace(' 5.5', '')} {model_info(m)[1]:.2f} / {model_info(m)[2]:.2f}" for m in names)
        B.append({"type": "table", "caption": "⟦T:llm⟧. Large language models on the test partition (de-identified text; accuracy with 95% bootstrap interval, values in %). "
                  "RAG-10: prompt with the ten most similar past documents and their units. Latency: median API time per document; cost at list prices "
                  f"(USD per million input / output tokens: {prices}).",
                  "columns": ["Model, mode", "Accuracy", "Macro-F1", "Top-3", "Latency (ms)", "USD / 1,000 docs"], "col_widths": [0.32, 0.2, 0.12, 0.1, 0.12, 0.14], "font_size": 15, "rows": rows})
        gains = [a["accuracy"] - b["accuracy"] for a, b in zip(rg, zs)]
        B.append(P(
            f"⟦T:llm⟧ reports the LLM results. In zero-shot mode, where the models know the units only through their descriptions, accuracy was "
            f"{pct(zs_lo)}–{pct(zs_hi)}% and macro-F1 {pct(min(r['macro_f1'] for r in zs))}–{pct(max(r['macro_f1'] for r in zs))}%, below every "
            f"supervised system of ⟦T:main⟧: the boundaries between the sub-units are institutional conventions that cannot be inferred from duty "
            f"descriptions alone. Retrieved examples raised the accuracy of every model by {100*min(gains):.0f}–{100*max(gains):.0f} points (Gemini from "
            f"{pct(gem0['accuracy'])}% to {pct(gem['accuracy'])}%, GPT-6 Luna from {pct(gpt0['accuracy'])}% to {pct(gpt['accuracy'])}%, Claude Haiku "
            f"from {pct(hai0['accuracy'])}% to {pct(hai['accuracy'])}%). The ten retrieved documents already form a nearest-neighbour classifier: the "
            f"majority unit among them reaches {pct(knn.get('test_accuracy'))}% accuracy and {pct(knn.get('test_macro_f1'))}% macro-F1, so the LLMs "
            f"add between {100*(hai['accuracy']-knn.get('test_accuracy', 0)):.0f} (Haiku) and {100*(gem['accuracy']-knn.get('test_accuracy', 0)):.0f} "
            f"points (Gemini; {mcp('Gemini RAG', 'kNN-10 vote')}) by weighing the examples against the document and the unit descriptions. With "
            f"retrieval, the cost per 1,000 documents ranged from USD {usd('gpt-6-luna')} (GPT-6 Luna) and USD {usd('claude-haiku-5-5')} (Haiku) to "
            f"USD {usd('gemini-3.8-flash')} (Gemini), so the most accurate model was also the most expensive by a factor of about "
            f"{small(round(float(usd('gemini-3.8-flash')) / float(usd('gpt-6-luna'))))}, for {100*(gem['accuracy']-gpt['accuracy']):.1f} accuracy points and "
            f"{100*(gem['macro_f1']-gpt['macro_f1']):.1f} macro-F1 points over GPT-6 Luna."))
    if casc is not None and len(casc):
        cb = casc[casc["local_system"].str.contains("BERTurk fine-tuned", regex=False)]
        rows = [[LLM_NAMES.get(r.llm.replace("__rag10", ""), r.llm), "within 1 pt of LLM" if r.rule.startswith("min") else "best val. accuracy",
                 f"{100*r.llm_share:.0f}", f"{100*r.local_alone:.1f}", f"{100*r.llm_alone:.1f}", f"{100*r.test_accuracy:.1f}"] for r in cb.itertuples()]
        B.append({"type": "table", "caption": "⟦T:cascade⟧. Cascade from the fine-tuned BERTurk (seed 0) to a retrieval-augmented LLM on the test partition. The share of "
                  "documents sent to the LLM (those with the lowest BERTurk confidence) is chosen on validation by the rule in the second column; accuracies in %.",
                  "columns": ["LLM (RAG-10)", "Rule for the share", "LLM share (%)", "BERTurk alone", "LLM alone", "Cascade"], "col_widths": [0.24, 0.24, 0.13, 0.13, 0.13, 0.13], "font_size": 15, "rows": rows})
        def tg_(rule):
            t = casc[casc["local_system"].str.contains("TF-IDF", regex=False) & casc["llm"].str.contains("gemini") & (casc["rule"] == rule)]
            return t.iloc[0] if len(t) else None
        tg, tg_min = tg_("best val accuracy"), tg_("min share within 1pt of LLM")
        mins = cb[cb["rule"].str.startswith("min")]["llm_share"]
        v_loc = cg.get("val_accuracy_local_alone", np.nan)
        v_llm = [c.get("val_accuracy_llm_alone", np.nan) for c in (cg, cgpt, chai)]
        B.append(P(
            f"⟦T:cascade⟧ and ⟦F:deferral⟧b evaluate the cascade with the fine-tuned BERTurk as first stage. With the share chosen on validation, passing "
            f"the {100*cg['llm_share']:.0f}% least confident documents to Gemini 3.8 Flash gave {pct(cg['test_accuracy'])}% accuracy and "
            f"{pct(cg['test_macro_f1'])}% macro-F1, more than BERTurk alone ({mcp('Cascade BERTurk -> Gemini', 'BERTurk fine-tuned')}) and as much as "
            f"the LLM alone ({mcp('Cascade BERTurk -> Gemini', 'Gemini RAG')}), while the LLM cost fell from USD {cg['llm_usd_per_1000_docs_alone']:.2f} "
            f"to {cg['llm_usd_per_1000_docs_cascade']:.2f} per 1,000 documents, and {100*(1-cg['llm_share']):.0f}% of the documents would not have to "
            f"be sent to an external service. With GPT-6 Luna and Claude Haiku, the cascade reached {pct(cgpt['test_accuracy'])}% (share "
            f"{100*cgpt['llm_share']:.0f}%) and {pct(chai['test_accuracy'])}% (share {100*chai['llm_share']:.0f}%), more than either LLM alone "
            f"({pct(gpt['accuracy'])}% and {pct(hai['accuracy'])}%). The complementarity is visible in ⟦F:deferral⟧b: accuracy rises while the first "
            f"two fifths of the documents are passed on and then declines towards the LLM's own accuracy, because the LLM is more accurate than BERTurk "
            f"on the low-confidence documents and slightly less accurate on the confident ones. The rule 'smallest share within one point of the LLM' "
            f"selects few or no LLM calls here ({100*mins.min():.0f}–{100*mins.max():.0f}%), because BERTurk alone was already close to the LLMs on the "
            f"validation half-year ({pct(v_loc)}% against {pct(min(v_llm))}–{pct(max(v_llm))}%); it is informative for weaker first stages. A CPU-only "
            f"first stage is also viable: with the TF-IDF model, sending {100*tg_min['llm_share'] if tg_min is not None else 0:.0f}% of the documents "
            f"to Gemini (the smallest share within one point of the LLM on validation) gave {pct(tg_min['test_accuracy']) if tg_min is not None else PENDING}% "
            f"and sending {100*tg['llm_share'] if tg is not None else 0:.0f}% (the best validation accuracy) "
            f"{pct(tg['test_accuracy']) if tg is not None else PENDING}%."))
    B.append(H(2, "5.8. Computational cost"))
    cost = R["cost"]
    if cost is not None and len(cost):
        keep = set(main.loc[main["group"] != "LLM", "system"]) if main is not None else set()
        cst = cost[cost["system"].isin(keep)].copy()
        cst["order"] = cst["group"].map({"Lexical (TF-IDF)": 0, "Frozen encoder": 1, "Fine-tuned encoder": 2}).fillna(3)
        cst = cst.sort_values(["order", "fit_seconds_trainval"])
        rows = [[r.system, f"{r.fit_seconds_trainval:.1f}" if r.fit_seconds_trainval < 10 else f"{r.fit_seconds_trainval:.0f}", f"{r.predict_ms_per_doc:.2f}",
                 f"{int(r.n_features):,}" if r.n_features else "–"] for r in cst.itertuples()]
        B.append({"type": "table", "caption": "⟦T:cost⟧. Training time on training plus validation documents and inference latency per document of the local systems "
                  "(classifier only for frozen encoders, whose grids ran as four parallel jobs, so their timings are indicative; fine-tuning and inference of "
                  "the fine-tuned encoders on the GPU stated in the text; LLM latency and cost in ⟦T:llm⟧).",
                  "columns": ["System", "Training (s)", "Inference (ms/doc)", "Features"], "col_widths": [0.46, 0.18, 0.18, 0.18], "font_size": 15, "rows": rows})
    lexc = row_of(cost, lex["system"]) if cost is not None else None
    ftc = row_of(cost, "BERTurk fine-tuned") if cost is not None else None
    B.append(P(
        f"⟦T:cost⟧ lists training and inference costs. The lexical systems train in about a second on a CPU "
        f"({lexc['fit_seconds_trainval']:.1f} s for {lex['system']}) and classify a document in a fraction of a millisecond, which makes retraining on "
        f"every new month of data trivial. Frozen encoders add a one-time encoding cost per document (tens of milliseconds on a GPU, hundreds on a CPU) "
        f"and a classifier that trains in seconds to about a minute; fine-tuning BERTurk took {ftc['fit_seconds_trainval']/60:.1f} minutes per run on "
        f"the GPU and inference {ftc['predict_ms_per_doc']:.1f} ms per document. LLM calls add a median latency of one to two seconds and a cost of "
        f"USD {usd('gpt-6-luna')}–{usd('gemini-3.8-flash')} per 1,000 documents with retrieval (⟦T:llm⟧), which the cascade reduces in proportion "
        f"to the share it sends. These figures are the basis of the deployment discussion below."))

    # ---------------------------------------------------------------- 6 discussion
    B.append(H(1, "6. Discussion"))
    B.append(H(2, "6.1. What the results support"))
    tr_ = lambda v: f"{v:.0f}"
    B.append(P(
        "Four findings are robust to the choice of estimate (chronological test or grouped cross-validation) and to sampling variation. First, a lexical "
        "model on Turkish text is a strong baseline for institutional routing: the correspondence is templated, the discriminative information is lexical "
        "(form names, unit names, domain terms), and TF-IDF with a linear classifier captures it at negligible cost, "
        f"{100*(ft['accuracy']-lex['accuracy']):.1f}–{100*(qwen['accuracy']-lex['accuracy']):.1f} accuracy points below the best local encoders and "
        "with practically the same macro-F1 as the fine-tuned BERTurk. Any claim that a representation model improves routing must therefore be made "
        "against this baseline, which the earlier version of this study and several related studies omit. Second, Turkish should be processed "
        f"natively: machine translation lowered the accuracy of the lexical model by {tr_drop:.1f} points on identical documents and did not help an "
        "encoder consistently, whereas stemming and lemmatisation were interchangeable. Third, what makes LLMs useful for an institution-specific label "
        "set is retrieval rather than instructions: with unit descriptions alone they stayed below every supervised model, while ten retrieved past "
        f"documents raised them by {100*min(gains):.0f}–{100*max(gains):.0f} points, "
        f"{100*(hai['accuracy']-knn.get('test_accuracy', 0)):.0f}–{100*(gem['accuracy']-knn.get('test_accuracy', 0)):.0f} points above the vote of "
        "the retrieved documents themselves, and gave the best of them the highest macro-F1 of the study. Fourth, local confidence is informative: "
        "errors of the fine-tuned and lexical models concentrate in the low-confidence tail, which is what lets a cascade match the accuracy of the "
        f"LLM while calling it for {100*cg['llm_share']:.0f}% of the documents."))
    B.append(P(
        "Three qualifications apply. Absolute confidence thresholds chosen on one half-year lost up to three points of accuracy on the next; "
        f"validation macro-F1 ranked {big['system'].split(' / ')[0]} almost level with Gemini but overstated its test macro-F1 by "
        f"{small(round(100*(big['val_macro_f1']-big['macro_f1'])))} points; and differences of one to two accuracy points between the best local models depend on "
        "the seed of the fine-tuned model. The first two argue for periodic re-validation on recent documents rather than for a one-off model choice, "
        "the third for reporting seed variation. The practical consequence is that the useful question is not 'which model is most accurate?' but "
        "'how many documents can be automated at the accuracy the institution accepts, and what happens to the rest?'. The framework answers it with "
        "the top-3 suggestion, deferral to staff, the back-off to department offices and the LLM cascade; with the back-off, only "
        f"{pct(hb90['test_staff_share'], 0)}–{pct(hb95['test_staff_share'], 0)}% of the documents still need a manual first-level decision."))
    B.append(P(
        "Decision support also needs reasons that a dispatcher can check. The cascade supplies them without extra machinery: the retrieved past "
        "documents and their units, which the LLM sees, can be displayed with every suggestion as case-based evidence (Kenny et al., 2021), and the "
        "lexical model is directly inspectable (Rudin, 2019), because every decision decomposes into the weights of the terms that the document "
        "contains. De-identification matters here as well. Staff and regular senders recur across documents, so a model trained on the original "
        "text can learn their names as routing cues; this helps as long as the same people handle the same matters, but it conflicts with privacy "
        "and breaks when staff change. On de-identified text only the presence of a name, not its identity, is available to the models, and the "
        "remaining cues are the form names, unit names and domain terms that a dispatcher would also use."))
    B.append(H(2, "6.2. Relation to previous findings"))
    B.append(P(
        f"The preliminary study on the same data reported 74.3% accuracy for a Universal Sentence Encoder with an SVM on a random split, after translation "
        f"to English. Its closest counterpart here, the distilled multilingual USE with an RBF SVM, reached {pct(muse['accuracy'])}% on the chronological "
        f"test and {pct(muse['cv_accuracy_mean'])}% in grouped cross-validation. The difference has three measurable sources: the chronological test is "
        f"harder than a random split (grouped cross-validation, which still shuffles time, gives up to {gap.get('max', 0):.1f} points more), "
        f"identical texts no longer cross partitions, and no configuration was selected on the test partition; a fourth, not quantified here, is "
        f"that person names are masked in the present study. Under grouped cross-validation the best encoders (Qwen3-Embedding-4B "
        f"{pct(qwen['cv_accuracy_mean'])}%, multilingual-E5 base {pct(e5b['cv_accuracy_mean'])}%, fine-tuned BERTurk {pct(ft['cv_accuracy_mean'])}%) "
        f"are slightly above the earlier figure, whereas on the chronological test no system reaches it; the cascade comes closest "
        f"({pct(cg['test_accuracy'])}%). The finding of the earlier study that contextual encoders outperform static embeddings is consistent "
        f"with the present results, but its conclusion that lemmatisation removes the translation cost could not be supported by its implementation, which "
        f"lemmatised the English translation; the controlled comparison here shows that translation is harmful for the lexical model and of no "
        f"consistent benefit for an encoder. The results also agree with the ticket-routing literature: a thresholded lexical model with abstention is "
        f"a strong, cheap and auditable policy (Gupta, 2026); fine-tuned encoders are among the most accurate supervised models (Alsaç et al., 2025; "
        f"Karakaş & Şimşek, 2026), although a large frozen embedding model was at least as accurate as the fine-tuned BERTurk here; and zero-shot "
        f"LLMs do not know institution-specific classes, whereas retrieval-augmented prompting makes them competitive and, for the best model, the "
        f"strongest classifier of the comparison in macro-F1."))
    B.append(H(2, "6.3. Limitations"))
    hs_share = float(hs_f1.iloc[0]) / n_cls if len(hs_f1) else np.nan
    B.append(P(
        "The data come from two departments of a single university and from a single EDMS; the unit set, the share of templated correspondence and the "
        "label noise are institution-specific, and transfer to other institutions requires retraining and a new validation partition. Labels are the first "
        "dispatch recorded by staff and inherit their inconsistencies; the label-consistency ceiling of Section 3.3 bounds what any model can reach and "
        "limits the interpretation of differences between the best systems. The unit hierarchy enters only through the back-off and the top-down "
        "decoding of flat classifiers (Section 5.6); classifiers trained hierarchically were not evaluated. The Health Services Unit is too small to "
        f"be evaluated, yet its single test document moves macro-F1 by up to {small(round(100*hs_share))} points (Section 5.4). A dedicated few-shot method such "
        "as SetFit (Tunstall et al., 2022) was not evaluated: with hundreds to thousands of labelled "
        "documents for the large units, full fine-tuning is the stronger instance of the same idea, and for the smallest units the retrieval-augmented "
        "LLM, which needs no training, plays the few-shot role; a learning-curve study with reduced labelled data would show where the approaches "
        "cross. The temperature used for calibration was fitted on the model trained without the validation documents and transferred to the "
        "refitted model, which is an approximation. LLM results reflect specific model versions and prices at the time of the experiment, and each "
        "answer is a single sample (Gemini at temperature 0, Claude Haiku and GPT-6 Luna at their default temperature), so repeated runs would vary "
        "slightly; Claude Sonnet 5.5 could not be included in the final comparison (Section 4.5). De-identification with a named-entity recogniser is "
        "imperfect: it can miss names written in unusual forms and mask words that are not names, the English recogniser masked more spans than the "
        "Turkish one (Section 5.2), and indirect identifiers remain, so an institution deploying the cascade must assess the residual disclosure risk "
        "under its own regulation; documents whose de-identified text consists of little more than a name placeholder cannot be routed from their "
        "text at all. Finally, the cascade and the deferral are simulated on stored predictions; an operational study with dispatchers would be "
        "needed to measure the actual reduction of processing time and of misrouting, which this paper does not claim."))

    # ---------------------------------------------------------------- 7 conclusion
    B.append(H(1, "7. Conclusion"))
    B.append(P(
        f"This study reformulated the automatic routing of EDMS documents as a decision-support problem and evaluated it under a leakage-controlled, "
        f"chronological protocol on {n_docs:,} de-identified Turkish documents with {n_cls} target units. A lexical baseline on natively processed "
        f"Turkish text was only {100*(ft['accuracy']-lex['accuracy']):.1f}–{100*(qwen['accuracy']-lex['accuracy']):.1f} accuracy points below the best "
        f"encoders and matched the fine-tuned BERTurk in macro-F1; the fine-tuned BERTurk and a large frozen Qwen3 embedding were the most accurate "
        f"local models ({pct(ft['accuracy'])}% and {pct(qwen['accuracy'])}%); and an LLM reached the highest accuracy and macro-F1 "
        f"({pct(gem['accuracy'])}% and {f3(gem['macro_f1'])}) only when retrieved past documents compensated for the institution-specific class "
        f"definitions, with a small accuracy advantage over the best local model and a clear one in macro-F1. Machine translation, used in earlier "
        f"work, lowered the accuracy of the lexical model. Local confidence was informative enough to automate about a quarter of the routing at "
        f"{pct(d_best['accuracy@95'], 0)}% accuracy, and a cascade that passes the {100*cg['llm_share']:.0f}% least confident documents to an LLM "
        f"reached {pct(cg['test_accuracy'])}% accuracy at {100*cg['llm_share']:.0f}% of the LLM cost. Temperature scaling made the confidence of the "
        f"fine-tuned model interpretable, and backing off to department offices reduced the documents that need a manual first-level decision to "
        f"{pct(hb90['test_staff_share'], 0)}–{pct(hb95['test_staff_share'], 0)}%; thresholds chosen on one half-year lost up to three points on the "
        f"next, so a deployment should recalibrate them regularly. Future work includes hierarchically trained routing models, sender and "
        f"correspondence-direction features (the sender unit alone raised the accuracy of the lexical model by about "
        f"{small(round(100*(sr['accuracy']-base['accuracy'])))} points), on-premise open-weight LLMs for the second stage, which would keep every document "
        f"inside the institution, and a prospective study of the framework in the dispatch workflow."))

    # ---------------------------------------------------------------- declarations
    # author contributions and funding are identifying: in the anonymized copy they appear on the title page only
    if not ANON:
        B.append(H(1, "CRediT authorship contribution statement"))
        B.append(P(S.CREDIT))
    B.append(H(1, "Declaration of competing interest"))
    B.append(P(S.COMPETING))
    if not ANON:
        B.append(H(1, "Funding"))
        B.append(P(S.FUNDING))
    B.append(H(1, "Ethics statement"))
    B.append(P(S.ethics(ANON)))
    B.append(H(1, "Data and code availability"))
    B.append(P(S.data(ANON)))
    B.append(H(1, S.AI_TITLE))
    B.append(P(S.AI))

    # ---------------------------------------------------------------- references
    B.append(H(1, "References"))
    B.append({"type": "references", "items": sorted(REFERENCES, key=ref_key)})

    # ---------------------------------------------------------------- appendices
    B.append({"type": "pagebreak"})
    B.append(H(1, "Appendix A. Results of every classifier family"))
    allc = R["all_configurations"]
    if allc is not None and len(allc):
        def hp(params, system):
            try:
                d = json.loads(params)
            except Exception:
                return str(params)[:70]
            if " / RAG-10" in system or " / zero-shot" in system:
                return "see Section 4.5"
            skip = {"seed", "max_len", "batch", "precision", "n_estimators", "max_features", "metric", "weights", "gamma", "kind"}
            out = [f"{k.replace('class_weight', 'cw').replace('min_samples_leaf', 'leaf')}={'none' if v is None else v}" for k, v in d.items() if k not in skip]
            return ", ".join(out) if out else d.get("kind", "–")
        rows = [[r.system, f"{100*r.accuracy:.1f}", f"{100*r.macro_f1:.1f}", f"{100*r.weighted_f1:.1f}", f"{100*r.balanced_accuracy:.1f}",
                 f"{100*r.top3_accuracy:.1f}" if not pd.isna(r.top3_accuracy) else "–", hp(r.params, r.system)] for r in allc.itertuples()]
        B.append({"type": "table", "caption": "Table A1. Test-partition results of every representation × classifier family (values in %) with the hyper-parameters selected on validation (cw: class weight; leaf: minimum leaf size; RF with 200 trees, KNN with cosine distance and distance weighting, RBF SVM with γ = scale; fine-tuned encoders: learning rate and epochs, one row per seed).",
                  "columns": ["System", "Acc.", "Macro-F1", "wF1", "Bal. acc.", "Top-3", "Selected hyper-parameters"],
                  "col_widths": [0.3, 0.08, 0.09, 0.08, 0.09, 0.08, 0.28], "font_size": 13, "rows": rows})
    B.append(H(1, "Appendix B. LLM prompt"))
    from llm import SYSTEM, UNIT_DESCRIPTIONS  # noqa: E402
    B.append(P("System instruction (translated from Turkish): *You are an assistant that routes incoming documents in a university's electronic document management "
               "system to the unit that must act on them. The subject and content of the document will be given. From the list of units below, rank the three units "
               "to which the document is most likely to be dispatched for action, starting with the most likely. Use the unit names exactly as written. Return only "
               "JSON: {\"birimler\": [\"unit1\", \"unit2\", \"unit3\"]}.* The list of units with their duty descriptions follows; in the retrieval-augmented mode the user "
               "message contains the ten most similar past documents with their units before the document to be routed."))
    B.append({"type": "table", "caption": "Table B1. Unit descriptions used in the prompt (Turkish, as sent to the models).", "columns": ["Unit", "Description"],
              "col_widths": [0.3, 0.7], "font_size": 14, "rows": [[u, d] for u, d in UNIT_DESCRIPTIONS.items()]})
    return {"meta": meta, "blocks": B}


def ref_key(r):
    """Alphabetical key that ignores diacritics (Akça < Akın < Alsaç; Søgaard sorts as Sogaard)."""
    import unicodedata
    t = r.replace("ı", "i").replace("ø", "o").replace("Ø", "O").replace("ł", "l")
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()


REFERENCES = [
    "Deng, J., Krause, J., Berg, A. C., & Fei-Fei, L. (2012). Hedging your bets: Optimizing accuracy-specificity trade-offs in large scale visual recognition. In *2012 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3450–3457). IEEE. https://doi.org/10.1109/CVPR.2012.6248086",
    "Desai, S., & Durrett, G. (2020). Calibration of pre-trained transformers. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 295–302). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.21",
    "Kenny, E. M., Ford, C., Quinn, M., & Keane, M. T. (2021). Explaining black-box classifiers using post-hoc explanations-by-example: The effect of explanations and error-rates in XAI user studies. *Artificial Intelligence, 294*, 103459. https://doi.org/10.1016/j.artint.2021.103459",
    "Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence* (pp. 2901–2907). AAAI Press. https://doi.org/10.1609/aaai.v29i1.9602",
    "Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In *Advances in Neural Information Processing Systems 32*.",
    "Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x",
    "Silla, C. N., Jr., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. *Data Mining and Knowledge Discovery, 22*(1–2), 31–72. https://doi.org/10.1007/s10618-010-0175-9",
    "Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., & Pereg, O. (2022). Efficient few-shot learning without prompts. *arXiv*. https://arxiv.org/abs/2209.11055",
    "Akça, O., Bayrak, G., Issifu, A. M., & Ganiz, M. C. (2022). Traditional machine learning and deep learning-based text classification for Turkish law documents using transformers and domain adaptation. In *2022 International Conference on INnovations in Intelligent SysTems and Applications (INISTA)* (pp. 1–6). IEEE. https://doi.org/10.1109/INISTA55318.2022.9894051",
    "Akın, A. A., & Akın, M. D. (2007). Zemberek, an open source NLP framework for Turkic languages. *Structure, 10*, 1–5.",
    "Alsaç, A., Yılmaz, Ü., Koçoğlu, F. Ö., & Şeker, Ş. E. (2025). Towards intelligent IT service management: A comparative evaluation of machine learning and language models for ticket classification and assignment. In *2025 10th International Conference on Computer Science and Engineering (UBMK)*. IEEE. https://doi.org/10.1109/UBMK67458.2025.11207046",
    "Alzoubi, Y. I., Topcu, A. E., & Erkaya, A. E. (2023). Machine learning-based text classification comparison: Turkish language context. *Applied Sciences, 13*(16), 9428. https://doi.org/10.3390/app13169428",
    "Binici, K. (2019). Makine öğrenmesi yaklaşımıyla e-belgelere standart dosya plan numaralarının otomatik olarak atanması üzerine bir çalışma [A study on automatic assignment of standard classification numbers to e-records with the machine learning approach]. *Bilgi Yönetimi, 2*(2), 116–126. https://doi.org/10.33721/by.654464",
    "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics, 5*, 135–146. https://doi.org/10.1162/tacl_a_00051",
    "Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., … Amodei, D. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems 33* (pp. 1877–1901).",
    "Cer, D., Yang, Y., Kong, S., Hua, N., Limtiaco, N., St. John, R., … Kurzweil, R. (2018). Universal Sentence Encoder for English. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 169–174). https://doi.org/10.18653/v1/D18-2029",
    "Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. *arXiv*. https://arxiv.org/abs/2402.03216",
    "Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to use large language models while reducing cost and improving performance. *arXiv*. https://arxiv.org/abs/2305.05176",
    "Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory, 16*(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406",
    "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., … Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440–8451). https://doi.org/10.18653/v1/2020.acl-main.747",
    "Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018",
    "Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.",
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186). https://doi.org/10.18653/v1/N19-1423",
    "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197",
    "Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. Chapman & Hall/CRC.",
    "El-Yaniv, R., & Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research, 11*, 1605–1641.",
    "Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). Language-agnostic BERT sentence embedding. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 878–891). https://doi.org/10.18653/v1/2022.acl-long.62",
    "Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems 30* (pp. 4878–4887).",
    "Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences, 120*(30), e2305016120. https://doi.org/10.1073/pnas.2305016120",
    "Gorman, K., & Bedrick, S. (2019). We need to talk about standard splits. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 2786–2791). https://doi.org/10.18653/v1/P19-1267",
    "Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class classification: An overview. *arXiv*. https://arxiv.org/abs/2008.05756",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 1321–1330).",
    "Gupta, S. (2026). Cost-aware abstention for LLM-based IT ticket classification. *arXiv*. https://arxiv.org/abs/2610.31657",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
    "Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J. (2023). A call to reflect on evaluation practices for failure detection in image classification. In *The Eleventh International Conference on Learning Representations (ICLR 2023)*.",
    "Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In *Machine Learning: ECML-98* (LNCS 1398, pp. 137–142). Springer. https://doi.org/10.1007/BFb0026683",
    "Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804. https://doi.org/10.1016/j.patter.2023.100804",
    "Karakaş, S., & Şimşek, Y. (2026). Do we still need fine tuning? Turkish sentiment analysis in the era of large language models. *arXiv*. https://arxiv.org/abs/2606.29614",
    "Kılınç, D., Özçift, A., Bozyiğit, F., Yıldırım, P., Yücalar, F., & Borandağ, E. (2017). TTC-3600: A new benchmark dataset for Turkish text categorization. *Journal of Information Science, 43*(2), 174–185. https://doi.org/10.1177/0165551515620551",
    "Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). ALBERT: A lite BERT for self-supervised learning of language representations. In *International Conference on Learning Representations (ICLR 2020)*.",
    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33* (pp. 9459–9474).",
    "Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., & Chen, W. (2022). What makes good in-context examples for GPT-3? In *Proceedings of Deep Learning Inside Out (DeeLIO 2022)* (pp. 100–114). https://doi.org/10.18653/v1/2022.deelio-1.10",
    "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., … Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv*. https://arxiv.org/abs/1907.11692",
    "Madras, D., Pitassi, T., & Zemel, R. (2018). Predict responsibly: Improving fairness and accuracy by learning to defer. In *Advances in Neural Information Processing Systems 31* (pp. 6147–6157).",
    "Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv*. https://arxiv.org/abs/1301.3781",
    "Mozannar, H., & Sontag, D. (2020). Consistent estimators for learning to defer to an expert. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 7076–7087).",
    "Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive text embedding benchmark. In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 2014–2037). https://doi.org/10.18653/v1/2023.eacl-main.148",
    "Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532–1543). https://doi.org/10.3115/v1/D14-1162",
    "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410",
    "Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 4512–4525). https://doi.org/10.18653/v1/2020.emnlp-main.365",
    "Republic of Türkiye. (2016). *Kişisel Verilerin Korunması Kanunu* [Personal Data Protection Law], Law No. 6698. Official Gazette No. 29677, 7 April 2016.",
    "Schwaar, S., Diez, F., Trebing, M., & Witznick, N. (2025). Study on text classification for public administration. *arXiv*. https://arxiv.org/abs/2504.09111",
    "Schweter, S. (2020). *BERTurk – BERT models for Turkish* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3770924",
    "Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys, 34*(1), 1–47. https://doi.org/10.1145/505282.505283",
    "Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management, 45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002",
    "Søgaard, A., Ebert, S., Bastings, J., & Filippova, K. (2021). We need to talk about random splits. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume* (pp. 1823–1832). https://doi.org/10.18653/v1/2021.eacl-main.156",
    "Sprague, R. H. (1995). Electronic document management: Challenges and opportunities for information systems managers. *MIS Quarterly, 19*(1), 29–49. https://doi.org/10.2307/249710",
    "Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification? In *Chinese Computational Linguistics (CCL 2019)* (LNCS 11856, pp. 194–206). Springer. https://doi.org/10.1007/978-3-030-32381-3_16",
    "Sun, X., Li, X., Li, J., Wu, F., Guo, S., Zhang, T., & Wang, G. (2023). Text classification via large language models. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 8990–9005). https://doi.org/10.18653/v1/2023.findings-emnlp.603",
    "Uysal, A. K., & Gunal, S. (2014). The impact of preprocessing on text classification. *Information Processing & Management, 50*(1), 104–112. https://doi.org/10.1016/j.ipm.2013.08.006",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30* (pp. 5998–6008).",
    "Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. *arXiv*. https://arxiv.org/abs/2402.05672",
    "Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. In *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 90–94).",
    "Yang, Y., Cer, D., Ahmad, A., Guo, M., Law, J., Constant, N., … Kurzweil, R. (2020). Multilingual Universal Sentence Encoder for semantic retrieval. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations* (pp. 87–94). https://doi.org/10.18653/v1/2020.acl-demos.12",
    "Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., … Zhou, J. (2025). Qwen3 Embedding: Advancing text embedding and reranking through foundation models. *arXiv*. https://arxiv.org/abs/2506.05176",
]


def number_blocks(M):
    """Assign table/figure numbers in order of appearance and substitute ⟦T:key⟧ / ⟦F:key⟧ tokens."""
    tn, fn = {}, {}
    for b in M["blocks"]:
        if b["type"] == "table":
            m = re.search(r"⟦T:(\w+)⟧", b["caption"])
            if m: tn[m.group(1)] = len(tn) + 1          # appendix tables keep their own labels
        if b["type"] in ("figure", "figure_missing"):
            m = re.search(r"⟦F:(\w+)⟧", b["caption"])
            if m: fn[m.group(1)] = len(fn) + 1
    def sub(x):
        if isinstance(x, str):
            x = re.sub(r"⟦T:(\w+)⟧", lambda m: f"Table {tn.get(m.group(1), '?')}", x)
            return re.sub(r"⟦F:(\w+)⟧", lambda m: f"Figure {fn.get(m.group(1), '?')}", x)
        if isinstance(x, list):
            return [sub(v) for v in x]
        if isinstance(x, dict):
            return {k: sub(v) for k, v in x.items()}
        return x
    M["blocks"] = sub(M["blocks"])
    return M


def main():
    R = load()
    M = number_blocks(build(R))
    OUT.mkdir(exist_ok=True)
    js, docx_out = (OUT / "manuscript_anon.json", OUT / "Manuscript_anonymized.docx") if ANON else (OUT / "manuscript.json", OUT / "Manuscript.docx")
    js.write_text(json.dumps(M, ensure_ascii=False, indent=1))
    n_pending = json.dumps(M, ensure_ascii=False).count(PENDING)
    subprocess.run(["node", str(OUT / "render_docx.js"), str(js), str(docx_out)], check=True)
    print(f"pending placeholders: {n_pending}; to be completed by the authors: {json.dumps(M, ensure_ascii=False).count('[[TO BE COMPLETED')}")


if __name__ == "__main__":
    main()

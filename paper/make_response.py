"""Builds the reviewer-comment -> change mapping (cover-letter appendix) as a Word document."""
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import statements as S          # noqa: E402
ANON = "--anon" in sys.argv     # copy for double-anonymized review (uploaded as the response to reviewers)

INTRO = (
    "This document accompanies the new submission of a substantially revised version of manuscript ESWA-D-26-37092 "
    "(\"Enhancement of document routing procedures in electronic document management systems through natural language processing and machine "
    "learning algorithms\"), which was rejected after review. The study was redesigned rather than edited: the experimental protocol, the set of "
    "models, the evaluation metrics, the released code and the text were all replaced. The tables below list every reviewer comment of the previous "
    "round and the corresponding change in the present manuscript, with the section in which it can be verified.")

# ---------------------------------------------------------------- numbers, read from the same result files as the manuscript
import pandas as pd             # noqa: E402
RES = OUT.parent / "results"; TAB = RES / "tables"
main = pd.read_csv(TAB / "main.csv"); ex = json.loads((TAB / "extra_stats.json").read_text())
hie = pd.read_csv(TAB / "hierarchical.csv"); cal = pd.read_csv(TAB / "calibration.csv")
abl = pd.read_csv(TAB / "ablation_lexical.csv").set_index("representation")
reps = pd.read_csv(TAB / "representatives.csv")


def row(contains):
    return main[main["system"].str.contains(contains, regex=False)].iloc[0]


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


final = reps.iloc[0]; gem = row(final["system"]); qwen = row("Qwen3-Embedding-4B"); ft = row("BERTurk fine-tuned")
lex = reps[reps["group"] == "Lexical (TF-IDF)"].iloc[0]; muse = row("distiluse")
cg = ex["cascade"]["Cascade BERTurk -> Gemini"]
hb95 = hie[hie["system"].str.startswith("BERTurk") & (abs(hie["target"] - 0.95) < 1e-9)].iloc[0]
cb = cal[cal["system"].str.startswith("BERTurk")].iloc[0]
tr_drop = 100 * (abl.loc["tfidf_content_tr", "accuracy"] - abl.loc["tfidf_content_en", "accuracy"])
local_mf1 = main.loc[main["group"] != "LLM", "macro_f1"].max()
staff = (100 * hie["test_staff_share"].min(), 100 * hie["test_staff_share"].max())
lex_gap = (100 * (ft["accuracy"] - lex["accuracy"]), 100 * (qwen["accuracy"] - lex["accuracy"]))
st = json.loads((RES / "dataset_stats.json").read_text())
ETHICS_REF = ("the written permission of the university that holds the records (approving body, date and reference number on the title page)"
              if ANON else f"the written permission of {S.PERMISSION['body']} ({S.PERMISSION['date']}, no. {S.PERMISSION['number']})")

RESULTS = (
    f"In brief, the revised study evaluates {st['n_documents']:,} de-identified documents and {st['n_classes']} target units chronologically "
    "(training up to 2021, validation on the first and testing on the second half of 2022) with validation-only model selection. The "
    f"retrieval-augmented {final['system'].split(' / ')[0]}, the system selected on validation, had the highest accuracy and macro-F1 "
    f"({pc(gem['accuracy'])}%, {gem['macro_f1']:.3f}; at most {local_mf1:.3f} for the local models), although its accuracy did not differ "
    f"significantly from a frozen Qwen3-Embedding-4B model with an SVM ({pc(qwen['accuracy'])}%); fine-tuned BERTurk reached "
    f"{pc(ft['accuracy'])}% and the TF-IDF baseline {pc(lex['accuracy'])}%, with practically equal macro-F1. A cascade passing the "
    f"{100 * cg['llm_share']:.0f}% least confident BERTurk decisions to the LLM reached {pc(cg['test_accuracy'])}% at "
    f"{100 * cg['llm_share']:.0f}% of the LLM cost; temperature scaling halved the calibration error, and a hierarchical back-off to department "
    f"offices left {100 * hb95['test_staff_share']:.0f}% of the documents to staff at a 95% target. Machine translation, the core of the earlier "
    f"pipeline, lowered the accuracy of the lexical model by {tr_drop:.1f} points. The earlier figure of 74.3% was obtained on a random split "
    f"with selection on the test data; its closest counterpart here obtains {pc(muse['accuracy'])}% on the chronological test and "
    f"{pc(muse['cv_accuracy_mean'])}% in grouped cross-validation, and the manuscript explains the difference. All text was de-identified before "
    f"any reported model was trained or queried, and the use of the records is covered by {ETHICS_REF}.")

R3 = [
    ("1. Inconsistency between the described preprocessing (Turkish lemmatisation) and the released code (lemmatisation of the English translation); the conclusion that lemmatisation removes the translation overhead was unsupported.",
     "Confirmed and corrected. The earlier code lemmatised the Google-Translate output with WordNet. The new pipeline processes Turkish natively (Turkish-aware casing, Snowball Turkish stemmer, Zeyrek/Zemberek morphological lemmatisation with a stated disambiguation rule) and evaluates translation separately as a controlled ablation on identical documents (Turkish body vs. machine-translated body, both with TF-IDF and with a sentence encoder). The discrepancy and its consequence for the earlier conclusion are stated explicitly.",
     "Sections 4.2, 5.2 (Table 6, Fig. 4), 6.2"),
    ("2. Insufficient description of the dataset and experimental design (number of documents, classes and their frequencies, partitions, missing values, duplicates, repeated templates, documents of the same case in several partitions, chronological evaluation).",
     "A dataset-construction table with every filtering step and count, the class distribution per partition, document-length statistics, the number of distinct texts and of documents with identical twins, the treatment of multi-dispatch chains and broadcast circulars, the label-consistency ceiling, and the number of test documents with an identical text in training are reported. The main evaluation is chronological (train ≤ 2021, validation 2022 H1, test 2022 H2); grouped cross-validation keeps identical texts in one fold.",
     "Section 3 (Tables 1–2, Fig. 1), Section 4.6"),
    ("3. Model selection on a single random train-test split without a validation stage; optimistic estimate when 126 configurations are compared; confidence intervals, variability and statistical comparisons requested.",
     "All hyper-parameters and the classifier family of every representation are selected on the validation partition only; the test partition is used once. 95% bootstrap intervals, grouped 5×2 cross-validation (mean ± SD) and exact McNemar tests with Holm correction are reported. Conclusions are restricted to differences these tests support.",
     "Sections 4.4, 4.6, 5.1, 5.4 (Tables 5, 8–9; Fig. 3)"),
    ("4. Limited methodological novelty; stronger baselines needed (TF-IDF + linear SVM/LR, modern sentence embeddings, Turkish/multilingual transformers, fine-tuned transformer, efficient few-shot).",
     "Added: TF-IDF word and character n-gram models with LR/linear SVM (which remain only " + f"{lex_gap[0]:.1f}–{lex_gap[1]:.1f}" + " accuracy points below the best encoders, with practically the same macro-F1 as fine-tuned BERTurk), eleven multilingual and Turkish sentence encoders (paraphrase-multilingual SBERT, mUSE, LaBSE, mE5 base and large, BGE-M3, Qwen3-Embedding 0.6B and 4B, BERTurk-SBERT, frozen BERTurk), end-to-end fine-tuned BERTurk and XLM-R, and three LLMs of three providers in zero-shot and retrieval-augmented few-shot modes, with the vote of the retrieved documents as a control. The contribution is reframed as a confidence-aware routing framework (top-k suggestion, calibrated deferral, hierarchical back-off to department offices, LLM cascade) evaluated under a leakage-controlled protocol, not as a grid of combinations. Efficient few-shot fine-tuning (SetFit) is discussed and its omission justified: the archive provides hundreds to thousands of labelled documents for the large units, where full fine-tuning is the stronger instance of the same idea, and the retrieval-augmented LLM covers the few-shot regime of the smallest units.",
     "Sections 1, 2.3, 4.3–4.5, 5.1, 5.6–5.7, 6.3"),
    ("5. Inadequate evaluation of class imbalance and routing errors (macro-F1, per-class precision/recall, support, multiclass confusion matrix, error analysis of confused units).",
     "Macro-F1, balanced accuracy, per-unit precision/recall/F1 with training and test support, a 13×13 confusion matrix with department frames, within- vs. cross-department error counts and the most frequent confusions are reported and discussed.",
     "Section 5.3 (Table 7, Figs. 5–6), Section 4.6"),
    ("6. Lack of statistical significance and robustness analysis; ablation separating subject, content, preprocessing and representation.",
     "Bootstrap intervals, cross-validation variance, McNemar tests, and ablations of preprocessing (raw/stem/lemma/char), input field (subject only, body only, both), translation and sender-unit context are included.",
     "Sections 5.2, 5.4"),
    ("7. Practical deployment claims without evidence; top-k routing accuracy and confidence-based human intervention suggested.",
     "The deployment claims are withdrawn and replaced by measured quantities: top-3 and lenient accuracy, risk-coverage curves, validation-chosen thresholds with the resulting automation rate and accuracy on the test half-year, calibrated and recency-based thresholds, a hierarchical back-off to department offices (staff share " + f"{staff[0]:.0f}–{staff[1]:.0f}" + "%), a local-to-LLM cascade with cost and latency, and training/inference times. The limitations section states that workload reduction was not measured.",
     "Sections 4.5, 5.5–5.8, 6.3 (Tables 10–14, Figs. 7–8)"),
    ("8. Reproducibility and data availability statements inconsistent with the repository; exact code version, checkpoints, seeds and hyper-parameters requested.",
     "The data statement now distinguishes the restricted raw export (personal data; access subject to the data controller) from the released materials: complete pipeline, partition and fold identifiers, seeds, hyper-parameter grids and selected values, prediction files and prompts. Checkpoint names are listed in Table 3.",
     "Data and code availability; Section 3.4; Tables 3–4"),
    ("9. Table 2 interchanged FP and FN; binary definitions used for a multiclass problem; averaging strategies undefined.",
     "The confusion-matrix table was removed. Per-unit one-versus-rest definitions (Eqs. 1–3), macro averaging (Eq. 4), support weighting and balanced accuracy are defined; the identity of weighted recall and accuracy is noted.",
     "Section 4.6"),
    ("10. Generalisation claims about USE not supported.",
     "All conclusions are restricted to the evaluated dataset, checkpoints and protocol; the discussion separates findings that are robust across both estimates from those within sampling variation.",
     "Sections 6.1–6.3"),
    ("Suggested references (Sentence-BERT, SetFit/efficient few-shot, interpretable decision support).",
     "Sentence-BERT and its multilingual extension are cited and evaluated; SetFit (Tunstall et al., 2022) is cited and discussed; interpretable decision support is addressed with references (Rudin, 2019; Kenny et al., 2021), case-based evidence from the retrieved documents and the directly inspectable lexical model; calibration (Guo et al., 2017; Naeini et al., 2015; Desai & Durrett, 2020; Ovadia et al., 2019) and hierarchical classification (Silla & Freitas, 2011; Deng et al., 2012) are covered; the related-work section was rewritten around contemporary sentence representations, LLM classification and cost-aware cascades.",
     "Sections 2, 6.1, 6.3"),
    ("Minor: long algorithm descriptions; crowded 126-combination figures; architectures/checkpoints/dimensions/pooling; stemming vs. lemmatisation; computational requirements; single-institution limitation; citation relevance; English; Highlights; graphical abstract.",
     "Algorithm tutorials removed; results shown as dot plots with intervals and compact tables; Table 3 lists checkpoints, dimensions and pooling; stemming and lemmatisation are defined and contrasted; training/inference costs reported; limitations state single-institution scope; every reference was re-checked for relevance and accuracy; the text was rewritten; Highlights state only measured findings; no graphical abstract is submitted.",
     "Throughout"),
]
R4 = [
    ("1. How data were divided and how the best combination was selected.", "Chronological train/validation/test partitions and validation-only selection are described; see Reviewer 3, items 2–3.", "Sections 3.2, 4.4, 4.6"),
    ("2. Table 1 (low resolution) did not report classes, counts or imbalance.", "The screenshot table was replaced by a construction table and a class-by-partition table; the imbalance is quantified.", "Tables 1–2, Fig. 1"),
    ("3. Embedding procedure under-specified (versions, languages, tokenisation, truncation, pooling, frozen vs. fine-tuned).", "Table 3 lists every checkpoint with training signal, languages, dimension and parameters; truncation (256 tokens), prefixes, pooling and the frozen/fine-tuned distinction are stated.", "Section 4.3, Table 3"),
    ("4. Classifier hyper-parameters, tuning procedure, scaling and validation strategy missing.", "Table 4 gives the grid; scaling of dense features and the validation-based selection are described; selected values are listed in Appendix A.", "Section 4.4, Table 4, Appendix A"),
    ("5. Preprocessing tools unspecified; application of stemming to contextual models unjustified.", "Tools are named (NLTK stop words, Snowball Turkish, Zeyrek); contextual encoders receive natural text and the reason is given.", "Section 4.2"),
    ("6. Missing baselines: TF-IDF + linear SVM; fine-tuned Turkish/multilingual transformer.", "Both added; TF-IDF + linear models are the reference system; BERTurk and XLM-R are fine-tuned end to end.", "Sections 4.3, 5.1"),
    ("7. Accuracy/weighted-F1 dominated by large classes; weighted recall equals accuracy; macro-F1, balanced accuracy, per-class metrics and confusion matrix requested.", "All requested metrics are reported; weighted recall is no longer reported.", "Sections 4.6, 5.3"),
    ("8. FP/FN labels interchanged in Table 2.", "Table removed; definitions corrected.", "Section 4.6"),
]
R6 = [
    ("1. No fine-tuning, no Turkish pre-trained models (BERTurk), no routing-specific mechanism (hierarchical routing, confidence-calibrated deferral).", "Fine-tuned BERTurk and XLM-R, Turkish SBERT and frozen BERTurk added. Both requested mechanisms are now implemented and evaluated: confidence calibration by temperature scaling with expected calibration error and reliability curves (ECE of BERTurk " + f"{cb['ece_test_before']:.3f} → {cb['ece_test_after']:.3f}" + " on the test half-year), compared across three deferral rules (empirical, calibrated, recent-quarter), and a hierarchical back-off that routes a document to its unit when the unit is certain, to the office of its department when only the department is certain, and to staff otherwise (staff share " + f"{100 * hb95['test_staff_share']:.0f}% instead of {100 * (1 - hb95['test_unit_share']):.0f}%" + " at a 95% target). Department-level accuracy and the error analysis by department are also reported.", "Sections 4.5, 5.3, 5.5–5.7 (Tables 10–12, Figs. 7–8)"),
    ("2. Dataset characterisation and validation protocol missing; duplicate document numbers routed to multiple units (multi-label ambiguity, leakage); timestamps from 2020 contradict the stated 2021–2022 window.", "Multi-dispatch chains are analysed (first dispatch = label; set of all units for lenient scoring); identical texts are grouped; the actual dispatch period (" + f"{pd.Timestamp(st['dispatch_time_min']):%B %Y} – {pd.Timestamp(st['dispatch_time_max']):%B %Y}" + "; eight documents had been registered in late 2019) is stated, and the document dates are used for the chronological partitions.", "Sections 3.1–3.3"),
    ("3. Numerical inconsistencies between Figure 5, the text and the heatmaps.", "All figures and tables are generated by the released scripts from the same result files; the manuscript text is produced from those files, so numbers cannot diverge.", "Section 5; code repository"),
    ("4. Real names exposed in Table 1; dataset described as public without anonymisation or ethics approval.", "No personal data appear in the manuscript; the restricted status of the export, the released materials and the cloud services on which the records were processed are described; all text is de-identified (person names, e-mail addresses, telephone and identity numbers) before any model is trained or queried; and the Ethics statement reports " + ETHICS_REF + " and states that no separate ethics committee approval was obtained, as the study analyses existing records and involves no human participants.", "Sections 3.4, 4.2; Ethics statement"),
    ("5. Uncited references, irrelevant citations (haematology), self-citations for elementary definitions.", "The reference list was rebuilt; every entry is cited in the text and relevant to the claim; the earlier self-citations were removed.", "References"),
    ("Typographical errors (missing periods, tense).", "The manuscript was rewritten in full.", "Throughout"),
]
R7 = [
    ("Abbreviations in the abstract; abstract structure.", "Every abbreviation in the abstract is defined at first use (EDMS, TF-IDF, LLM) and model names are written out (e.g. XLM-RoBERTa); the abstract is a single paragraph of fewer than 250 words.", "Abstract"),
    ("Section 1 needs citation support, a clear contribution statement and an organisation paragraph.", "The introduction cites the motivating literature, lists five contributions and ends with the organisation of the paper.", "Section 1"),
    ("A dedicated motivation/research-gap section before the methodology.", "The introduction states the gap (random splits, missing baselines, translation, no routing mechanism, privacy) and Section 2.5 reviews the evaluation literature that motivates the protocol.", "Sections 1, 2.5"),
    ("Paragraph structure and sentence length.", "Rewritten.", "Throughout"),
    ("References for adopted models, algorithms and metrics.", "Every model, classifier family and metric is cited at the point of use.", "Sections 2, 4"),
    ("Results section needs clearer interpretation, comparison with related studies and citations.", "Results are interpreted per table; Section 6.1 states which findings the data support, and Section 6.2 compares them with the earlier version and with the routing/ticket literature.", "Sections 5, 6.1–6.2"),
    ("Reference list outdated.", "The list now includes work from 2022–2026 on sentence encoders, LLM classification, cost-aware cascades, Turkish text classification and public-administration routing, alongside foundational references.", "References"),
    ("Inconsistent F1 terminology.", "Macro-F1 is the primary measure throughout; weighted F1 is labelled as such wherever it appears.", "Throughout"),
    ("Highlights too descriptive or over-stated.", "Highlights state only measured findings: the protocol, the lexical baseline and the translation result, the role of retrieval for LLMs, the cascade and the back-off.", "Highlights"),
    ("A dedicated Proposed Method section.", "Section 4 (Proposed framework) with an overview figure and sub-sections for each stage.", "Section 4"),
    ("Suggested citation of an unrelated ESWA paper on memetic scheduling.", "Not cited: it does not support any claim in the manuscript.", "–"),
]


def table(cap, rows):
    return {"type": "table", "caption": cap, "columns": ["Reviewer comment", "Change in the revised manuscript", "Where"],
            "col_widths": [0.36, 0.5, 0.14], "font_size": 16, "rows": [list(r) for r in rows]}


M = {"meta": {"title": f"Summary of changes relative to {S.PREVIOUS_ID} (reviewer comments and corresponding revisions)",
              "authors": "" if ANON else ", ".join(a["name"] for a in S.AUTHORS), "affiliations": [],
              "corresponding": "Response to the reviewers of the previous submission",
              "creator": "Anonymous" if ANON else "; ".join(a["name"] for a in S.AUTHORS)},
     "blocks": [{"type": "paragraph", "text": INTRO}, {"type": "paragraph", "text": RESULTS},
                {"type": "heading", "level": 1, "text": "Reviewer 3"}, table("Table R3. Reviewer 3.", R3),
                {"type": "heading", "level": 1, "text": "Reviewer 4"}, table("Table R4. Reviewer 4.", R4),
                {"type": "heading", "level": 1, "text": "Reviewer 6"}, table("Table R6. Reviewer 6.", R6),
                {"type": "heading", "level": 1, "text": "Reviewer 7"}, table("Table R7. Reviewer 7.", R7)]}
js, out = (OUT / "response_anon.json", OUT / "Response_to_reviewers_anonymized.docx") if ANON else (OUT / "response.json", OUT / "Summary_of_changes.docx")
js.write_text(json.dumps(M, ensure_ascii=False, indent=1))
subprocess.run(["node", str(OUT / "render_docx.js"), str(js), str(out)], check=True)

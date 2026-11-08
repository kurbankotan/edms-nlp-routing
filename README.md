# Confidence-aware document routing for EDMS — reproducible pipeline

Code, splits and evaluation protocol for the study *"Confidence-aware routing of institutional documents in electronic
document management systems: a leakage-controlled benchmark of lexical, embedding, fine-tuned and large language model
classifiers"* (Kotan & Kırışoğlu). The pipeline routes Turkish official documents of a university EDMS to the
organisational unit that must act on them, and turns the classifier into a routing mechanism with top-k suggestions,
confidence-based deferral to staff and an optional retrieval-augmented LLM second stage.

## Layout

```
data/        raw_dataset.xlsx (restricted; not in the repository), documents.pkl, text_variants.pkl
src/
  prepare_data.py     dispatch log -> document table, labels, chronological split, grouped folds, dataset_stats.json
  preprocess.py       Turkish-native variants (raw / Snowball stem / Zeyrek lemma) + English translation arm
  common.py           shared protocol: validation-based selection, test once, grouped 5x2 CV, metrics, bootstrap, McNemar
  run_tfidf.py        lexical models and ablations (preprocessing, input field, translation, sender context)
  embed.py            frozen sentence-embedding extraction (cached .npy)
  run_embeddings.py   classifier grid on frozen embeddings
  run_emb_parallel.py the frozen-embedding grid as parallel jobs (one representation and one results file per job)
  finetune.py         end-to-end fine-tuning (BERTurk, XLM-R): selection on validation, 3 seeds, optional CV
  llm.py              LLM routing (zero-shot / retrieval-augmented), de-identification, resumable JSONL cache,
                      answer parsing and a second pass (--requery) for answers without a valid unit
  analysis.py         tables (results/tables.xlsx) and figures (figures/)
  routing.py          calibration (temperature scaling, ECE), deferral rules and hierarchical back-off to department offices
paper/       extra_stats.py, make_manuscript.py + render_docx.js (manuscript generated from the result files)
results/     dataset_stats.json, results_*.csv, preds/, preds_val/, val_grid/, ft_runs_v2/, tables/, llm/
figures/
EBYS_Routing_Colab.ipynb   the whole pipeline as a resumable Colab notebook (Drive-backed)
```

## Protocol in one paragraph

Documents dated up to 2021-12-31 form the training partition, 2022 H1 the validation partition and 2022 H2 the test
partition. Every hyper-parameter and the classifier family of every representation are chosen on the validation
partition by macro-F1; the selected configuration is refitted on train+validation and evaluated **once** on the test
partition (predictions and class scores are stored in `results/preds/`). A stratified **group** 5-fold cross-validation
with two repeats (groups = identical texts) gives the complementary variance estimate. Deferral thresholds and the LLM
cascade share are chosen on validation predictions (`results/preds_val/`) and applied unchanged to the test partition.

## Running

```bash
pip install -r requirements.txt
python src/prepare_data.py && python src/preprocess.py
python src/run_tfidf.py                                   # lexical models (+ --with_sender for the context variant)
python src/embed.py && python src/run_emb_parallel.py    # frozen encoders (embedding extraction on a GPU)
python src/finetune.py --cv                               # fine-tuned encoders (GPU)
python src/llm.py --smoke && python src/llm.py            # needs API keys in the environment
python src/llm.py --requery && python src/llm.py --summarise
python src/analysis.py && python src/routing.py          # tables + figures, calibration and hierarchical routing
python paper/extra_stats.py                               # further statistics quoted in the paper
```

All scripts are resumable: finished representations, embeddings, fine-tuning runs and LLM responses are skipped on
re-runs. Seeds: data folds 0/1, classifiers 0, fine-tuning 0/1/2, bootstrap 0.

## Data availability

The raw EDMS export contains personal data and is available only under the conditions of the institutional ethics
approval. The repository releases the construction script, the document identifiers of every partition and fold
(`results/dataset_stats.json` and the identifiers stored with every prediction file), all predictions, prompts and
configuration, so that the evaluation can be audited without the raw data.

## LLM settings

Claude Haiku 5.5 / Sonnet 5.5, GPT-6 Luna and Gemini 3.8 Flash were queried in November 2026 with extended reasoning
switched off (Gemini: thinking budget 0, temperature 0; GPT-6 Luna: reasoning effort `none`, strict JSON schema;
Claude: thinking `disabled`, or `between_tools` for Sonnet 5.5, which rejects `disabled`), 300 output tokens and
de-identified text. Every response is stored in `results/llm/<model>__<mode>__<split>.jsonl` with its token counts and
effective setting; re-requested answers are appended with `"pass": 2`, and the last record of a document is its answer.

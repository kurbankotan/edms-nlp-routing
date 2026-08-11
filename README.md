# EDMS Document Routing via NLP and Machine Learning

> **Paper:** "Enhancement of Document Routing Procedures in Electronic Document Management Systems Through Natural Language Processing and Machine Learning Algorithms"  
> **Authors:** Kurban KOTAN, Serdar KIRIŞOĞLU  
> **Submitted to:** Applied Soft Computing (Elsevier)

---

## Overview

This repository contains the full experimental pipeline for automated document routing in Electronic Document Management Systems (EDMS) using NLP-based embedding models and supervised machine learning classifiers.

**126 model combinations** were evaluated across:
- **7 language representation models:** BERT, ALBERT, RoBERTa, USE, Word2Vec, GloVe, FastText
- **3 preprocessing strategies:** English translation, Lemmatisation, Stemming
- **6 classifiers:** Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Gaussian Naïve Bayes, SVM

**Best result:** USE + Lemma + SVM → **74.27% Accuracy**, **71.62% weighted F1-Score**

---

## Repository Structure

```
edms-nlp-routing/
│
├── src/
│   └── train_and_evaluate.py     # Main experimental script
│
├── data/
│   ├── raw_dataset.xlsx          # Place raw EDMS dataset here (see Data section)
│   └── preprocessed_data.pkl     # Auto-generated cache after first run
│
├── models/                       # Place pre-trained model files here
│   ├── GloVe/
│   │   └── glove.840B.300d-001.txt
│   ├── Word2Vec/
│   │   └── word2vec-google-news-300.bin
│   ├── USE/                      # TF Hub SavedModel directory
│   └── FastText/
│       └── cc.en.300.bin
│
├── results/
│   └── Final_Supervised_Performance.xlsx   # Auto-generated results
│
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt** covers:
```
sentence-transformers>=2.2.0
deep-translator>=1.9.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.7
beautifulsoup4>=4.11.0
gensim>=4.2.0
tensorflow>=2.10.0
tensorflow-hub>=0.13.0
fasttext>=0.9.2
openpyxl>=3.0.0
```

---

## Pre-trained Models

Download the following models manually and place them in the `models/` directory:

| Model | Source | Notes |
|-------|--------|-------|
| GloVe (840B, 300d) | [Stanford NLP](https://nlp.stanford.edu/projects/glove/) | `glove.840B.300d.txt` |
| Word2Vec (Google News) | [Google Archive](https://code.google.com/archive/p/word2vec/) | `GoogleNews-vectors-negative300.bin` |
| FastText (English, cc.en.300) | [Meta AI](https://fasttext.cc/docs/en/crawl-vectors.html) | `cc.en.300.bin` |
| USE (Universal Sentence Encoder) | [TF Hub](https://tfhub.dev/google/universal-sentence-encoder/4) | SavedModel format |

BERT, ALBERT, and RoBERTa are downloaded automatically via `sentence-transformers`.

---

## Dataset

The dataset used in this study consists of **33,829 official correspondence records** from Düzce University's EDMS (January 2021 – December 2022), collected under institutional ethics board approval.

**Due to ethics committee conditions, the full dataset cannot be released publicly.** A processed, anonymised sample will be made available upon journal acceptance. Please contact the corresponding author for research collaboration inquiries: `kurban.kotan@cbu.edu.tr`

Place your dataset as `data/raw_dataset.xlsx` with the following columns:
- `EvrakId` — Document ID
- `EvrakKonusu` — Document subject
- `EvrakIcerigi` — Document content
- `EvrakinGeregineKisiKullaniciBirimi` — Target routing unit (label)

---

## Usage

```bash
python src/train_and_evaluate.py
```

On first run, the script will:
1. Load and clean the raw dataset
2. Apply preprocessing (noise removal, stopword filtering, stemming/lemmatisation)
3. Translate Turkish text to English via Google Translate API
4. Cache the processed data to `data/preprocessed_data.pkl`

On subsequent runs, the cached file is loaded directly (skipping translation).

Results are saved to `results/Final_Supervised_Performance.xlsx`.

---

## Results Summary

| Rank | Combination | Precision | Recall | F1-Score | Accuracy |
|------|-------------|-----------|--------|----------|----------|
| 1 | USE – Lemma – SVM | 0.716 | 0.743 | 0.716 | 0.743 |
| 2 | USE – English – SVM | 0.714 | 0.739 | 0.713 | 0.739 |
| 3 | USE – Stem – SVM | 0.704 | 0.731 | 0.700 | 0.731 |
| 4 | RoBERTa – Lemma – LR | 0.692 | 0.714 | 0.700 | 0.714 |
| 5 | ALBERT – English – LR | 0.693 | 0.714 | 0.698 | 0.714 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{kotan2026edms,
  title   = {Enhancement of Document Routing Procedures in Electronic Document 
             Management Systems Through Natural Language Processing and 
             Machine Learning Algorithms},
  author  = {Kotan, Kurban and K{\i}r{\i}{\c{s}}o{\u{g}}lu, Serdar},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code: MIT License  
Dataset: Available upon request (subject to institutional ethics conditions)

---

## Contact

**Kurban KOTAN** — `kurban.kotan@cbu.edu.tr`  
Department of Artificial Intelligence and Machine Learning  
Manisa Celal Bayar University, Turkey  
ORCID: [0000-0002-6660-0565](https://orcid.org/0000-0002-6660-0565)

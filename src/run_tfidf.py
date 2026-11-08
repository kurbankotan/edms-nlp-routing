"""
run_tfidf.py
------------
Lexical baselines under the shared protocol (see common.py).

Representations
  tfidf_word_raw     word 1-2 grams on raw Turkish tokens (lower-cased, stop words removed)
  tfidf_word_stem    ... on Snowball-stemmed tokens
  tfidf_word_lemma   ... on Zeyrek-lemmatised tokens
  tfidf_char_raw     character 2-5 grams (within word boundaries) on raw tokens
  tfidf_wordchar_raw union of word and character n-grams
Translation ablation (content only, identical documents in both arms)
  tfidf_content_tr   Turkish content, raw tokens
  tfidf_content_en   machine-translated English content (Google Translate, first study)
Input ablation
  tfidf_subject_raw  subject line only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import sparse

from common import (TfidfFeatures, UnionFeatures, append_results, load_docs, load_variants, run_representation)


class SenderAugmented:
    """Appends a one-hot encoding of the dispatching (sender) unit to a sparse text representation."""
    def __init__(self, feat, docs):
        self.feat = feat
        units = sorted(docs["sender_unit"].dropna().unique())
        M = np.zeros((len(docs), len(units)), dtype=np.float32)
        for j, u in enumerate(units):
            M[:, j] = (docs["sender_unit"].values == u)
        self.M = sparse.csr_matrix(M)
        self.n_features = None

    def fit(self, idx):
        self.feat.fit(idx); self.n_features = self.feat.n_features + self.M.shape[1]; return self

    def transform(self, idx):
        return sparse.hstack([self.feat.transform(idx), self.M[idx]]).tocsr()

REPS = {
    "tfidf_word_raw": lambda v: TfidfFeatures(v["text_raw"]),
    "tfidf_word_stem": lambda v: TfidfFeatures(v["text_stem"]),
    "tfidf_word_lemma": lambda v: TfidfFeatures(v["text_lemma"]),
    "tfidf_char_raw": lambda v: TfidfFeatures(v["text_raw"], char=True),
    "tfidf_wordchar_raw": lambda v: UnionFeatures(TfidfFeatures(v["text_raw"]), TfidfFeatures(v["text_raw"], char=True)),
    "tfidf_content_tr": lambda v: TfidfFeatures(v["content_raw"].fillna("")),
    "tfidf_content_en": lambda v: TfidfFeatures(v["content_en"].fillna("")),
    "tfidf_subject_raw": lambda v: TfidfFeatures(v["text_raw"].str.split(" ").str[:0].str.join(" ")),  # placeholder, replaced below
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=list(REPS))
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--no_cv", action="store_true")
    ap.add_argument("--with_sender", action="store_true", help="append the sender-unit one-hot (context-aware variant)")
    args = ap.parse_args()
    docs, labels = load_docs()
    var = load_variants()
    # subject-only variant uses the raw tokens of the subject line
    from preprocess import variant
    var["subject_raw"] = docs["subject"].map(lambda t: variant(t, "raw"))
    REPS["tfidf_subject_raw"] = lambda v: TfidfFeatures(v["subject_raw"])
    for name in args.reps:
        feat = REPS[name](var)
        if args.with_sender:
            feat = SenderAugmented(feat, docs); name += "+sender"
        run_representation(name, feat, docs, labels, dense=False, families=args.families, cv=not args.no_cv,
                           results_file="results_tfidf.csv")


if __name__ == "__main__":
    main()

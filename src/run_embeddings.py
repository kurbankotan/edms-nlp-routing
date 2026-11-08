"""
run_embeddings.py
-----------------
Frozen sentence-embedding representations under the shared protocol.  Embeddings are
read from cache/emb_<key>__<field>.npy (see embed.py); each is evaluated with the full
classifier grid (dense=True).

    python src/run_embeddings.py                       # every cached embedding on text_clean
    python src/run_embeddings.py --keys e5_base labse
    python src/run_embeddings.py --keys e5_base --field content_clean   # input ablation
    python src/run_embeddings.py --keys e5_base --with_sender           # context-aware variant
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PrecomputedFeatures, load_docs, run_representation
from embed import CACHE, MODELS


def sender_onehot(docs):
    units = sorted(docs["sender_unit"].dropna().unique())
    M = np.zeros((len(docs), len(units)), dtype=np.float32)
    for j, u in enumerate(units):
        M[:, j] = (docs["sender_unit"].values == u)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--field", default="text_clean")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--with_sender", action="store_true")
    ap.add_argument("--no_cv", action="store_true")
    ap.add_argument("--results_file", default="results_emb.csv", help="one file per parallel job avoids write races")
    args = ap.parse_args()
    docs, labels = load_docs()
    keys = args.keys or [k for k in MODELS if (CACHE / f"emb_{k}__{args.field}.npy").exists()]
    for key in keys:
        f = CACHE / f"emb_{key}__{args.field}.npy"
        if not f.exists():
            print("missing embedding cache:", f.name, "- run embed.py first"); continue
        X = np.load(f)
        name = f"emb_{key}" + ("" if args.field == "text_clean" else f"__{args.field}")
        if args.with_sender:
            X = np.hstack([X, sender_onehot(docs)]); name += "+sender"
        meta = {"checkpoint": MODELS[key][0], "model_family": MODELS[key][1], "embedding_dim": int(X.shape[1])}
        if not np.isfinite(X).all():
            print(f"!! {name}: non-finite values in the cached embedding - skipped"); continue
        try:
            run_representation(name, PrecomputedFeatures(X), docs, labels, dense=True, families=args.families,
                               cv=not args.no_cv, extra_meta=meta, results_file=args.results_file)
        except Exception as e:
            print(f"!! FAILED {name}: {repr(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()

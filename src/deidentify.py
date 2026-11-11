"""
deidentify.py
-------------
De-identifies the text fields of data/documents.pkl in place, so that every model -- lexical, frozen and
fine-tuned encoders, the retrieval of examples for the LLM prompts and the LLMs themselves -- sees only
de-identified text.  Run it after prepare_data.py and before preprocess.py (GPU recommended).

Person names are replaced by [KİŞİ] (Turkish NER, savasy/bert-base-turkish-ner-cased) and e-mail
addresses, telephone and 11-digit identity numbers by [E-POSTA], [TELEFON] and [TCKN]:

  text       : subject and body as one text, exactly as sent to the LLMs (results/llm/masked_texts.pkl,
               computed here when it does not exist yet)
  subject    : masked separately (subject-only ablation)
  content    : masked separately (body-only and translation ablations)
  content_en : English machine translation of the body (earlier study): the names found in the Turkish
               body are replaced, and an English NER model (dslim/bert-base-NER) masks the remaining person
               names ([PERSON]).

The raw export is read only by prepare_data.py and by this script.  Neighbour lists of the LLM prompts
that were computed on the original text are kept as results/llm/neighbours__*__original_text.pkl so that
llm.py --rerun_changed can ask again exactly the prompts whose examples change.

    python src/deidentify.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm                                    # noqa: E402
from common import RESULTS, ROOT              # noqa: E402
from preprocess import clean_text             # noqa: E402

DOCS = ROOT / "data" / "documents.pkl"
PLACEHOLDERS = ["[KİŞİ]", "[PERSON]", "[E-POSTA]", "[TELEFON]", "[TCKN]"]


def archive_original_neighbours():
    """Neighbour lists computed before de-identification are renamed once (first run only)."""
    if any(llm.LLM_DIR.glob("neighbours__*__original_text.pkl")):
        return []
    moved = []
    for f in sorted(llm.LLM_DIR.glob("neighbours__*.pkl")):
        f.rename(f.with_name(f.stem + "__original_text.pkl")); moved.append(f.name)
    return moved


def main():
    docs = pd.read_pickle(DOCS)
    if "deidentified" in docs.columns and bool(docs["deidentified"].all()):
        print("data/documents.pkl is already de-identified; nothing to do")
        return
    moved = archive_original_neighbours()
    if moved:
        print("neighbour lists computed on the original text kept as *__original_text.pkl:", moved)

    # 1. subject and body as one text: identical to the text the LLMs received
    text = None
    if llm.MASKED.exists():
        m = pd.read_pickle(llm.MASKED)
        if len(m) == len(docs) and list(m.index) == list(docs.index):
            text = m
            print("combined text: re-using results/llm/masked_texts.pkl (identical to the LLM prompts)")
    if text is None:
        print("combined text: de-identifying with the Turkish NER model ...")
        text = pd.Series(llm.mask_texts(docs["text"].map(clean_text).tolist()), index=docs.index)
        llm.LLM_DIR.mkdir(parents=True, exist_ok=True)
        text.to_pickle(llm.MASKED)

    # 2. subject and body separately (input-field and translation ablations)
    print("subject lines ...")
    subject = llm.mask_texts(docs["subject"].map(clean_text).tolist())
    print("bodies ...")
    content, names = llm.mask_texts(docs["content"].map(clean_text).tolist(), return_names=True)

    # 3. English translation of the body
    en = pd.Series([np.nan] * len(docs), index=docs.index, dtype=object)
    has_en = docs["content_en"].notna().values
    if has_en.any():
        print("English translations ...")
        rows = np.where(has_en)[0]
        en.iloc[rows] = llm.mask_texts_en(docs["content_en"].iloc[rows].map(clean_text).tolist(), [names[i] for i in rows])

    docs["text"] = text.values
    docs["subject"] = subject
    docs["content"] = content
    docs["content_en"] = en.values
    docs["deidentified"] = True
    docs.to_pickle(DOCS)

    def count(series):
        s = pd.Series(series).fillna("").astype(str)
        return {p: int(s.str.count(p.replace("[", r"\[").replace("]", r"\]")).sum()) for p in PLACEHOLDERS}
    stats = {"documents": int(len(docs)), "text": count(docs["text"]), "subject": count(docs["subject"]),
             "content": count(docs["content"]), "content_en": count(docs["content_en"]),
             "documents_with_a_person_name": int(docs["text"].str.contains(r"\[KİŞİ\]").sum())}
    (RESULTS / "deidentification_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

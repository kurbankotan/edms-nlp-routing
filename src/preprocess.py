"""
preprocess.py
-------------
Turkish-native preprocessing variants used by the lexical (TF-IDF) models.

  raw   : Turkish-aware lower-casing, letters only, stop words removed
  stem  : raw + Snowball Turkish stemmer (suffix stripping)
  lemma : raw + Zeyrek morphological analysis (lemma of the most likely parse)
  en    : machine-translated English content (from the first study; content only)

Contextual encoders receive the cleaned text with original casing and punctuation
(`text_clean`), because sub-word tokenisers are trained on natural text.
"""
import html
import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "documents.pkl"
OUT = ROOT / "data" / "text_variants.pkl"

# NLTK Turkish stop-word list (53 words) embedded so the pipeline has no download step
TR_STOPWORDS = set("""acaba ama aslında az bazı belki biri birkaç birşey biz bu çok çünkü da daha de defa diye eğer en
gibi hem hep hepsi her hiç için ile ise kez ki kim mı mu mü nasıl ne neden nerde nerede nereye niçin niye o sanki
şey siz şu tüm ve veya ya yani""".split())
# institutional boiler-plate that carries no routing information (closing formulas)
TR_STOPWORDS |= {"arz", "ederim", "rica", "gereğini", "bilgilerinizi", "bilgilerinize", "hk", "hakkında"}


# placeholders of the de-identified text (src/deidentify.py) become single tokens that cannot collide
# with ordinary words (e.g. "[KİŞİ]" would otherwise become the common word "kişi")
PLACEHOLDER_TOKENS_TR = {"[KİŞİ]": " kisiadi ", "[PERSON]": " kisiadi ", "[E-POSTA]": " epostaadresi ",
                         "[TELEFON]": " telefonnumarasi ", "[TCKN]": " kimliknumarasi "}
PLACEHOLDER_TOKENS_EN = {"[PERSON]": " personname ", "[KİŞİ]": " personname ", "[E-POSTA]": " emailaddress ",
                         "[TELEFON]": " phonenumber ", "[TCKN]": " idnumber "}


def placeholders_to_tokens(s: str, table=PLACEHOLDER_TOKENS_TR) -> str:
    for k, v in table.items():
        s = s.replace(k, v)
    return s


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def clean_text(s: str) -> str:
    """Light cleaning for contextual encoders: unescape HTML entities, normalise whitespace."""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens_raw(s: str):
    s = tr_lower(clean_text(placeholders_to_tokens(str(s))))
    s = re.sub(r"[^a-zçğıöşü]+", " ", s)
    return [w for w in s.split() if w not in TR_STOPWORDS and len(w) > 1]


_stemmer = None


def stem_word(w: str) -> str:
    global _stemmer
    if _stemmer is None:
        import snowballstemmer
        _stemmer = snowballstemmer.stemmer("turkish")
    return _stemmer.stemWord(w)


_analyzer = None


@lru_cache(maxsize=None)
def lemma_word(w: str) -> str:
    """Zeyrek lemma. Zeyrek has no contextual disambiguation, so among the candidate
    lemmas of a word the longest stem is taken (longest-stem heuristic; ties -> first
    analysis). Falls back to the surface form when no analysis exists."""
    global _analyzer
    if _analyzer is None:
        import zeyrek
        logging.getLogger("zeyrek").setLevel(logging.ERROR)
        _analyzer = zeyrek.MorphAnalyzer()
    try:
        res = _analyzer.lemmatize(w)
        cands = sorted(set(res[0][1])) if res else []
        if cands:
            return tr_lower(max(cands, key=len))      # longest stem; ties -> alphabetical (deterministic)
    except Exception:
        pass
    return w


def variant(text: str, kind: str) -> str:
    toks = tokens_raw(text)
    if kind == "raw":
        return " ".join(toks)
    if kind == "stem":
        return " ".join(stem_word(w) for w in toks)
    if kind == "lemma":
        return " ".join(lemma_word(w) for w in toks)
    raise ValueError(kind)


def ensure_nltk():
    """Zeyrek needs NLTK's punkt tokenizer; download it once if missing."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab/turkish/")
        except LookupError:
            import os
            os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
            nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        print("nltk setup:", e)


def main():
    ensure_nltk()
    docs = pd.read_pickle(DOCS)
    if not ("deidentified" in docs.columns and bool(docs["deidentified"].all())):
        print("NOTE: data/documents.pkl is not de-identified; run src/deidentify.py first so that no model sees person names")
    out = pd.DataFrame({"doc_id": docs["doc_id"]})
    out["text_clean"] = docs["text"].map(clean_text)
    out["content_clean"] = docs["content"].map(clean_text)
    for kind in ["raw", "stem", "lemma"]:
        out[f"text_{kind}"] = docs["text"].map(lambda t: variant(t, kind))
        out[f"content_{kind}"] = docs["content"].map(lambda t: variant(t, kind))
        print(kind, "done; example:", out[f"text_{kind}"].iloc[2][:120])
    # English arm (content only): lower-case, letters only, English stop words removed
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    def en_tokens(s):
        if not isinstance(s, str):
            return None
        s = re.sub(r"[^a-z]+", " ", placeholders_to_tokens(s, PLACEHOLDER_TOKENS_EN).lower())
        return " ".join(w for w in s.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1)
    out["content_en"] = docs["content_en"].map(en_tokens)
    out["content_en_clean"] = docs["content_en"].map(lambda t: clean_text(t) if isinstance(t, str) else "")
    out.to_pickle(OUT)
    vocab = {k: len(set(" ".join(out[f"text_{k}"]).split())) for k in ["raw", "stem", "lemma"]}
    print("vocabulary sizes:", vocab)
    (ROOT / "results" / "preprocessing_stats.json").write_text(json.dumps(vocab, indent=2))


if __name__ == "__main__":
    # Zeyrek's candidate analyses depend on Python's string-hash seed; fix it for reproducibility
    import os
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()

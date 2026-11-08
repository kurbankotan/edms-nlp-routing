"""
llm.py
------
Large-language-model routing experiments (API based; resumable; privacy-preserving).

Modes
  zeroshot : the model receives the list of the 13 units with short duty descriptions and
             the (de-identified) document, and returns the three most likely units, ranked.
  rag<k>   : additionally receives the k most similar past documents (TF-IDF cosine over the
             training pool: `train` for validation documents, `train+val` for test documents)
             together with the units they were dispatched to (retrieval-augmented few-shot).

Privacy: before any text leaves the machine, person names are replaced with [KİŞİ] by a
Turkish NER model (savasy/bert-base-turkish-ner-cased), and e-mail addresses, phone numbers
and 11-digit identity numbers are masked by regular expressions.

Every response is appended to results/llm/<model>__<mode>__<split>.jsonl as soon as it
arrives; documents already present in that file are skipped on re-runs.

    python src/llm.py --models claude-haiku-5-5 gemini-3.8-flash --modes zeroshot rag10 --splits test val
    python src/llm.py --smoke            # one document per model, prints raw responses
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PREDS, RESULTS, append_results, bootstrap_ci, load_docs, load_variants, metrics, split_indices

LLM_DIR = RESULTS / "llm"
MASKED = RESULTS / "llm" / "masked_texts.pkl"

# provider, input price, output price (USD per 1M tokens; edit to the current list prices)
MODELS = {
    "claude-haiku-5-5": ("anthropic", 0.10, 0.50),
    "claude-sonnet-5-5": ("anthropic", 2.00, 10.00),
    "gpt-6-luna": ("openai", 0.10, 0.50),
    "gpt-6.1-terra": ("openai", 1.00, 5.00),
    "gemini-3.8-flash": ("google", 0.75, 3.75),     # standard paid tier, Oct-Dec 2026 (verify current list price)
    "gemini-3.6-flash": ("google", None, None),
}

# Duty descriptions of the 13 target units (written from the units' functions and the
# document types observed in the training data; edit freely).
UNIT_DESCRIPTIONS = {
    "Rektörlük": "Üst yönetim; Yönetim Kurulu, Senato ve Disiplin Kurulu toplantı gündemleri, üst yönetime arz edilen yazılar, rektörlük oluru gerektiren işler.",
    "Bilgi İşlem Daire Başkanlığı": "BİDB'nin daire başkanlığı makamı; daire düzeyinde vekâlet, görevlendirme, anket, mezun bilgi sistemi ve merkez müdürlüğü yazışmaları, şube müdürlüklerine dağıtılacak genel bilgi işlem talepleri.",
    "Yazılım Şube Müdürlüğü": "Yazılım geliştirme ve bilgi sistemleri: EBYS, öğrenci/personel bilgi sistemi, web sayfası güncellemeleri, kullanıcı tanımlama; atama, göreve başlama, ayrılan personel, anabilim dalı/bölüm başkanlığı değişikliği gibi sistemde güncelleme gerektiren personel yazıları.",
    "Bilgi İşlem Şube Müdürlüğü": "Donanım ve teknik destek: bilgisayar, tablet, webcam, yazıcı talepleri, laboratuvar bilgisayarları, bakım-onarım, hurda ve demirbaş işlemleri, teknik malzeme irsaliyeleri.",
    "İnternet Hizmetleri Şube Müdürlüğü": "Ağ ve internet hizmetleri: internet erişimi, e-posta ve parola, IP adresi, kablolama, web alan adı, kamera ağ bağlantısı, teknopark/uzak bağlantı talepleri.",
    "Haberleşme Hizmetleri Şube Müdürlüğü": "Telefon ve haberleşme altyapısı: dahili/dış hat talepleri, santral, telefon arızaları, faks, şehirlerarası arama açma, kamera sistemi arızaları.",
    "Sağlık Kültür ve Spor Daire Başkanlığı": "SKS'nin daire başkanlığı makamı; daire düzeyinde vekâlet, görevlendirme, olur alınması, yolluk ve izin yazıları, şube müdürlüklerine dağıtılacak genel talepler.",
    "Kültür Hizmetleri Şube Müdürlüğü": "Kültürel etkinlikler: konferans salonu tahsisi, mezuniyet ve kuruluş yıldönümü törenleri, öğrenci toplulukları, geziler, tiyatro/konser, etkinlikler için araç tahsisi.",
    "Satın Alma Şube Müdürlüğü": "Satın alma ve mali işler: kısmi zamanlı öğrenci çalıştırma kontenjanları, başvuru, puantaj ve ödeme işlemleri; cübbe/şal gibi malzeme talepleri; ihale ve satın alma yazışmaları.",
    "Sosyal Tesisler Şube Müdürlüğü": "Sosyal tesisler ve kiralama: kantin, kafeterya, çay ocağı ve restoran işletmeleri, kira bedeli ve su birim fiyatı belirleme, otomat, dış sınavlara (AUZEF vb.) mekân tahsisi.",
    "Beslenme Hizmetleri Şube Müdürlüğü": "Beslenme hizmetleri: yemekhane ve yemek hizmeti, yemek bursu, kantin ve yemekhane denetleme komisyonu, yemek ihalesi ve maliyet hesapları, personel ek gösterge yazıları.",
    "Spor Tesisleri Şube Müdürlüğü": "Spor tesisleri ve faaliyetleri: üniversite içi turnuvalar, spor temsilcileri, hakem görevlendirme ve izinleri, havuz ve spor salonu işletmesi, havuz suyu analizleri.",
    "Sağlık Hizmetleri Şube Müdürlüğü": "Sağlık hizmetleri: sağlık merkezi, Covid-19 testleri, sağlık çalışma grupları, fiili hizmet zammı gibi sağlık personeli yazıları.",
}

SYSTEM = (
    "Sen bir üniversitenin Elektronik Belge Yönetim Sistemi'nde (EBYS) gelen evrakı, gereği için sevk edilmesi gereken "
    "birime yönlendiren bir asistansın. Evrakın konusu ve içeriği verilecektir. Aşağıdaki birim listesinden, evrakın "
    "gereği için sevk edilmesi en olası üç birimi, en olasıdan başlayarak sırala. Birim adlarını listede yazıldığı "
    "biçimde, harfi harfine kullan. Yalnızca JSON döndür: {\"birimler\": [\"birim1\", \"birim2\", \"birim3\"]}\n\n"
    "BİRİMLER VE GÖREV ALANLARI:\n" + "\n".join(f"- {u}: {d}" for u, d in UNIT_DESCRIPTIONS.items())
)

def json_schema():
    """Structured-output schema: an array of unit names restricted to the 13 labels."""
    return {"type": "object",
            "properties": {"birimler": {"type": "array", "items": {"type": "string", "enum": list(UNIT_DESCRIPTIONS)}}},
            "required": ["birimler"], "additionalProperties": False}


# ----------------------------------------------------------------------------- de-identification
def _mask_patterns(t):
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[E-POSTA]", t)
    t = re.sub(r"(?<!\d)\d{11}(?!\d)", "[TCKN]", t)
    t = re.sub(r"(?<!\d)(?:\+90|0)?\s?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)", "[TELEFON]", t)
    return t


def mask_texts(texts, device=None, words_per_chunk=120):
    """Replace person names ([KİŞİ]), e-mails, phone and 11-digit numbers before API calls.
    Texts are cut into windows of `words_per_chunk` words (well below the 512-token limit of
    the NER model); masked windows are re-joined with single spaces."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    device = device if device is not None else (0 if torch.cuda.is_available() else -1)
    ckpt = "savasy/bert-base-turkish-ner-cased"
    tok = AutoTokenizer.from_pretrained(ckpt)
    tok.model_max_length = 512
    ner = pipeline("ner", model=AutoModelForTokenClassification.from_pretrained(ckpt), tokenizer=tok,
                   aggregation_strategy="simple", device=device)
    # flatten into chunks
    chunks, owner = [], []
    for n, t in enumerate(texts):
        words = str(t).split()
        if not words:
            chunks.append(""); owner.append(n); continue
        for i in range(0, len(words), words_per_chunk):
            chunks.append(" ".join(words[i:i + words_per_chunk])); owner.append(n)
    masked_chunks = []
    bs = 64
    for i in range(0, len(chunks), bs):
        batch = chunks[i:i + bs]
        nonempty = [c for c in batch if c.strip()]
        ents = iter(ner(nonempty, batch_size=32) if nonempty else [])
        for c in batch:
            if not c.strip():
                masked_chunks.append(c); continue
            es = next(ents)
            spans = sorted([(e["start"], e["end"]) for e in es if e.get("entity_group") == "PER"], reverse=True)
            for s_, e_ in spans:
                c = c[:s_] + "[KİŞİ]" + c[e_:]
            masked_chunks.append(c)
        if (i // bs) % 20 == 0:
            print(f"   masked {min(i + bs, len(chunks))}/{len(chunks)} chunks", flush=True)
    out = [[] for _ in texts]
    for n, c in zip(owner, masked_chunks):
        out[n].append(c)
    return [_mask_patterns(" ".join(parts)) for parts in out]


def get_masked(docs, var):
    if MASKED.exists():
        m = pd.read_pickle(MASKED)
        if len(m) == len(docs):
            return m
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    print("de-identifying texts with Turkish NER ...")
    m = pd.Series(mask_texts(var["text_clean"].fillna("").tolist()), index=docs.index)
    m.to_pickle(MASKED)
    return m


# ----------------------------------------------------------------------------- retrieval
def build_retriever(texts_pool):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    M = vec.fit_transform(texts_pool)
    return vec, M


def neighbours(vec, M, text, k):
    q = vec.transform([text])
    sims = (M @ q.T).toarray().ravel()
    return np.argsort(-sims)[:k], sims


# ----------------------------------------------------------------------------- providers
_clients = {}
_lock = threading.Lock()
_THINKING = {}      # Claude model -> accepted "thinking off" parameter
_CFG = {}           # model -> effective reasoning/thinking setting of the last successful call (stored per record)


def _client(provider):
    with _lock:
        if provider in _clients:
            return _clients[provider]
        if provider == "anthropic":
            import anthropic
            c = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ["CLAUDE_API_KEY"])
        elif provider == "openai":
            import openai
            c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif provider == "google":
            from google import genai
            c = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"])
        _clients[provider] = c
        return c


def model_info(model):
    """(provider, price_in, price_out); unknown models are assigned a provider by name prefix."""
    if model in MODELS:
        return MODELS[model]
    low = model.lower()
    prov = "anthropic" if low.startswith("claude") else "openai" if (low.startswith("gpt") or low.startswith("o")) else "google" if low.startswith("gemini") else None
    if prov is None:
        raise ValueError(f"unknown provider for model {model}; add it to llm.MODELS")
    return (prov, None, None)


def call_model(model, user_text, max_tokens=300):
    """Returns (raw_text, usage_in, usage_out).  Reasoning/thinking is kept minimal."""
    provider = model_info(model)[0]
    c = _client(provider)
    if provider == "anthropic":
        kw = dict(model=model, max_tokens=max_tokens, system=SYSTEM, messages=[{"role": "user", "content": user_text}])
        # "Thinking off" is spelled differently across Claude models: {"type": "disabled"} (e.g. Haiku 5.5) or
        # {"type": "between_tools"} (e.g. Sonnet 5.5, which rejects "disabled": thinking only between tool
        # calls, i.e. none for this tool-free request).  The accepted form is cached per model.  The
        # parameter is never omitted: the default of some models is adaptive thinking, which can use up
        # the answer budget (observed for Sonnet 5.5 in the first pass).
        tries = [_THINKING[model]] if model in _THINKING else [{"type": "disabled"}, {"type": "between_tools"}]
        last = None
        for th in tries:
            try:
                r = c.messages.create(**kw, thinking=th)
                _THINKING[model] = th; _CFG[model] = {"thinking": th["type"]}
                return "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text"), r.usage.input_tokens, r.usage.output_tokens
            except Exception as e:
                if "thinking" not in str(e).lower():
                    raise
                last = e
        raise last
    if provider == "openai":
        try:
            r = c.responses.create(model=model, instructions=SYSTEM, input=user_text, max_output_tokens=max_tokens,
                                   reasoning={"effort": "none"},
                                   text={"format": {"type": "json_schema", "name": "routing", "schema": json_schema(), "strict": True}})
            _CFG[model] = {"reasoning_effort": "none"}
            return (r.output_text or ""), r.usage.input_tokens, r.usage.output_tokens
        except Exception as e:
            if "reasoning" in str(e).lower() or "effort" in str(e).lower():
                r = c.responses.create(model=model, instructions=SYSTEM, input=user_text, max_output_tokens=max(max_tokens, 2048),
                                       text={"format": {"type": "json_schema", "name": "routing", "schema": json_schema(), "strict": True}})
                _CFG[model] = {"reasoning_effort": "default"}
                return (r.output_text or ""), r.usage.input_tokens, r.usage.output_tokens
            raise
    if provider == "google":
        from google.genai import types
        base = dict(system_instruction=SYSTEM, temperature=0, max_output_tokens=max_tokens, response_mime_type="application/json")
        try:
            cfg = types.GenerateContentConfig(**base, thinking_config=types.ThinkingConfig(thinking_budget=0))
            r = c.models.generate_content(model=model, contents=user_text, config=cfg)
            _CFG[model] = {"thinking_budget": 0, "temperature": 0}
        except Exception as e:
            if "thinking" not in str(e).lower():
                raise
            r = c.models.generate_content(model=model, contents=user_text, config=types.GenerateContentConfig(**base))
            _CFG[model] = {"thinking_budget": "default", "temperature": 0}
        um = r.usage_metadata
        txt = None
        try:
            txt = r.text
        except Exception:
            txt = None
        return (txt or ""), (um.prompt_token_count or 0) if um else 0, (um.candidates_token_count or 0) if um else 0
    raise ValueError(provider)


_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.translate(_ASCII).lower()).strip()


MIN_NAME_SIMILARITY = 0.85      # spelling variants ("Sağlık, Kültür ve ...") are kept; other units are not


def parse_units(raw, labels, min_sim=MIN_NAME_SIMILARITY):
    """Map the model answer to at most three unit names, in the model's order.

    1. The first complete JSON object in the answer is used (text or a repeated object after it is ignored).
    2. A truncated object keeps the complete quoted names that follow the "birimler" key.
    3. Anything else (refusal, free text, empty answer) is no answer: [].
    A name that is not literally a unit is mapped to the most similar unit (accent-insensitive
    difflib ratio) only if the similarity is >= min_sim; names of units outside the list
    (e.g. "Personel Daire Başkanlığı") are discarded."""
    import difflib
    raw = raw or ""
    cands = []
    s = raw.find("{")
    if s >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[s:])
            cands = obj.get("birimler", []) if isinstance(obj, dict) else []
        except Exception:
            k = raw.find('"birimler"', s)
            if k >= 0:
                cands = re.findall(r'"([^"]+)"', raw[k + len('"birimler"'):])
    if isinstance(cands, str):
        cands = [cands]
    out = []
    norm_labels = {l: _norm(l) for l in labels}
    for cnd in cands if isinstance(cands, list) else []:
        if not isinstance(cnd, str):
            continue
        if cnd in labels:
            u = cnd
        else:
            sims = {l: difflib.SequenceMatcher(None, _norm(cnd), norm_labels[l]).ratio() for l in labels}
            u = max(sims, key=sims.get)
            if sims[u] < min_sim:
                continue
        if u not in out:
            out.append(u)
    return out[:3]


def out_of_list(raw, labels, min_sim=MIN_NAME_SIMILARITY):
    """True if the answer names at least one unit that is not in the list (diagnostic)."""
    s = (raw or "").find("{")
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[s:]) if s >= 0 else ({}, 0)
        c = obj.get("birimler", []) if isinstance(obj, dict) else []
    except Exception:
        return False
    return any(isinstance(x, str) and x not in labels and not parse_units(json.dumps({"birimler": [x]}), labels, min_sim)
               for x in (c if isinstance(c, list) else []))


def read_records(f):
    """All records of a jsonl file in file order (a line cut by an interruption is skipped)."""
    recs = []
    if Path(f).exists():
        for l in Path(f).read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    return recs


def final_records(f):
    """Last record per document: a re-requested answer (pass 2) supersedes the first one."""
    out = {}
    for r in read_records(f):
        out[r["doc_id"]] = r
    return out


def final_answers(model, mode, split, labels):
    """doc_id -> parsed units of the final record (None if the file does not exist)."""
    f = LLM_DIR / f"{model}__{mode}__{split}.jsonl"
    if not f.exists():
        return None
    return {d: parse_units(r.get("raw", ""), labels) for d, r in final_records(f).items()}


def hidden_output_tokens(rec):
    """Output tokens not accounted for by the visible answer (Claude tokenizer: 5.4 + 0.623 tokens per
    character, fitted on Claude Haiku 5.5 answers without thinking; residual SD 1.6 tokens).  Values
    well above zero mean that the model produced hidden thinking before the answer."""
    return rec["out_tokens"] - (5.4 + 0.623 * len(rec.get("raw") or ""))


# ----------------------------------------------------------------------------- main loop
def build_user_text(subject_text, examples):
    s = ""
    if examples:
        s += "GEÇMİŞ EVRAKLAR VE SEVK EDİLDİKLERİ BİRİMLER (benzerlik sırasına göre):\n"
        for j, (t, u) in enumerate(examples, 1):
            s += f"{j}. Birim: {u}\n   Evrak: {t[:700]}\n"
        s += "\n"
    s += f"YÖNLENDİRİLECEK EVRAK:\n{subject_text[:3000]}\n\nJSON:"
    return s


# which splits each mode is run on: zero-shot only needs the test split; the retrieval-augmented
# mode is also run on validation because the cascade threshold is chosen there
PLAN = {"zeroshot": ["test"], "rag10": ["test", "val"]}


def run(models, modes, splits=None, workers=6, max_docs=None, smoke=False, max_consecutive_failures=15):
    """Returns a list of (model, mode, split) combinations that failed or were aborted."""
    docs, labels = load_docs(); var = load_variants(); idx = split_indices(docs)
    masked = get_masked(docs, var)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    all_splits = splits or sorted({s for m in modes for s in PLAN.get(m, ["test", "val"])})
    failed = []
    for split in all_splits:
        target = idx[split][:max_docs] if max_docs else idx[split]
        for mode in modes:
            if splits is None and split not in PLAN.get(mode, ["test", "val"]):
                continue
            k = int(mode[3:]) if mode.startswith("rag") else 0
            # neighbours are computed once per document (shared by all models) and cached
            nbs = _neighbours(mode, split, docs, var, idx) if k else None
            for model in models:
                f = LLM_DIR / f"{model}__{mode}__{split}.jsonl"
                done = set()
                if f.exists():
                    for l in f.read_text(encoding="utf-8").splitlines():
                        try:
                            done.add(json.loads(l)["doc_id"])
                        except Exception:
                            pass          # a line cut by an interruption is ignored and redone
                todo = [i for i in target if int(docs["doc_id"].values[i]) not in done]
                if smoke:
                    todo = todo[:1]
                print(f"[{model} | {mode} | {split}] {len(done)} done, {len(todo)} to do", flush=True)
                if not todo:
                    continue
                lock = threading.Lock()
                state = {"consecutive": 0, "abort": False, "last_error": ""}

                def work(i):
                    if state["abort"]:
                        return False
                    did = int(docs["doc_id"].values[i])
                    ex = [(masked.iloc[j], docs["label"].values[j]) for j in nbs[did]] if k else []
                    user = build_user_text(masked.iloc[i], ex)
                    for attempt in range(6):
                        if state["abort"]:
                            return False
                        try:
                            t0 = time.time(); raw, tin, tout = call_model(model, user); lat = time.time() - t0
                            rec = {"doc_id": did, "raw": raw, "units": parse_units(raw, labels), "in_tokens": tin, "out_tokens": tout,
                                   "latency_s": lat, "ts": time.time(), "cfg": _CFG.get(model)}
                            with lock:
                                with open(f, "a", encoding="utf-8") as fh:
                                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                state["consecutive"] = 0
                            if smoke:
                                print(f"--- {model} RAW: {raw[:300]}\n    parsed: {rec['units']}  true: {docs['label'].values[i]}", flush=True)
                            return True
                        except Exception as e:
                            state["last_error"] = str(e)[:300]
                            if smoke and attempt >= 1:
                                break
                            wait = 2 ** attempt
                            print(f"  retry {attempt+1} for doc {did} ({model}): {str(e)[:160]} -> sleeping {wait}s", flush=True)
                            time.sleep(wait)
                    with lock:
                        state["consecutive"] += 1
                        if state["consecutive"] >= max_consecutive_failures:
                            state["abort"] = True
                    return False

                with ThreadPoolExecutor(max_workers=workers) as ex_:
                    futs = [ex_.submit(work, i) for i in todo]
                    n_ok = 0
                    for n, fu in enumerate(as_completed(futs), 1):
                        n_ok += fu.result()
                        if n % 200 == 0:
                            print(f"   {model} | {mode} | {split}: {n}/{len(todo)} processed", flush=True)
                print(f"   finished {n_ok}/{len(todo)}", flush=True)
                if state["abort"] or n_ok < len(todo):
                    failed.append((model, mode, split))
                    msg = "ABORTED after repeated failures" if state["abort"] else f"{len(todo) - n_ok} documents failed"
                    print(f"!! [{model} | {mode} | {split}] {msg}. Last error: {state['last_error']}", flush=True)
    return failed


def _neighbours(mode, split, docs, var, idx):
    """Retrieved examples (positions in `docs`) for every document of the split; cached on disk."""
    k = int(mode[3:])
    nb_file = LLM_DIR / f"neighbours__{mode}__{split}.pkl"
    nbs = pd.read_pickle(nb_file) if nb_file.exists() else None
    if nbs is None or any(int(d) not in nbs for d in docs["doc_id"].values[idx[split]]):
        pool = idx["train"] if split == "val" else idx["trainval"]
        vec, M = build_retriever(var["text_raw"].values[pool])
        nbs = {int(docs["doc_id"].values[i]): [int(pool[j]) for j in neighbours(vec, M, var["text_raw"].values[i], k)[0]]
               for i in idx[split]}
        pd.to_pickle(nbs, nb_file)
    return nbs


def requery(models, modes, workers=6, max_attempts=3, hidden_threshold=20, max_consecutive_failures=15):
    """Second pass over finished runs.  A document is requested again when its final answer
      (i)  contains no valid unit (truncated JSON, refusal, empty answer, only units outside the list), or
      (ii) for Claude models, was preceded by hidden thinking (hidden_output_tokens > hidden_threshold);
           the first Claude Sonnet 5.5 pass ran with the model's default adaptive thinking.
    At most `max_attempts` requests per document, stopping at the first answer with a valid unit.  Every
    response is appended to the same file with "pass": 2, so the last record of a document is its final
    answer and every call is counted in the cost.  Documents that already have a pass-2 record are skipped."""
    docs, labels = load_docs(); var = load_variants(); idx = split_indices(docs)
    masked = get_masked(docs, var)
    pos = {int(d): i for i, d in enumerate(docs["doc_id"].values)}
    failed = []
    for mode in modes:
        k = int(mode[3:]) if mode.startswith("rag") else 0
        for split in PLAN.get(mode, ["test", "val"]):
            nbs = _neighbours(mode, split, docs, var, idx) if k else None
            for model in models:
                f = LLM_DIR / f"{model}__{mode}__{split}.jsonl"
                if not f.exists():
                    continue
                provider = model_info(model)[0]
                fin = final_records(f)
                missing = set(int(d) for d in docs["doc_id"].values[idx[split]]) - set(fin)
                if missing:
                    print(f"[{model} | {mode} | {split}] first pass incomplete ({len(missing)} missing): run it first", flush=True)
                    continue
                redo = []
                for d, r in fin.items():
                    if r.get("pass") == 2:
                        continue
                    if provider == "anthropic" and hidden_output_tokens(r) > hidden_threshold:
                        redo.append((d, "hidden thinking"))
                    elif not parse_units(r.get("raw", ""), labels):
                        redo.append((d, "no valid unit"))
                n_th = sum(1 for _, why in redo if why == "hidden thinking")
                print(f"[requery {model} | {mode} | {split}] {len(redo)} documents ({n_th} hidden thinking, {len(redo) - n_th} no valid unit)", flush=True)
                if not redo:
                    continue
                lock = threading.Lock()
                state = {"consecutive": 0, "abort": False, "last_error": "", "valid": 0}

                def work(item):
                    d, why = item
                    i = pos[d]
                    ex = [(masked.iloc[j], docs["label"].values[j]) for j in nbs[d]] if k else []
                    user = build_user_text(masked.iloc[i], ex)
                    for attempt in range(1, max_attempts + 1):
                        raw = None
                        for net_try in range(6):
                            if state["abort"]:
                                return False
                            try:
                                t0 = time.time(); raw, tin, tout = call_model(model, user); lat = time.time() - t0
                                break
                            except Exception as e:
                                state["last_error"] = str(e)[:300]
                                time.sleep(2 ** net_try)
                        if raw is None:
                            with lock:
                                state["consecutive"] += 1
                                if state["consecutive"] >= max_consecutive_failures:
                                    state["abort"] = True
                            return False
                        units = parse_units(raw, labels)
                        rec = {"doc_id": d, "raw": raw, "units": units, "in_tokens": tin, "out_tokens": tout, "latency_s": lat,
                               "ts": time.time(), "cfg": _CFG.get(model), "pass": 2, "attempt": attempt, "reason": why}
                        with lock:
                            with open(f, "a", encoding="utf-8") as fh:
                                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            state["consecutive"] = 0
                        if units:
                            with lock:
                                state["valid"] += 1
                            return True
                    return True          # all attempts answered without a valid unit: kept as no answer

                # one request first: a model that cannot be called without thinking (or a key problem) stops here
                n_ok = int(work(redo[0]))
                if not n_ok:
                    failed.append((model, mode, split))
                    print(f"!! [requery {model}] first request failed: {state['last_error']}", flush=True)
                    continue
                with ThreadPoolExecutor(max_workers=workers) as ex_:
                    futs = [ex_.submit(work, it) for it in redo[1:]]
                    for n, fu in enumerate(as_completed(futs), 2):
                        n_ok += fu.result()
                        if n % 100 == 0:
                            print(f"   requery {model} | {mode} | {split}: {n}/{len(redo)}", flush=True)
                print(f"   finished {n_ok}/{len(redo)}; with a valid unit: {state['valid']}", flush=True)
                if state["abort"] or n_ok < len(redo):
                    failed.append((model, mode, split))
                    print(f"!! [requery {model} | {mode} | {split}] {len(redo) - n_ok} documents not done (re-run to resume). "
                          f"Last error: {state['last_error']}", flush=True)
                if provider == "anthropic":
                    p2 = [r for r in read_records(f) if r.get("pass") == 2]
                    print(f"   check: pass-2 answers with hidden thinking: {sum(hidden_output_tokens(r) > hidden_threshold for r in p2)}/{len(p2)}; "
                          f"setting: {sorted({json.dumps(r.get('cfg')) for r in p2})}", flush=True)
    return failed


DECODING = {   # reported in the params column
    "anthropic": {"thinking": "off (disabled; between_tools for models that reject disabled)", "temperature": "provider default", "max_tokens": 300},
    "openai": {"reasoning_effort": "none", "structured_output": "strict JSON schema, enum of units", "temperature": "provider default", "max_tokens": 300},
    "google": {"thinking_budget": 0, "temperature": 0, "response_mime_type": "application/json", "max_tokens": 300},
}


def summarise(models, modes, split="test"):
    """Metrics + prediction files in the common format for every finished (model, mode).  Answers are
    re-parsed from the raw text of the final record of each document (see parse_units)."""
    docs, labels = load_docs(); idx = split_indices(docs)
    fallback = docs["label"].values[idx["trainval"]]
    fallback = pd.Series(fallback).value_counts().index[0]      # majority unit of the training data
    test_ids = docs["doc_id"].values[idx[split]]
    y_true = docs["label"].values[idx[split]]; label_sets = docs["label_set"].values[idx[split]]
    rows = []
    for model in models:
        for mode in modes:
            f = LLM_DIR / f"{model}__{mode}__{split}.jsonl"
            if not f.exists():
                continue
            recs = final_records(f)
            if len(set(test_ids) - set(recs)) > 0:
                print(f"[{model} | {mode}] incomplete: {len(set(test_ids) - set(recs))} missing"); continue
            prov, pin, pout = model_info(model)
            scores = np.zeros((len(test_ids), len(labels)))
            y_pred = []
            parsed = {did: parse_units(recs[did].get("raw", ""), labels) for did in test_ids}
            n_empty = sum(1 for did in test_ids if not parsed[did])
            for n, did in enumerate(test_ids):
                units = parsed[did] or [fallback]
                for rank, u in enumerate(units):
                    scores[n, labels.index(u)] = 3 - rank
                y_pred.append(units[0])
            y_pred = np.array(y_pred)
            m = metrics(y_true, y_pred, labels, scores=scores, label_sets=label_sets)
            ci = bootstrap_ci(y_true, y_pred, labels)
            fin = [recs[did] for did in test_ids]
            alls = [r for r in read_records(f) if r["doc_id"] in recs]
            tin = sum(r["in_tokens"] for r in fin); tout = sum(r["out_tokens"] for r in fin)
            tin_all = sum(r["in_tokens"] for r in alls); tout_all = sum(r["out_tokens"] for r in alls)
            cost = (tin * pin + tout * pout) / 1e6 if pin is not None else None
            cost_all = (tin_all * pin + tout_all * pout) / 1e6 if pin is not None else None
            lat = float(np.median([r["latency_s"] for r in fin]))
            name = f"llm_{model}"
            row = {"representation": name, "family": mode, "params": json.dumps({"mode": mode, **DECODING[prov]}), **m, **ci,
                   "fit_seconds_trainval": 0.0, "predict_ms_per_doc": lat * 1000, "n_features": 0, "model_family": "LLM (" + prov + ")",
                   "in_tokens": tin, "out_tokens": tout, "usd_cost": cost, "usd_per_1000_docs": (cost / len(test_ids) * 1000) if cost else None,
                   "usd_cost_all_calls": cost_all, "n_calls": len(alls),
                   "n_empty_answers": n_empty, "n_requeried": sum(1 for r in fin if r.get("pass") == 2),
                   "n_out_of_list": sum(1 for r in fin if out_of_list(r.get("raw", ""), labels)),
                   "n_hidden_thinking": sum(1 for r in fin if hidden_output_tokens(r) > 20) if prov == "anthropic" else 0}
            pd.to_pickle({"doc_id": test_ids, "y_true": y_true, "y_pred": y_pred, "scores": scores, "labels": labels,
                          "params": row["params"], "family": mode, "representation": name}, PREDS / f"{name}__{mode}.pkl")
            rows.append(row)
            print(f"[{model} | {mode}] acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} top3={m['top3_accuracy']:.4f} "
                  f"empty={n_empty} requeried={row['n_requeried']} hidden_thinking={row['n_hidden_thinking']} "
                  f"cost=${cost if cost is None else round(cost, 3)} (all calls ${cost_all if cost_all is None else round(cost_all, 3)})")
    append_results(rows, file="results_llm.csv")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["claude-haiku-5-5", "claude-sonnet-5-5", "gpt-6-luna", "gemini-3.8-flash"])
    ap.add_argument("--modes", nargs="*", default=["zeroshot", "rag10"])
    ap.add_argument("--splits", nargs="*", default=None, help="default: zeroshot -> test; rag10 -> test and val")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--requery", action="store_true", help="second pass: re-request answers without a valid unit and "
                    "Claude answers produced with hidden thinking (see requery)")
    ap.add_argument("--mask_only", action="store_true", help="only build the de-identified texts (GPU recommended)")
    a = ap.parse_args()
    if a.mask_only:
        d_, _ = load_docs(); get_masked(d_, load_variants()); print("masked texts ready:", MASKED)
    elif a.summarise:
        summarise(a.models, a.modes)
    elif a.requery:
        bad = requery(a.models, a.modes, a.workers)
        if bad:
            print("\nREQUERY not finished for:", bad, "- run the same command again to resume")
            sys.exit(1)
    else:
        failed = run(a.models, a.modes, a.splits, a.workers, a.max_docs, a.smoke)
        if a.smoke:
            bad = sorted({m for m, _, _ in failed})
            status = {m: ("failed" if m in bad else "ok") for m in a.models}
            (LLM_DIR / "smoke_status.json").write_text(json.dumps(status, indent=2))
            print("\nSMOKE TEST:", status)
            if bad:
                print("Failed models are skipped; check the model name (LLM_MODELS / src/llm.py MODELS), the API key and the quota.")
            if len(bad) == len(a.models):
                sys.exit(1)

"""Builds EBYS_Routing_Colab.ipynb from the src/ modules (single source of truth)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_FILES = ["prepare_data.py", "preprocess.py", "common.py", "embed.py", "run_tfidf.py", "run_embeddings.py", "run_emb_parallel.py",
             "finetune.py", "llm.py", "analysis.py", "routing.py"]
PAPER_FILES = ["extra_stats.py"]          # statistics quoted in the manuscript (imports src/ modules)
BASE = "/content/drive/MyDrive/ebys-routing"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# EBYS belge yönlendirme deneyleri (Colab) — v5

v5: gömme sınıflandırıcı ızgarası 4 paralel işte (yoğun gömmelerde doğrusal SVM yok); BDM yanıtları için ikinci geçiş (yeniden sorgu).
v4: fine-tuning T4'te fp16 + dinamik dolgu ile ~10 kat hızlı; gömme sınıflandırıcıları fine-tuning ile aynı anda arka planda koşar.
Claude çağrıları ek defterde (EBYS_LLM_ek.ipynb) yapılır; bu defter Gemini ve GPT'yi çalıştırır.

**Nasıl çalıştırılır:** Çalışma zamanı → Çalışma zamanı türünü değiştir → **GPU** (A100 > L4 > T4) ve varsa **Yüksek RAM**.
Sonra **Çalışma zamanı → Tümünü çalıştır**. Drive ve Secrets için çıkan izin pencerelerini onaylayın.

* Ham veri: Drive'da `Drive'ım/ebys-routing/data/` klasörüne `Rapor-Orjinal-HamIlkHali-18102024-v3.xlsx`
  (isteğe bağlı: `Rapor-Onislenmis.xlsx`). Dosya adını değiştirmeniz gerekmez.
* API anahtarları: Colab Secrets (sol menüde anahtar simgesi); "Not defteri erişimi" açık olmalı.
* Her ara sonuç Drive'a yazılır. Bağlantı koparsa yeniden bağlanıp **Tümünü çalıştır** deyin; biten işler atlanır.
* Sonunda `ebys_results_<tarih>.zip` otomatik iner (ayrıca Drive'da `ebys-routing/` klasörüne kopyalanır).

Akış: veri hazırlama → [arka planda: TF-IDF (CPU) ve BDM API çağrıları] → gömme çıkarımı (GPU) → fine-tuning (GPU)
→ gömme sınıflandırıcıları (CPU) → arka plan işlerini bekleme → analiz → zip + indirme.""")

code(f"""# === 1. Drive bağlantısı, klasörler ve veri dosyalarının bulunması ===
from google.colab import drive
drive.mount('/content/drive')
import os, sys, subprocess, json, time, shutil, glob
BASE = "{BASE}"
for d in ["data", "src", "paper", "results", "figures", "cache", "logs"]:
    os.makedirs(f"{{BASE}}/{{d}}", exist_ok=True)
os.chdir(BASE)

import pandas as pd

def _candidates(patterns):
    out = []
    for folder in [f"{{BASE}}/data", BASE, "/content/drive/MyDrive"]:
        for pat in patterns:
            for h in sorted(glob.glob(os.path.join(folder, pat))):
                if h not in out:
                    out.append(h)
    return out

def _cols(path):
    try:
        return set(pd.read_excel(path, nrows=0).columns)
    except Exception:
        return set()

RAW = f"{{BASE}}/data/raw_dataset.xlsx"
if not os.path.exists(RAW):
    need = {{"EvrakId", "EvrakKonusu", "EvrakIcerigi", "EvrakinGeregineKisiKullaniciBirimi", "EvrakinSevkedilmeTarihi"}}
    for c in _candidates(["Rapor-Orjinal-HamIlkHali-18102024-v3.xlsx", "*HamIlkHali*v3.xlsx", "*HamIlkHali*.xlsx"]):
        cols = _cols(c)
        if need <= cols and not ({{"English", "MessageOnislenmis"}} & cols):     # the raw export, not a derived file
            shutil.copy(c, RAW); print("Ham veri bulundu:", c, "-> data/raw_dataset.xlsx"); break
V1 = f"{{BASE}}/data/preprocessed_dataset_v1.xlsx"
if not os.path.exists(V1):
    for c in _candidates(["Rapor-Onislenmis.xlsx", "*Onislenmis*.xlsx", "*Önişlenmiş*.xlsx"]):
        if {{"EvrakId", "English"}} <= _cols(c):
            shutil.copy(c, V1); print("İngilizce çeviri sütunlu dosya bulundu:", c, "-> data/preprocessed_dataset_v1.xlsx"); break
if not os.path.exists(RAW):
    raise FileNotFoundError("Ham veri bulunamadı. Rapor-Orjinal-HamIlkHali-18102024-v3.xlsx dosyasını Drive'da "
                            "'Drive'ım/ebys-routing/data' klasörüne yükleyip bu hücreyi yeniden çalıştırın.")
print("Çalışma klasörü:", BASE)
print("data/:", os.listdir(f"{{BASE}}/data"))
if not os.path.exists(V1):
    print("Not: Rapor-Onislenmis.xlsx yok; yalnızca çeviri ablasyonu atlanacak, diğer her şey çalışır.")""")

code("""# === 2. Yapılandırma, donanım kontrolü ve API anahtarları ===
RUN_TFIDF, RUN_EMBED, RUN_FINETUNE, RUN_LLM, RUN_ANALYSIS = True, True, True, True, True

EMBED_MODELS    = ["minilm_multi", "distiluse_multi", "sbert_tr", "berturk_mean", "mpnet_multi", "e5_base", "labse",
                   "e5_large", "bge_m3", "qwen3_0_6b", "qwen3_4b"]
FINETUNE_MODELS = ["dbmdz/bert-base-turkish-cased", "xlm-roberta-base"]
LLM_MODELS      = ["gemini-3.8-flash", "gpt-6-luna"]     # Claude ek defterde (EBYS_LLM_ek.ipynb) çalıştırılıyor
SUMMARY_MODELS  = ["claude-haiku-5-5", "claude-sonnet-5-5", "gpt-6-luna", "gemini-3.8-flash"]   # özetlenecek tüm BDM'ler
LLM_MODES       = ["zeroshot", "rag10"]   # zeroshot -> yalnız test; rag10 -> test + doğrulama (kademe eşiği için)
LLM_WORKERS     = 6                       # sağlayıcı başına eşzamanlı istek; 429 hataları görürseniz 3'e indirin

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU bulunamadı. Çalışma zamanı -> Çalışma zamanı türünü değiştir -> GPU (A100/L4/T4) seçip "
                       "Tümünü çalıştır'a yeniden basın.")
GPU = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
print(f"GPU: {GPU} ({GPU_GB:.0f} GB) | RAM: {RAM_GB:.0f} GB | CPU çekirdeği: {os.cpu_count()}")
PARALLEL = RAM_GB >= 20     # yeterli RAM varsa TF-IDF, GPU işleriyle aynı anda arka planda koşar
if not PARALLEL:
    print("RAM 20 GB'ın altında: TF-IDF arka planda değil, GPU işlerinden sonra sırayla koşacak (daha uzun sürer). "
          "Mümkünse 'Yüksek RAM'i açın.")

from google.colab import userdata
CANDIDATES = {
    "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY", "CLAUDE_KEY", "CLAUDE", "ANTHROPIC"],
    "OPENAI_API_KEY":    ["OPENAI_API_KEY", "OPENAI_KEY", "CHATGPT_API_KEY", "GPT_API_KEY", "OPENAI", "CHATGPT"],
    "GOOGLE_API_KEY":    ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_GENAI_API_KEY", "GEMINI", "GOOGLE_AI_KEY"],
}
found = {}
for env_name, names in CANDIDATES.items():
    for n in names:
        try:
            v = userdata.get(n)
        except Exception:
            v = None
        if v:
            os.environ[env_name] = v.strip(); found[env_name] = n; break
print("Bulunan anahtarlar:", found if found else "YOK")
def _provider(m):
    m = m.lower()
    return "ANTHROPIC_API_KEY" if m.startswith("claude") else "OPENAI_API_KEY" if m.startswith(("gpt", "o")) else "GOOGLE_API_KEY"
missing = [m for m in LLM_MODELS if _provider(m) not in found]
if missing:
    print("UYARI: anahtarı bulunamayan modeller atlanacak:", missing,
          "- Secrets'taki adlar şunlardan biri olmalı:", {k: v[:2] for k, v in CANDIDATES.items()})
LLM_MODELS = [m for m in LLM_MODELS if _provider(m) in found]
print("Çalıştırılacak BDM'ler:", LLM_MODELS)""")

code("""# === 3. Paketler (Colab'ın hazır numpy/pandas/torch sürümlerine dokunulmaz) ===
%pip install -q zeyrek snowballstemmer
%pip install -q anthropic openai google-genai
%pip install -q sentence-transformers statsmodels openpyxl
import nltk; nltk.download("punkt_tab", quiet=True)
!apt-get -qq install -y graphviz > /dev/null 2>&1 || true
import platform, datetime, importlib.metadata as im
def _v(p):
    try: return im.version(p)
    except Exception: return None
env = {"python": platform.python_version(), "sklearn": _v("scikit-learn"), "torch": _v("torch"), "transformers": _v("transformers"),
       "sentence_transformers": _v("sentence-transformers"), "zeyrek": _v("zeyrek"), "anthropic": _v("anthropic"), "openai": _v("openai"),
       "google_genai": _v("google-genai"), "gpu": GPU, "gpu_gb": round(GPU_GB), "ram_gb": round(RAM_GB), "cpu_count": os.cpu_count(),
       "date": datetime.date.today().strftime("%B %Y")}
json.dump(env, open(f"{BASE}/results/environment.json", "w"), indent=2); print(env)""")

md("## Kaynak kodu (src/)\nAşağıdaki hücreler pipeline modüllerini Drive'daki `src/` klasörüne yazar (yerel çalışmayla birebir aynı kod).")
for f in SRC_FILES:
    src = (ROOT / "src" / f).read_text()
    code(f"%%writefile {BASE}/src/{f}\n" + src)
for f in PAPER_FILES:
    code(f"%%writefile {BASE}/paper/{f}\n" + (ROOT / "paper" / f).read_text())

code("""# === 4. Yardımcılar: ön planda ve arka planda komut çalıştırma ===
FAILURES = []
BG = {}
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}

def _quiet(line):
    l = line.lower()
    return ("warning" in l or "warn(" in l) and "retry" not in l

def run(cmd, log_name, check=True):
    print("\\n$", cmd, flush=True)
    with open(f"{BASE}/logs/{log_name}.log", "a") as log:
        log.write(f"\\n===== {time.ctime()} :: {cmd}\\n")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=BASE, env=ENV)
        for line in p.stdout:
            log.write(line)
            if not _quiet(line):
                print(line, end="", flush=True)
        p.wait()
    if p.returncode != 0:
        FAILURES.append(cmd)
        if check:
            raise RuntimeError(f"Komut başarısız ({p.returncode}): {cmd}")
        print(f"!! Komut başarısız ({p.returncode}), devam ediliyor: {cmd}")
        return False
    return True

def start_bg(name, cmd):
    if name in BG and BG[name][0].poll() is None:
        print(f"[arka plan] {name} zaten çalışıyor"); return
    log = open(f"{BASE}/logs/{name}.log", "a")
    log.write(f"\\n===== {time.ctime()} :: {cmd}\\n"); log.flush()
    p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, cwd=BASE, env=ENV)
    BG[name] = (p, log, cmd)
    print(f"[arka plan] {name} başlatıldı -> logs/{name}.log")

def _last_line(name):
    try:
        lines = [l for l in open(f"{BASE}/logs/{name}.log", errors="ignore").read().splitlines() if l.strip() and not _quiet(l)]
        return lines[-1][:150] if lines else ""
    except Exception:
        return ""

def bg_status():
    for n, (p, _, _) in BG.items():
        st = "çalışıyor" if p.poll() is None else f"bitti (kod {p.returncode})"
        print(f"  {n:16s} {st:16s} {_last_line(n)}")

def wait_bg(every=120):
    while any(p.poll() is None for p, _, _ in BG.values()):
        print(time.strftime("%H:%M:%S"), "arka plan işleri:"); bg_status(); time.sleep(every)
    print(time.strftime("%H:%M:%S"), "tüm arka plan işleri bitti:"); bg_status()
    for n, (p, log, cmd) in BG.items():
        log.close()
        if p.returncode != 0:
            FAILURES.append(cmd)""")

code("""# === 5. Veri hazırlama ve Türkçe önişleme (~3 dk) ===
if not os.path.exists(f"{BASE}/data/documents.pkl"):
    run("python src/prepare_data.py", "prepare")
if not os.path.exists(f"{BASE}/data/text_variants.pkl"):
    run("python src/preprocess.py", "prepare")
st = json.load(open(f"{BASE}/results/dataset_stats.json"))
print({k: st[k] for k in ["n_dispatch_records_raw", "n_documents", "n_classes", "split_sizes"]})
ref = {"n_documents": 9345, "split_sizes": {"train": 5717, "val": 1975, "test": 1653}}
if st["n_documents"] == ref["n_documents"] and st["split_sizes"] == ref["split_sizes"]:
    print("OK: referans koşuyla birebir aynı veri seti ve bölmeler.")
else:
    print("UYARI: belge sayıları referans koşudan farklı:", ref)""")

code("""# === 6. TF-IDF taban çizgileri ve ablasyonlar (CPU, arka planda; ~1-1,5 saat) ===
TFIDF_CMD = ("python src/run_tfidf.py && "
             "python src/run_tfidf.py --reps tfidf_word_lemma tfidf_wordchar_raw --with_sender")
if RUN_TFIDF and PARALLEL:
    start_bg("tfidf", TFIDF_CMD)""")

code("""# === 7. BDM: kimliksizleştirme (GPU) + duman testi, ardından sağlayıcı başına arka planda tam koşu ===
LLM_OK = False
if RUN_LLM and LLM_MODELS:
    LLM_OK = run("python src/llm.py --smoke --models " + " ".join(LLM_MODELS) + " --modes " + " ".join(LLM_MODES), "llm", check=False)
    if LLM_OK:
        status = json.load(open(f"{BASE}/results/llm/smoke_status.json"))
        LLM_MODELS = [m for m in LLM_MODELS if status.get(m) == "ok"]
        print("Duman testini geçen BDM'ler:", LLM_MODELS)
        groups = {}
        for m in LLM_MODELS:
            groups.setdefault(_provider(m), []).append(m)
        for prov, models in groups.items():
            one = f"python src/llm.py --workers {LLM_WORKERS} --models " + " ".join(models) + " --modes " + " ".join(LLM_MODES)
            start_bg("llm_" + prov.split("_")[0].lower(), f"{one} ; {one}")   # ikinci geçiş, başarısız belgeleri yeniden dener
    else:
        print("\\n!! BDM DUMAN TESTİ BAŞARISIZ. Yukarıdaki hata mesajına bakın (model adı / anahtar / kota). "
              "Diğer deneyler devam ediyor; düzelttikten sonra yalnızca 2., 4. ve 7. hücreleri yeniden çalıştırabilirsiniz.")""")

code("""# === 8. Gömme çıkarımı (GPU; ilk seferde ~20-40 dk, sonra önbellekten) + sınıflandırıcı ızgarası arka planda ===
GRID_CMD = "python src/run_emb_parallel.py --workers 4"   # her temsil ayrı iş ve ayrı sonuç dosyası (results_emb__*.csv)
if RUN_EMBED:
    run("python src/embed.py --models " + " ".join(EMBED_MODELS), "embed", check=False)
    run("python src/embed.py --models mpnet_multi --field content_clean", "embed", check=False)
    if os.path.exists(V1):
        run("python src/embed.py --models mpnet_en --field content_en_clean", "embed", check=False)
    if PARALLEL:
        start_bg("embed_grid", GRID_CMD)      # CPU işi; fine-tuning (GPU) ile aynı anda koşar""")

code("""# === 9. Uçtan uca fine-tuning: BERTurk ve XLM-R (GPU; T4 ~1-1,5 saat, L4/A100 daha kısa) ===
if RUN_FINETUNE:
    run("python src/finetune.py --cv --models " + " ".join(FINETUNE_MODELS), "finetune", check=False)
if BG: bg_status()""")

code("""# === 10. RAM azsa: gömme sınıflandırıcıları ve TF-IDF burada sırayla (yeterliyse zaten arka planda) ===
if RUN_EMBED and not PARALLEL:
    run(GRID_CMD, "embed_grid", check=False)
if RUN_TFIDF and not PARALLEL:
    run(TFIDF_CMD, "tfidf", check=False)
if BG: bg_status()""")

code("""# === 11. Arka plan işlerini bekle (TF-IDF, BDM) ve BDM sonuçlarını özetle ===
if BG:
    wait_bg()
if RUN_LLM:
    # ikinci geçiş: geçerli birim içermeyen yanıtlar ve gizli düşünmeyle üretilmiş Claude yanıtları yeniden sorulur
    run("python src/llm.py --requery --models " + " ".join(LLM_MODELS) + " --modes " + " ".join(LLM_MODES), "llm", check=False)
    run("python src/llm.py --summarise --models " + " ".join(SUMMARY_MODELS) + " --modes " + " ".join(LLM_MODES), "llm", check=False)""")

code("""# === 12. Analiz: tablolar (results/tables.xlsx) ve şekiller (figures/) ===
if RUN_ANALYSIS:
    run("python src/analysis.py", "analysis", check=False)
    run("python src/routing.py", "analysis", check=False)     # kalibrasyon ve hiyerarşik geri çekilme (tablolar + fig_routing)
    run("python paper/extra_stats.py", "analysis", check=False)   # sıra uyumu, kNN oylaması, ek McNemar testleri, kaskad maliyeti
    from IPython.display import Image, display
    for f in ["fig_main.png", "fig_deferral.png", "fig_routing.png"]:
        p = f"{BASE}/figures/{f}"
        if os.path.exists(p): display(Image(p, width=900))""")

code("""# === 13. Paketle ve indir (results + figures + logs + src; gömmeler ve kimliksizleştirilmiş metinler hariç) ===
import zipfile
stamp = time.strftime("%Y%m%d-%H%M")
out = f"/content/ebys_results_{stamp}.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ["results", "figures", "logs", "src"]:
        for root, _, files in os.walk(f"{BASE}/{sub}"):
            for fn in files:
                if fn.endswith(".npy") or "masked_texts" in fn:
                    continue
                full = os.path.join(root, fn); z.write(full, os.path.relpath(full, BASE))
print("Zip:", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
shutil.copy(out, f"{BASE}/ebys_results_{stamp}.zip")
print("Drive kopyası:", f"{BASE}/ebys_results_{stamp}.zip")
if FAILURES:
    print("\\nUYARI - başarısız komutlar (zip yine de oluşturuldu, loglar içinde):")
    for c in FAILURES: print("  -", c)
else:
    print("\\nTüm adımlar başarıyla tamamlandı.")
from google.colab import files
files.download(out)""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "EBYS_Routing_Colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("written", out, "cells:", len(cells))

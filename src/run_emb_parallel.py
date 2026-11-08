"""
run_emb_parallel.py
-------------------
Runs the frozen-embedding classifier grid (run_embeddings.py) as several parallel jobs, one
representation per job, each writing its own results file (results_emb__<job>.csv) so that
parallel processes never write the same file.  Most scikit-learn fits here are single-threaded,
so N jobs with 2 BLAS threads each use the CPU far better than one sequential process.
Re-running resumes: finished representations are skipped by run_representation.

    python src/run_emb_parallel.py --workers 4
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# longest first (embedding dimension), so that the large models do not finish last
JOBS = [
    ("qwen3_4b", "--keys qwen3_4b"),
    ("e5_large", "--keys e5_large"),
    ("bge_m3", "--keys bge_m3"),
    ("qwen3_0_6b", "--keys qwen3_0_6b"),
    ("sbert_tr", "--keys sbert_tr"),
    ("berturk_mean", "--keys berturk_mean"),
    ("mpnet_multi", "--keys mpnet_multi"),
    ("e5_base", "--keys e5_base"),
    ("labse", "--keys labse"),
    ("distiluse_multi", "--keys distiluse_multi"),
    ("minilm_multi", "--keys minilm_multi"),
    ("mpnet_multi__content_clean", "--keys mpnet_multi --field content_clean"),
    ("mpnet_en__content_en_clean", "--keys mpnet_en --field content_en_clean"),
    ("e5_base+sender", "--keys e5_base --with_sender"),
    ("mpnet_multi+sender", "--keys mpnet_multi --with_sender"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="BLAS / joblib threads per job")
    ap.add_argument("--only", nargs="*", default=None, help="run only these jobs (names as in JOBS)")
    a = ap.parse_args()
    jobs = [j for j in JOBS if a.only is None or j[0] in a.only]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads),
           "MKL_NUM_THREADS": str(a.threads), "LOKY_MAX_CPU_COUNT": str(a.threads)}
    (ROOT / "logs").mkdir(exist_ok=True)
    t_start = time.time()
    status = {}

    def job(item):
        name, args = item
        t0 = time.time()
        print(time.strftime("%H:%M"), f"başladı  {name}", flush=True)
        cmd = f"{sys.executable} src/run_embeddings.py {args} --results_file results_emb__{name}.csv"
        with open(ROOT / "logs" / f"emb__{name}.log", "a") as log:
            log.write(f"\n===== {time.ctime()} :: {cmd}\n"); log.flush()
            rc = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        status[name] = rc
        done = sum(1 for v in status.values() if v == 0)
        print(time.strftime("%H:%M"), f"bitti    {name} ({(time.time() - t0) / 60:.0f} dk, kod {rc}) - "
              f"{done}/{len(jobs)} tamam, toplam {(time.time() - t_start) / 60:.0f} dk", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(job, jobs))
    bad = [n for n, rc in status.items() if rc != 0]
    print(time.strftime("%H:%M"), "TÜM GÖMME İŞLERİ BİTTİ" + (f" - hata veren: {bad} (logs/emb__<ad>.log)" if bad else ""), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

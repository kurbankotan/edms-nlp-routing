"""
prepare_data.py
---------------
Builds the document-level routing dataset from the raw EDMS dispatch log and
defines the evaluation protocol (chronological split + stratified group folds).

Input : data/raw_dataset.xlsx            (one row per dispatch record)
        data/preprocessed_dataset_v1.xlsx (optional; supplies the machine-translated
                                           English content from the first study)
Output: data/documents.pkl               (one row per document)
        results/dataset_stats.json       (all counts reported in the paper)

Dataset construction (every step is counted and written to dataset_stats.json):
  1. Keep dispatch records that have a target unit (EvrakinGeregineKisiKullaniciBirimi).
  2. Text = subject + ". " + content.  Drop records whose text is empty.
  3. Broadcast filter: identical texts dispatched to more than BROADCAST_MAX distinct
     units are circulars/announcements, not routing decisions -> removed.
  4. One row per document (EvrakId). The label is the target unit of the FIRST dispatch
     (earliest EvrakinSevkedilmeTarihi).  The set of all units that received the
     document for action is kept as `label_set` for lenient evaluation.
  5. Near-identical documents are grouped by a normalised text key (`group_id`) so
     that identical texts never fall on both sides of a split.

Splits:
  * chronological (deployment-realistic): train = documents dated <= 2021-12-31,
    val = 2022-01-01 .. 2022-06-30, test = 2022-07-01 .. 2022-12-31.
  * 5-fold StratifiedGroupKFold (two repeats with different seeds) over all documents,
    groups = group_id, for variance estimates.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_dataset.xlsx"
PRE_V1 = ROOT / "data" / "preprocessed_dataset_v1.xlsx"
OUT = ROOT / "data" / "documents.pkl"
STATS = ROOT / "results" / "dataset_stats.json"

LABEL = "EvrakinGeregineKisiKullaniciBirimi"
SENDER = "EvrakiSevkedenKullaniciBirimi"
BROADCAST_MAX = 3          # identical text dispatched to > 3 distinct units = broadcast
VAL_START, TEST_START, TEST_END = "2022-01-01", "2022-07-01", "2023-01-01"
SEEDS = (0, 1)             # repeats of the 5-fold grouped CV

# organisational hierarchy (parent department of each unit)
PARENT = {
    "Bilgi İşlem Daire Başkanlığı": "BİDB",
    "Yazılım Şube Müdürlüğü": "BİDB",
    "Bilgi İşlem Şube Müdürlüğü": "BİDB",
    "İnternet Hizmetleri Şube Müdürlüğü": "BİDB",
    "Haberleşme Hizmetleri Şube Müdürlüğü": "BİDB",
    "Sağlık Kültür ve Spor Daire Başkanlığı": "SKS",
    "Kültür Hizmetleri Şube Müdürlüğü": "SKS",
    "Satın Alma Şube Müdürlüğü": "SKS",
    "Sosyal Tesisler Şube Müdürlüğü": "SKS",
    "Beslenme Hizmetleri Şube Müdürlüğü": "SKS",
    "Spor Tesisleri Şube Müdürlüğü": "SKS",
    "Sağlık Hizmetleri Şube Müdürlüğü": "SKS",
    "Rektörlük": "Rektörlük",
}


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def norm_key(s: str) -> str:
    """Normalised key used to group identical/near-identical texts."""
    s = tr_lower(str(s))
    s = re.sub(r"[^a-zçğıöşü]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    stats = {}
    df = pd.read_excel(RAW)
    stats["n_dispatch_records_raw"] = int(len(df))
    stats["n_documents_raw"] = int(df["EvrakId"].nunique())
    stats["date_min_raw"] = str(df["EvrakTarihi"].min().date())
    stats["date_max_raw"] = str(df["EvrakTarihi"].max().date())

    # 1. target unit present
    df = df[df[LABEL].notna()].copy()
    stats["n_records_with_target_unit"] = int(len(df))

    # 2. text
    df["subject"] = df["EvrakKonusu"].fillna("").astype(str).str.strip()
    df["content"] = df["EvrakIcerigi"].fillna("").astype(str).str.strip()
    df["text"] = (df["subject"] + ". " + df["content"]).str.strip(" .")
    df = df[df["text"].str.len() > 0].copy()
    stats["n_records_nonempty_text"] = int(len(df))
    stats["n_records_empty_content"] = int((df["content"] == "").sum())

    # 3. broadcast filter
    units_per_text = df.groupby("text")[LABEL].nunique()
    broadcast_texts = units_per_text[units_per_text > BROADCAST_MAX].index
    is_bc = df["text"].isin(broadcast_texts)
    stats["broadcast_threshold_units"] = BROADCAST_MAX
    stats["n_broadcast_texts"] = int(len(broadcast_texts))
    stats["n_broadcast_records_removed"] = int(is_bc.sum())
    stats["n_documents_with_target_unit"] = int(df["EvrakId"].nunique())
    bc_docs = df.loc[is_bc].drop_duplicates("EvrakId")
    stats["n_broadcast_documents_removed"] = int(df["EvrakId"].nunique() - df.loc[~is_bc, "EvrakId"].nunique())
    stats["n_broadcast_documents_empty_body"] = int((bc_docs["content"] == "").sum())
    stats["n_broadcast_documents_bare_petition"] = int(((bc_docs["content"] == "") & (bc_docs["subject"].str.upper() == "DİLEKÇE")).sum())
    stats["n_broadcast_documents_with_body"] = int((bc_docs["content"] != "").sum())
    stats["broadcast_units_per_text_median"] = float(units_per_text[broadcast_texts].median())
    df = df[~is_bc].copy()
    stats["n_records_after_broadcast_filter"] = int(len(df))

    # 4. one row per document: first dispatch defines the label
    df["dispatch_time"] = pd.to_datetime(df["EvrakinSevkedilmeTarihi"], errors="coerce", format="mixed")
    df = df.sort_values(["EvrakId", "dispatch_time"], kind="stable")
    g = df.groupby("EvrakId", sort=False)
    docs = g.first()[["EvrakTarihi", "subject", "content", "text", SENDER, LABEL, "dispatch_time",
                      "EvrakiSevkedenKullaniciUnvani", "EvrakKategorisiAdi"]].copy()
    docs = docs.rename(columns={"EvrakTarihi": "date", SENDER: "sender_unit", LABEL: "label",
                                "EvrakiSevkedenKullaniciUnvani": "sender_title",
                                "EvrakKategorisiAdi": "category"})
    docs["label_set"] = g[LABEL].agg(lambda s: tuple(dict.fromkeys(s)))   # ordered unique
    docs["n_dispatches"] = g.size()
    docs["n_units"] = docs["label_set"].map(len)
    docs = docs.reset_index().rename(columns={"EvrakId": "doc_id"})
    docs["date"] = pd.to_datetime(docs["date"])
    docs["department"] = docs["label"].map(PARENT)
    docs["sender_department"] = docs["sender_unit"].map(PARENT)
    stats["n_documents"] = int(len(docs))
    stats["n_documents_multi_unit"] = int((docs["n_units"] > 1).sum())
    stats["n_documents_self_dispatch"] = int((docs["sender_unit"] == docs["label"]).sum())
    stats["n_classes"] = int(docs["label"].nunique())
    stats["class_counts"] = docs["label"].value_counts().to_dict()
    stats["sender_unit_counts"] = docs["sender_unit"].value_counts().to_dict()
    stats["date_min"] = str(docs["date"].min().date())
    stats["date_max"] = str(docs["date"].max().date())
    stats["docs_per_year"] = docs["date"].dt.year.value_counts().sort_index().to_dict()
    wc = docs["text"].str.split().str.len()
    stats["words_mean"], stats["words_median"], stats["words_p90"], stats["words_max"] = float(wc.mean()), float(wc.median()), float(wc.quantile(.9)), int(wc.max())
    sc = docs["subject"].str.split().str.len()
    stats["subject_words_mean"] = float(sc.mean())

    # 5. groups of identical texts
    docs["group_id"] = docs["text"].map(norm_key).factorize()[0]
    stats["n_unique_texts"] = int(docs["group_id"].nunique())
    dup_mask = docs.groupby("group_id")["doc_id"].transform("size") > 1
    stats["n_documents_with_identical_twin"] = int(dup_mask.sum())
    maj = docs.groupby("group_id")["label"].agg(lambda s: s.value_counts().iloc[0]).sum()
    stats["label_consistency_ceiling"] = float(maj / len(docs))   # max accuracy of any function of the text

    # chronological split
    docs["split"] = np.where(docs["date"] < VAL_START, "train",
                     np.where(docs["date"] < TEST_START, "val", "test"))
    assert (docs["date"] < TEST_END).all()
    stats["split_sizes"] = docs["split"].value_counts().to_dict()
    stats["split_class_counts"] = {s: docs.loc[docs.split == s, "label"].value_counts().to_dict() for s in ["train", "val", "test"]}
    # identical texts across chronological splits (reported, not removed: realistic for templated correspondence)
    tr_groups = set(docs.loc[docs.split == "train", "group_id"])
    stats["n_test_docs_with_identical_text_in_train"] = int(docs.loc[docs.split == "test", "group_id"].isin(tr_groups).sum())

    # grouped stratified folds (two repeats)
    for seed in SEEDS:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold = np.full(len(docs), -1)
        for k, (_, te) in enumerate(skf.split(docs, docs["label"], docs["group_id"])):
            fold[te] = k
        docs[f"fold_s{seed}"] = fold

    # machine-translated English content from the first study (content only; subjects were not translated)
    docs["content_en"] = np.nan
    if PRE_V1.exists():
        try:
            v1 = pd.read_excel(PRE_V1, usecols=["EvrakId", "English"]).drop_duplicates("EvrakId")
            docs = docs.drop(columns="content_en").merge(v1.rename(columns={"EvrakId": "doc_id", "English": "content_en"}), on="doc_id", how="left")
            stats["n_documents_with_english_content"] = int(docs["content_en"].notna().sum())
        except Exception as e:
            print("WARNING: could not read the English column from", PRE_V1.name, "-", e, "(translation ablation will be skipped)")

    docs.to_pickle(OUT)
    STATS.parent.mkdir(exist_ok=True)
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

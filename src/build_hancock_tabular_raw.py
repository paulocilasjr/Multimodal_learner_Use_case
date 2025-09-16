#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# ---------- Paths / Layout ----------
DATASETS_DIRNAME = "datasets"
STRUCTURED = "StructuredData"
TEXTDATA = "TextData"
PDL1_ZIP_NAME = "PDL1_images.zip"

# only use English text dirs
ENGLISH_TEXT_DIRS = {
    "histories_english": "histories_english_text",
    "reports_english": "reports_english_text",
    "surgery_descriptions_english": "surgery_desc_english_text",
    # code dirs have no language variants, keep them:
    "icd_codes": "icd_codes_text",
    "ops_codes": "ops_codes_text",
}

# regex to pull patient id
RE_ZIP_PID = re.compile(r"patient(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
RE_TRAILING_DIGITS = re.compile(r"(\d+)$")

THREE_YEARS_DAYS = 365 * 3  # 1095


# ---------- Small helpers ----------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root() / DATASETS_DIRNAME


def default_pdl1_zip() -> Path:
    return default_data_root() / PDL1_ZIP_NAME


def normalize_patient_id_str(x: Union[str, int, float]) -> str:
    """
    Standardize patient_id to a string with zero-padding (3 digits if numeric <= 999).
    Works whether source ids are '001', '1', '0001', or 'CASE001'.
    """
    if pd.isna(x):
        return ""
    s = str(x)
    # prefer trailing digit group if present
    m = RE_TRAILING_DIGITS.search(s)
    if not m:
        digits = "".join(ch for ch in s if ch.isdigit())
    else:
        digits = m.group(1)
    if digits == "":
        return s.strip()
    try:
        n = int(digits)
        return str(n).zfill(3) if n < 1000 else str(n)
    except Exception:
        return digits


def read_json_any(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_to_frame(obj: Union[dict, list], src_name: str) -> pd.DataFrame:
    # Accept mapping id->record or list of dicts
    if isinstance(obj, dict):
        rows = []
        for k, v in obj.items():
            if isinstance(v, dict):
                vv = v.copy()
                vv["patient_id"] = normalize_patient_id_str(k)
                rows.append(vv)
        return pd.DataFrame(rows)
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].map(normalize_patient_id_str)
        return df
    else:
        raise ValueError(f"Unsupported JSON structure in '{src_name}'")


def load_structured(struct_dir: Path, prefix: str) -> pd.DataFrame:
    p = struct_dir / f"{prefix}_data.json"
    if not p.exists():
        return pd.DataFrame()
    df = json_to_frame(read_json_any(p), p.name)
    # prefix columns (except patient_id)
    cols = ["patient_id"] + [f"{prefix}__{c}" for c in df.columns if c != "patient_id"]
    df.columns = cols
    return df


def map_images_from_zip(zip_path: Path) -> pd.DataFrame:
    """
    Return a dataframe with one row per patient_id:
      patient_id | image_path  (basename only)
    Matching based on '...patient<digits>.ext'
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Images zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = [zi.filename for zi in zf.infolist() if not zi.is_dir()]

    rows = []
    for fn in files:
        base = Path(fn).name
        m = RE_ZIP_PID.search(base)
        if not m:
            continue
        pid = normalize_patient_id_str(m.group(1))
        rows.append((pid, base))

    if not rows:
        return pd.DataFrame(columns=["patient_id", "image_path"])

    df = pd.DataFrame(rows, columns=["patient_id", "image_path"])
    # if multiple tiles per patient, pick lexicographically first
    df = df.sort_values(["patient_id", "image_path"]).groupby("patient_id", as_index=False).first()
    return df


def read_text_dir(dir_path: Path, colname: str) -> pd.DataFrame:
    """
    Read all files in dir, map to patient by trailing digits in filename stem,
    and concatenate into one string per patient.
    """
    if not dir_path.exists():
        return pd.DataFrame(columns=["patient_id", colname])

    buckets: Dict[str, List[str]] = {}
    for p in sorted(dir_path.glob("**/*")):
        if not p.is_file():
            continue
        stem = p.stem
        m = RE_TRAILING_DIGITS.search(stem)
        if not m:
            continue
        pid = normalize_patient_id_str(m.group(1))
        try:
            if p.suffix.lower() == ".json":
                txt = json.dumps(read_json_any(p), ensure_ascii=False)
            else:
                txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            txt = f"[READ_ERROR:{p.name}] {e}"
        buckets.setdefault(pid, []).append(txt)

    if not buckets:
        return pd.DataFrame(columns=["patient_id", colname])

    data = {"patient_id": [], colname: []}
    for pid, chunks in buckets.items():
        data["patient_id"].append(pid)
        data[colname].append("\n\n".join(chunks))
    return pd.DataFrame(data)


def build_english_text_block(text_root: Path) -> pd.DataFrame:
    """Only English dirs (+ ICD/OPS). One row per patient."""
    parts = []
    for subdir, colname in ENGLISH_TEXT_DIRS.items():
        parts.append(read_text_dir(text_root / subdir, colname))
    parts = [df for df in parts if not df.empty]
    if not parts:
        return pd.DataFrame(columns=["patient_id"])

    out = parts[0]
    for nxt in parts[1:]:
        out = out.merge(nxt, on="patient_id", how="outer")
    return out


def pivot_blood_to_wide(df_blood: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long blood table to wide, one row per patient.
    Keep numeric 'value' per analyte chosen by closest to treatment day
    (min abs(days_before_first_treatment)). Drop unit and other long-form columns.
    """
    if df_blood.empty:
        return df_blood

    # Expected columns (after prefixing): blood__value, blood__analyte_name, blood__days_before_first_treatment, ...
    need = {"patient_id", "blood__analyte_name", "blood__value"}
    if not need.issubset(df_blood.columns):
        # If structure is unexpected, at least return unique patient_ids
        return df_blood[["patient_id"]].drop_duplicates()

    df_blood = df_blood.copy()

    # choose record per patient+analyte closest to treatment day (abs(days))
    if "blood__days_before_first_treatment" in df_blood.columns:
        df_blood["_abs_days"] = pd.to_numeric(
            df_blood["blood__days_before_first_treatment"], errors="coerce"
        ).abs()
    else:
        df_blood["_abs_days"] = 0.0

    # force numeric value if possible
    df_blood["blood__value"] = pd.to_numeric(df_blood["blood__value"], errors="coerce")

    df_blood = (
        df_blood.sort_values(["patient_id", "blood__analyte_name", "_abs_days"])
        .groupby(["patient_id", "blood__analyte_name"], as_index=False)
        .first()[["patient_id", "blood__analyte_name", "blood__value"]]
    )

    # pivot: analyte_name → column
    df_w = df_blood.pivot(index="patient_id", columns="blood__analyte_name", values="blood__value")

    # sanitize column names and prefix
    def _san(c: str) -> str:
        c = re.sub(r"\s+", "_", str(c)).strip("_")
        c = re.sub(r"[^0-9A-Za-z_]+", "_", c)
        return f"blood__{c}"

    df_w.columns = [_san(c) for c in df_w.columns]
    df_w = df_w.reset_index()

    return df_w


def build_survival_label(df_clin: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Return (label_survival, status_cleaned) after excluding non–tumor-specific deaths.
    label_survival is in {'living','deceased'} (lowercase strings).
    """
    stat = df_clin["clinical__survival_status"].astype(str).str.strip().str.lower()
    cause = df_clin.get("clinical__survival_status_with_cause", pd.Series(index=df_clin.index, dtype="object"))
    cause = cause.astype(str).str.strip().str.lower()

    # exclude non–tumor-specific deaths
    keep_mask = ~(cause == "deceased not tumor specific")
    stat = stat.where(keep_mask, np.nan)

    # normalize to labels
    lbl = pd.Series(index=stat.index, dtype="object")
    lbl[stat == "living"] = "living"
    lbl[stat == "deceased"] = "deceased"

    return lbl, stat


def build_recurrence_label(df_clin: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    3-year recurrence label:
    - 'recurrence' if recurrence == "yes" and days_to_recurrence <= 1095
    - 'no_recurrence' if recurrence == "no"  and days_to_last_information >= 1095
    - else NaN
    """
    rec = df_clin.get("clinical__recurrence", pd.Series(index=df_clin.index, dtype="object"))
    dtr = df_clin.get("clinical__days_to_recurrence", pd.Series(index=df_clin.index, dtype="float64"))
    dtli = df_clin.get("clinical__days_to_last_information", pd.Series(index=df_clin.index, dtype="float64"))

    rec = rec.astype(str).str.strip().str.lower().replace({"nan": np.nan})
    dtr = pd.to_numeric(dtr, errors="coerce")
    dtli = pd.to_numeric(dtli, errors="coerce")

    pos = (rec == "yes") & (dtr.notna()) & (dtr <= THREE_YEARS_DAYS)
    neg = (rec == "no") & (dtli.notna()) & (dtli >= THREE_YEARS_DAYS)

    lbl = pd.Series(index=rec.index, dtype="object")
    lbl[pos] = "recurrence"
    lbl[neg] = "no_recurrence"

    # (optional) reason (not used downstream)
    reason = pd.Series(index=rec.index, dtype="object")
    reason[pos] = "recurrence_within_3y"
    reason[neg] = "no_recurrence_>=3y_followup"

    return lbl, reason


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious post-outcome / time-to-event leakage features.
    """
    to_drop = set()

    # Any days_to_* in clinical
    to_drop.update([c for c in df.columns if c.startswith("clinical__days_to_")])

    # Progression & metastasis indicators (and their locations/days)
    to_drop.update([c for c in df.columns if c.startswith("clinical__progress_")])
    to_drop.update([c for c in df.columns if c.startswith("clinical__metastasis_")])

    # Raw target sources already superseded by labels
    for c in ("clinical__survival_status", "clinical__survival_status_with_cause", "clinical__recurrence"):
        if c in df.columns:
            to_drop.add(c)

    if to_drop:
        df = df.drop(columns=list(to_drop), errors="ignore")

    return df


# ---------- Main build ----------
def main():
    ap = argparse.ArgumentParser(
        description="Build HANCOCK tabular CSV: one row per patient, English text only, "
                    "drop patients without image, create label_survival & label_recurrence, "
                    "and automatically drop leakage columns."
    )
    ap.add_argument("--data_root", default=None, help="Path to datasets/ (default: <repo>/datasets)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--pdl1_zip", default=None, help="Path to PDL1_images.zip (default: <data_root>/PDL1_images.zip)")
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    pdl1_zip = Path(args.pdl1_zip) if args.pdl1_zip else default_pdl1_zip()

    struct_dir = data_root / STRUCTURED
    text_dir = data_root / TEXTDATA

    # --- Structured data
    df_clin = load_structured(struct_dir, "clinical")
    if df_clin.empty:
        raise FileNotFoundError(f"Missing {struct_dir/'clinical_data.json'} or empty data.")

    df_path = load_structured(struct_dir, "path")
    df_blood = load_structured(struct_dir, "blood")

    # normalize all patient ids
    for d in (df_clin, df_path, df_blood):
        if not d.empty:
            d["patient_id"] = d["patient_id"].map(normalize_patient_id_str)

    # pivot blood → wide (drop unit etc.)
    if not df_blood.empty:
        df_blood_w = pivot_blood_to_wide(df_blood)
    else:
        df_blood_w = pd.DataFrame(columns=["patient_id"])

    # merge structured (one row per patient)
    df = df_clin[["patient_id"]].drop_duplicates()
    for part in (df_clin, df_path, df_blood_w):
        if not part.empty:
            df = df.merge(part, on="patient_id", how="left")

    # --- English text only; map filenames to trailing digits
    df_text = build_english_text_block(text_dir)
    if not df_text.empty:
        df = df.merge(df_text, on="patient_id", how="left")

    # --- Image paths; drop patients w/o image
    df_imgs = map_images_from_zip(pdl1_zip)
    df = df.merge(df_imgs, on="patient_id", how="left")
    before = len(df)
    df = df[df["image_path"].notna()].copy()
    print(f"[OK] Dropped patients without image: kept {len(df)}/{before}")

    # --- Labels
    # survival label & exclusion of non–tumor-specific deaths
    label_surv, _surv_status = build_survival_label(df)
    keep_surv = label_surv.notna()  # drops non–tumor-specific and missing status
    dropped = (~keep_surv).sum()
    if dropped:
        print(f"[OK] Excluded {dropped} patients (non–tumor-specific death or missing survival).")
    df = df[keep_surv].copy()
    label_surv = label_surv[keep_surv]

    # recurrence label (3y rule) — leave NaN where insufficient follow-up
    label_rec, _rec_reason = build_recurrence_label(df)

    # attach labels
    df["label_survival"] = label_surv.astype("string")
    df["label_recurrence"] = label_rec.astype("string")

    # --- Drop leakage columns automatically
    df = drop_leakage_columns(df)

    # --- Final ordering (patient_id will be dropped right before saving)
    front = [
        "patient_id", "image_path",
        "label_survival",
        "label_recurrence",
    ]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # --- Drop patient_id (not a feature for training)
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])

    # --- Write
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with shape {df.shape}")


if __name__ == "__main__":
    main()

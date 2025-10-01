#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attach image filenames to a patient table by matching the 3 digits right before '.png'
(e.g., TumorCenter_CD8_block7_x1_y2_patient003.png -> patient_id '003').

- For each directory passed via --image_dirs, create a new column named after the
  last directory component. The value is the chosen image filename for that patient.

- If multiple images exist for the same patient in a directory, apply a deterministic
  tie-breaking strategy aligned with the HANCOCK repo conventions:
    1) Prefer files whose stem contains any of these (in this order): "TumorCenter", "Center", "Core"
    2) Prefer lower block numbers if a 'block<INT>' pattern exists (e.g., 'block1' < 'block7')
    3) Prefer the lexicographically smallest filename as a final tiebreak.

- Saves a combined TSV to --out_tsv, and (NEW) also saves per-split TSVs if a column
  named 'split' exists:
    split == 0 -> "Train_<basename(out_tsv)>"
    split == 1 -> "Validation_<basename(out_tsv)>"
    split == 2 -> "Test_<basename(out_tsv)>"
  Files are written into the same directory as --out_tsv.

- Verbose logging explains which patients matched, which didn’t, and directory progress.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# --------------------- Logging helpers ---------------------

def log(msg: str):
    print(f"[attach] {msg}")


# --------------------- Filename parsing ---------------------

PATIENT_TAIL_RE = re.compile(r"(\d{3})(?=\.png$)", flags=re.IGNORECASE)
BLOCK_RE = re.compile(r"block(\d+)", flags=re.IGNORECASE)

PAPER_PREF_TOKENS = ["TumorCenter", "Center", "Core"]  # preference order


def extract_patient_tail(fname: str) -> str | None:
    """
    Return the 3-digit patient string immediately before '.png', or None if no match.
    """
    m = PATIENT_TAIL_RE.search(fname)
    return m.group(1) if m else None


def extract_block_num(fname: str) -> int | None:
    """
    Find 'block<INT>' pattern in filename, return INT or None.
    """
    m = BLOCK_RE.search(fname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def paper_style_priority_key(fname: str):
    """
    Build a tuple that implements the paper-style tie-breaking order.

    Lower tuple sorts earlier (i.e., higher priority):
      1) presence index of PAPER_PREF_TOKENS in stem (0 best, large if none)
      2) block number (smaller first; None treated as large)
      3) lexicographic filename as final deterministic tiebreaker
    """
    stem = Path(fname).stem
    # 1) token priority
    token_rank = min(
        (i for i, tok in enumerate(PAPER_PREF_TOKENS) if tok.lower() in stem.lower()),
        default=len(PAPER_PREF_TOKENS) + 1,
    )
    # 2) block number
    block = extract_block_num(fname)
    block_rank = block if block is not None else 10_000_000
    # 3) final tiebreaker
    return (token_rank, block_rank, fname.lower())


# --------------------- I/O ---------------------

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    # CSV or TSV autodetect
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    # Normalize patient_id to 3-digit, as strings
    if "patient_id" not in df.columns:
        raise ValueError("Input table must have a 'patient_id' column.")
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.zfill(3)
    return df


def scan_image_dirs(image_dirs: list[Path]) -> dict[str, dict[str, list[str]]]:
    """
    For each directory, index filenames by patient tail 'NNN'.
    Returns mapping:
        by_dir[dir_name][patient_tail] -> list of filenames (not full paths).
    """
    by_dir: dict[str, dict[str, list[str]]] = {}
    for d in image_dirs:
        if not d.exists() or not d.is_dir():
            log(f"Directory does not exist or is not a dir: {d} (skipping)")
            continue
        col_name = d.name  # column name is the last directory component
        log(f"Scanning directory: {d} → column '{col_name}'")
        bucket = defaultdict(list)
        cnt = 0
        for p in d.glob("*.png"):
            pid = extract_patient_tail(p.name)
            if pid is None:
                continue
            bucket[pid].append(p.name)
            cnt += 1
        by_dir[col_name] = dict(bucket)
        log(f"  Collected {cnt} .png files with valid patient tails in '{col_name}'.")
    return by_dir


def choose_one_filename(flist: list[str]) -> str:
    """
    Choose exactly one filename from a list using paper-style priority.
    """
    if len(flist) == 1:
        return flist[0]
    return sorted(flist, key=paper_style_priority_key)[0]


# --------------------- Main processing ---------------------

def attach_images(df: pd.DataFrame, by_dir: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    """
    For each directory/column, attach one filename per patient according to priority rules.
    """
    out = df.copy()
    for col_name, mapping in by_dir.items():
        log(f"Attaching images for column '{col_name}'...")
        if col_name in out.columns:
            log(f"  WARNING: Column '{col_name}' already exists in table. It will be overwritten.")
        chosen = []
        found = 0
        missing = 0
        for pid in out["patient_id"].astype(str):
            file_list = mapping.get(pid)
            if not file_list:
                chosen.append("")
                missing += 1
            else:
                best = choose_one_filename(file_list)
                chosen.append(best)
                found += 1
        out[col_name] = chosen
        log(f"  Done '{col_name}': matched={found}, missing={missing}")
    return out


def save_outputs(df: pd.DataFrame, out_tsv: Path):
    out_dir = out_tsv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write the combined table first
    log(f"Writing combined table to: {out_tsv}")
    df.to_csv(out_tsv, sep="\t", index=False)

    # Then, if 'split' exists, write split-specific files
    if "split" not in df.columns:
        log("Column 'split' not found; skipping split-specific TSVs.")
        return

    name_map = {0: "Train", 1: "Validation", 2: "Test"}
    base = out_tsv.name  # e.g., "paper_table_recurrence_in_HE.tsv"

    # Make sure 'split' is numeric-like
    split_series = pd.to_numeric(df["split"], errors="coerce")

    for split_value, group_name in name_map.items():
        mask = split_series == split_value
        n_rows = int(mask.sum())
        if n_rows == 0:
            log(f"No rows found for split={split_value} ({group_name}); not writing a file.")
            continue
        out_path = out_dir / f"{group_name}_{base}"
        log(f"Writing {group_name} split (split={split_value}, rows={n_rows}) to: {out_path}")
        df.loc[mask].to_csv(out_path, sep="\t", index=False)


# --------------------- CLI ---------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Attach image filenames to patient rows by matching 3-digit patient tails."
    )
    ap.add_argument(
        "--in_table",
        required=True,
        help="Path to the input CSV/TSV patient table (must include 'patient_id' and ideally 'split').",
    )
    ap.add_argument(
        "--image_dirs", "-image_dirs",
        nargs="+",
        required=True,
        help="One or more directories containing .png images to match. "
             "Each directory becomes a new column named after its last path component.",
    )
    ap.add_argument(
        "--out_tsv",
        required=True,
        help="Path to write the combined TSV. The split-specific TSVs will be written into the same folder, "
             "named as Train_<basename>, Validation_<basename>, Test_<basename>.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    in_table = Path(args.in_table)
    out_tsv = Path(args.out_tsv)
    image_dirs = [Path(p) for p in args.image_dirs]

    log(f"Loading table: {in_table}")
    df = load_table(in_table)
    log(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # Scan image dirs and index by patient tail
    by_dir = scan_image_dirs(image_dirs)

    # Attach image names per directory/column
    df_out = attach_images(df, by_dir)

    # Save combined + per-split TSVs
    save_outputs(df_out, out_tsv)

    log("All done.")


if __name__ == "__main__":
    main()


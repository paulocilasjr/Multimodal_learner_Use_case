#!/usr/bin/env python3
import re
from pathlib import Path
from glob import glob
import pandas as pd

# --- run from: src/ ---
train_txt = Path("../datasets/train_paper_table_recurrence_in.txt")
test_txt  = Path("../datasets/test_paper_table_recurrence_in.txt")
tma_root  = Path("../datasets/TMA_Cores")  # contains tma_tumorcenter_* dirs
out_train = Path("../datasets/train_paper_recurrence_in_img_all.csv")
out_test  = Path("../datasets/test_paper_recurrence_in_img_all.csv")

# Only CD3 and CD8 images (replacing the tabular counts)
MODALITY_DIRS = {
    "cd3_image_path": "tma_tumorcenter_CD3",
    "cd8_image_path": "tma_tumorcenter_CD8",
}

# columns to drop (counts)
DROP_COLS = {"cd3_z", "cd3_inv", "cd8_z", "cd8_inv"}

# Selection policy for multiple tiles per patient
# Options implemented: 'first_sorted' (by numeric (block, x, y))
SELECT_POLICY = "first_sorted"

tile_re = re.compile(r"block(\d+)_x(\d+)_y(\d+)_patient(\d+)", flags=re.IGNORECASE)

def load_table(path: Path) -> pd.DataFrame:
    """Load paper tables (.txt/.tsv/.csv), normalize patient_id as string."""
    sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(path, sep=sep)
    if "patient_id" not in df.columns:
        raise KeyError(f"'patient_id' column missing in {path}")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df

def parse_tile(fname: str):
    """
    Extract (block, x, y, patient) from a filename like:
    TumorCenter_CD3_block1_x6_y9_patient510.png
    Returns tuple of ints (block, x, y, patient) or None if not matched.
    """
    m = tile_re.search(fname)
    if not m:
        return None
    b, x, y, pid = map(int, m.groups())
    return b, x, y, pid

def build_selection(mod_folder: Path):
    """
    Build patient_id -> selected_image_path using SELECT_POLICY.
    If multiple tiles exist for a patient, choose deterministically.
    """
    per_pid = {}  # pid (str) -> list[(key_tuple, path)]
    for png in glob(str(mod_folder / "*.png")):
        name = Path(png).name
        parsed = parse_tile(name)
        if parsed is None:
            # fallback: try to capture 'patient(\d+)' anywhere in name
            m = re.search(r"patient(\d+)", name, flags=re.IGNORECASE)
            if not m:
                continue
            pid = int(m.group(1))
            key = (10**9, 10**9, 10**9)  # push unmatched to the end
        else:
            b, x, y, pid = parsed
            key = (b, x, y)
        per_pid.setdefault(str(pid), []).append((key, str(Path(png).resolve())))

    # apply selection policy
    selected = {}
    multi_counts = 0
    for pid, items in per_pid.items():
        if len(items) > 1:
            multi_counts += 1
        if SELECT_POLICY == "first_sorted":
            items.sort(key=lambda t: t[0])  # sort by (block, x, y)
            selected[pid] = items[0][1]
        else:
            # default fallback: first as-is (but deterministic sort anyway)
            items.sort(key=lambda t: t[0])
            selected[pid] = items[0][1]
    return selected, multi_counts, len(per_pid)

def attach_images(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    summary = []
    for col, subdir in MODALITY_DIRS.items():
        folder = tma_root / subdir
        if not folder.exists():
            raise FileNotFoundError(f"Missing modality folder: {folder}")
        selection, n_multi, n_with_any = build_selection(folder)
        summary.append((col, n_multi, n_with_any))
        df[col] = df["patient_id"].map(selection).astype("string")
    # brief summary to stdout
    for col, n_multi, n_with_any in summary:
        print(f"[info] {col}: patients with >=2 tiles={n_multi} / patients with ≥1 tile={n_with_any} "
              f"(policy='{SELECT_POLICY}')")
    return df

def prep(df: pd.DataFrame) -> pd.DataFrame:
    # drop CD3/CD8 count features if present
    keep = [c for c in df.columns if c not in DROP_COLS]
    df = df[keep].copy()
    # add image paths (CD3/CD8 only)
    df = attach_images(df)

    # ensure numeric target if present
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)

    # reorder for clarity
    first = [c for c in ["patient_id", "split", "target"] if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]

def main():
    train = prep(load_table(train_txt))
    test  = prep(load_table(test_txt))

    # require both images present
    need_all_images = False
    if need_all_images:
        im_cols = list(MODALITY_DIRS.keys())
        bt, bs = len(train), len(test)
        train = train.dropna(subset=im_cols)
        test  = test.dropna(subset=im_cols)
        print(f"[info] dropped due to missing images → train: {bt-len(train)} | test: {bs-len(test)}")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)

    print(f"Wrote:\n  {out_train} ({train.shape[0]} rows, {train.shape[1]} cols)\n"
          f"  {out_test}  ({test.shape[0]} rows, {test.shape[1]} cols)")

if __name__ == "__main__":
    main()

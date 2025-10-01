#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import warnings
import zipfile
import tempfile
import shutil
import time
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

# Reduce transient NFS .nfs* cleanup issues
os.environ.setdefault("TMPDIR", "/tmp")

# AutoGluon
from autogluon.multimodal import MultiModalPredictor

# Seeding / Torch
import random
import torch
torch.set_float32_matmul_precision("high")


# --------------------------- Utils ---------------------------
def set_seed_all(seed: int | None):
    """Deterministic seeding across libs + cuDNN for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_table(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if str(path).lower().endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def check_and_drop_missing_images(df: pd.DataFrame, image_cols: list[str], tag: str) -> pd.DataFrame:
    """Drop rows whose image paths do not exist on disk for any required image column."""
    df = df.copy()
    total_dropped = 0
    for col in image_cols:
        if col not in df.columns:
            warnings.warn(f"[warn] Column '{col}' not found in {tag}.")
            continue
        mask_bad = df[col].notna() & (~df[col].astype(str).apply(lambda p: Path(p).exists()))
        n_bad = int(mask_bad.sum())
        if n_bad:
            print(f"[clean] dropping {n_bad} rows with non-existent '{col}'")
            df = df.loc[~mask_bad].copy()
            total_dropped += n_bad
    print(f"[clean] total rows dropped due to image issues: {total_dropped}")
    return df


def stratified_train_val_split(df: pd.DataFrame, label_col: str, val_frac: float, random_state: int):
    y = df[label_col].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    idx_train, idx_val = next(sss.split(np.zeros(len(y)), y))
    return df.iloc[idx_train].copy(), df.iloc[idx_val].copy()


def numeric_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in set(exclude)]


def compute_train_medians(train_df: pd.DataFrame, label_col: str, extra_exclude: list[str]) -> pd.Series:
    exclude = set([label_col] + extra_exclude)
    cols = numeric_columns(train_df, exclude=list(exclude))
    med = train_df[cols].median(numeric_only=True)
    return med


def impute_with_medians(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in medians.index if c in df.columns]
    if cols:
        df[cols] = df[cols].fillna(medians[cols])
    return df


def smote_augment_numeric_only(
    train_df: pd.DataFrame,
    label_col: str,
    image_cols: list[str],
    pid_col: str | None,
    smote_k: int = 5,
    smote_random_state: int = 42,
) -> pd.DataFrame:
    """Apply SMOTE to numeric columns only; reattach non-numeric cols."""
    df = train_df.copy()
    exclude = set(image_cols + ([pid_col] if pid_col and pid_col in df.columns else []) + [label_col])
    num_cols = numeric_columns(df, exclude=list(exclude))

    if num_cols and df[num_cols].isna().any().any():
        raise ValueError("NaNs detected in numeric features prior to SMOTE. Impute before SMOTE.")

    X_num = df[num_cols].to_numpy() if num_cols else np.empty((len(df), 0))
    y = df[label_col].astype(int).to_numpy()

    if X_num.shape[1] == 0:
        print("[smote] No numeric columns found (after excluding ID/image/label). Skipping SMOTE.")
        return df

    sm = SMOTE(k_neighbors=smote_k, random_state=smote_random_state)
    X_res, y_res = sm.fit_resample(X_num, y)

    df_num = pd.DataFrame(X_res, columns=num_cols, index=None)
    df_lab = pd.Series(y_res, name=label_col)

    idx_resampled = getattr(sm, "sample_indices_", None)
    if idx_resampled is None:
        minority_idx = np.flatnonzero(df[label_col].astype(int).to_numpy() == 1)
        if len(minority_idx) == 0:
            print("[smote] Fallback failed: no minority rows. Returning original df.")
            return df
        meta_source_idx = np.resize(minority_idx, len(df_num))
        df_meta = df.iloc[meta_source_idx].reset_index(drop=True)
    else:
        df_meta = df.iloc[idx_resampled].reset_index(drop=True)

    non_num_cols = [c for c in df.columns if c not in num_cols and c != label_col]
    res = pd.concat(
        [
            df_meta[non_num_cols].reset_index(drop=True),
            df_num.reset_index(drop=True),
            df_lab.reset_index(drop=True),
        ],
        axis=1,
    )
    res[label_col] = res[label_col].astype(int)
    print(f"[smote] Done. Rows: {len(df)} -> {len(res)}")
    return res


def ensure_binary_int_labels(df: pd.DataFrame, label_col: str):
    if df[label_col].dtype == bool:
        df[label_col] = df[label_col].astype(int)
    elif df[label_col].dtype.kind in "OUS":
        df[label_col] = (
            df[label_col]
            .astype(str)
            .str.lower()
            .map({"no": 0, "yes": 1, "negative": 0, "positive": 1})
            .fillna(df[label_col])
            .astype(int)
        )
    else:
        df[label_col] = df[label_col].astype(int)
    return df


def _choose_positive_label_from_columns(cols) -> object:
    for cand in (1, "1", True, "True", "true", "yes", "positive", "Positive", "YES"):
        if cand in cols:
            return cand
    numeric_vals, all_numeric = [], True
    for c in cols:
        try:
            numeric_vals.append(float(c))
        except Exception:
            all_numeric = False
            break
    if all_numeric and len(numeric_vals) == len(cols):
        return cols[int(np.argmax(numeric_vals))]
    return cols[-1]


def positive_class_proba(y_proba, predictor: MultiModalPredictor | None = None) -> np.ndarray | None:
    if y_proba is None:
        return None
    if hasattr(y_proba, "values") and hasattr(y_proba, "columns"):
        cols = list(y_proba.columns)
        pos_col = _choose_positive_label_from_columns(cols)
        return y_proba[pos_col].to_numpy()
    arr = np.asarray(y_proba)
    if arr.ndim == 2:
        return arr[:, 0] if arr.shape[1] == 1 else arr[:, -1]
    return arr


def normalize_preset(preset: str) -> str:
    if preset is None:
        return "medium_quality"
    low = str(preset).lower()
    if low in {"medium", "medium_quality"}:
        return "medium_quality"
    if low in {"high", "high_quality"}:
        return "high_quality"
    if low in {"best_quality"}:
        return "best_quality"
    return preset


def materialize_images_from_zip(images_zip: str, image_cols: list[str], df_list: list[pd.DataFrame]) -> str | None:
    """
    Extract images_zip to a temp folder and rewrite image columns in-place
    to absolute paths inside the extracted directory, matching by filename.
    """
    if not images_zip:
        return None

    zippath = Path(images_zip)
    if not zippath.exists():
        raise FileNotFoundError(f"--images_zip not found: {images_zip}")

    extract_dir = Path(tempfile.mkdtemp(prefix="ag_images_", dir=os.environ.get("TMPDIR", "/tmp")))
    print(f"[zip] Extracting images from {images_zip} -> {extract_dir}")
    with zipfile.ZipFile(zippath, "r") as zf:
        zf.extractall(extract_dir)

    name_to_path = {}
    cnt_files = 0
    for p in extract_dir.rglob("*"):
        if p.is_file():
            name_to_path[p.name] = str(p.resolve())
            cnt_files += 1
    print(f"[zip] Indexed {cnt_files} files from extracted archive")

    def _rewrite_col(df: pd.DataFrame, col: str) -> tuple[int, int, int]:
        if col not in df.columns:
            print(f"[zip] Column '{col}' not in DataFrame; skipping rewrite.")
            return (0, 0, 0)
        matched = missing = skipped_empty = 0
        new_vals = []
        for v in df[col].astype(str).tolist():
            if v in ("", "nan", "None", "NaN"):
                new_vals.append(v); skipped_empty += 1; continue
            fname = os.path.basename(v)
            new_path = name_to_path.get(fname, None)
            if new_path and os.path.exists(new_path):
                new_vals.append(new_path); matched += 1
            else:
                new_vals.append(v); missing += 1
        df[col] = new_vals
        return (matched, missing, skipped_empty)

    for col in image_cols:
        total_m = total_x = total_e = 0
        for d in df_list:
            m, x, e = _rewrite_col(d, col)
            total_m += m; total_x += x; total_e += e
        print(f"[zip] Column '{col}': matched={total_m}, not_found_in_zip={total_x}, empty={total_e}")

    print(f"[zip] Extraction directory: {extract_dir}")
    return str(extract_dir)


# --------------------------- Cleanup helpers ---------------------------
def safe_remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)  # py3.8+: ignore if not exists
    except Exception as e:
        print(f"[cleanup] Could not remove file {path}: {e}")


def safe_rmtree(path: Path, retries: int = 5, delay_sec: float = 2.0) -> None:
    """NFS-friendly recursive delete with retries."""
    if not path or not Path(path).exists():
        return
    for i in range(retries):
        try:
            shutil.rmtree(path)
            print(f"[cleanup] Removed directory {path}")
            return
        except Exception as e:
            print(f"[cleanup] rmtree failed ({i+1}/{retries}) for {path}: {e}")
            time.sleep(delay_sec)
    print(f"[cleanup] Gave up removing {path}. You may delete it manually.")


def prune_training_artifacts(run_dir: Path) -> None:
    """
    Remove large intermediate artifacts that AutoGluon/Lightning create during training.
    This keeps the trained predictor loadable (we do NOT remove the predictor state).
    """
    if not run_dir.exists():
        return

    # Delete Lightning logs
    for logdir in ("lightning_logs",):
        p = run_dir / logdir
        if p.exists():
            print(f"[cleanup] Removing {p}")
            safe_rmtree(p)

    # Delete all .ckpt files (top-k checkpoints, etc.). Predictor state is saved separately.
    patterns = ["**/*.ckpt"]
    removed = 0
    for pat in patterns:
        for f in run_dir.glob(pat):
            print(f"[cleanup] Removing checkpoint {f}")
            safe_remove(f)
            removed += 1
    print(f"[cleanup] Pruned {removed} checkpoint files in {run_dir}")


# --------------------------- Main ---------------------------

def main():
    p = argparse.ArgumentParser(
        description="Multimodal (tabular + images) training with train-median imputation and optional SMOTE."
    )
    p.add_argument("--train_csv", required=True, type=str)
    p.add_argument("--test_csv", required=True, type=str)
    p.add_argument("--label", default="target", type=str)
    p.add_argument("--pid_col", default="patient_id", type=str)
    p.add_argument("--image_cols", nargs="+", required=True, help="e.g. cd3_image_path cd8_image_path")
    p.add_argument("--val_frac", type=float, default=0.125)
    p.add_argument("--preset", type=str, default="high", help="medium|high|best_quality (mapped to AG 1.4 names)")
    p.add_argument("--seed", type=int, default=None, help="single seed (optional)")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="multiple seeds, overrides --seed if provided")
    p.add_argument("--out_dir", type=str, default="./AutogluonModels/ag-mm-exp")
    p.add_argument("--save_metrics", action="store_true")
    # SMOTE
    p.add_argument("--smote", action="store_true")
    p.add_argument("--smote_k", type=int, default=5)
    p.add_argument("--smote_random_state", type=int, default=42)
    # Images zip
    p.add_argument("--images_zip", type=str, default=None, help="Optional: path to a .zip with the images")
    # Cleanup controls
    p.add_argument("--cleanup_extracted", action="store_true",
                   help="Delete the extracted images directory after training.")
    p.add_argument("--prune_checkpoints", action="store_true",
                   help="Delete *.ckpt and lightning_logs/ inside each run directory after training to save space.")

    args = p.parse_args()

    ag_preset = normalize_preset(args.preset)
    print(f"[info] Running seed sweep: {args.seeds if args.seeds else [args.seed] if args.seed is not None else [42]} with preset='{ag_preset}'")

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    image_cols = args.image_cols
    label_col = args.label
    pid_col = args.pid_col if args.pid_col not in ("", None) else None

    seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [42])

    # Load
    df_train_all = read_table(train_path)
    df_test_all = read_table(test_path)

    # Zip extract and rewrite paths by filename (if provided)
    extracted_dir = None
    if args.images_zip:
        extracted_dir = materialize_images_from_zip(args.images_zip, image_cols, [df_train_all, df_test_all])

    # Ensure labels
    df_train_all = ensure_binary_int_labels(df_train_all, label_col)
    df_test_all = ensure_binary_int_labels(df_test_all, label_col)

    # Drop rows with missing image paths (pre-clean)
    df_train_all = check_and_drop_missing_images(df_train_all, image_cols, tag="train")
    df_test_all = check_and_drop_missing_images(df_test_all, image_cols, tag="test")

    results = []
    out_dir_root = Path(args.out_dir)
    out_dir_root.mkdir(parents=True, exist_ok=True)

    for s in seeds:
        set_seed_all(s)

        # Split
        df_train, df_val = stratified_train_val_split(df_train_all, label_col, args.val_frac, random_state=s)

        # Impute stats from TRAIN only
        extra_exclude = image_cols + ([pid_col] if (pid_col and pid_col in df_train.columns) else [])
        med = compute_train_medians(df_train, label_col, extra_exclude=extra_exclude)

        # Impute
        df_train = impute_with_medians(df_train, med)
        df_val   = impute_with_medians(df_val, med)
        df_test  = impute_with_medians(df_test_all.copy(), med)

        # SMOTE (train only)
        if args.smote:
            df_train = smote_augment_numeric_only(
                train_df=df_train,
                label_col=label_col,
                image_cols=image_cols,
                pid_col=pid_col if (pid_col and pid_col in df_train.columns) else None,
                smote_k=args.smote_k,
                smote_random_state=args.smote_random_state,
            )

        print(f"Rows → train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

        # Save path for this seed
        save_path = out_dir_root / f"{ag_preset}-seed{s}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Train
        predictor = MultiModalPredictor(
            label=label_col,
            problem_type="binary",
            eval_metric="roc_auc",
            path=str(save_path),
        )

        predictor.fit(
            train_data=df_train,
            tuning_data=df_val,
            presets=ag_preset,
            hyperparameters={"data.image.missing_value_strategy": "skip"},
        )

        # Eval
        proba_train = positive_class_proba(predictor.predict_proba(df_train), predictor)
        proba_val   = positive_class_proba(predictor.predict_proba(df_val), predictor)
        proba_test  = positive_class_proba(predictor.predict_proba(df_test), predictor)

        y_train = df_train[label_col].to_numpy()
        y_val   = df_val[label_col].to_numpy()
        y_test  = df_test[label_col].to_numpy()

        auc_train = roc_auc_score(y_train, proba_train) if proba_train is not None else np.nan
        auc_val   = roc_auc_score(y_val,   proba_val)   if proba_val   is not None else np.nan
        auc_test  = roc_auc_score(y_test,  proba_test)  if proba_test  is not None else np.nan

        print(f"[SEED {s}] ROC-AUC → TRAIN: {auc_train:.4f} | VAL: {auc_val:.4f} | TEST: {auc_test:.4f}")

        results.append({
            "seed": s,
            "preset": ag_preset,
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
            "roc_auc_train": float(auc_train),
            "roc_auc_val": float(auc_val),
            "roc_auc_test": float(auc_test),
            "save_path": str(save_path),
            "images_zip_extracted_dir": extracted_dir or "",
        })

        # Optional pruning of training artifacts per seed
        if args.prune_checkpoints:
            prune_training_artifacts(save_path)

    # Save run summary
    if args.save_metrics:
        out_csv = out_dir_root / "metrics_mm.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"[info] wrote metrics to {out_csv}")

    if results:
        df_res = pd.DataFrame(results)
        print("\n=== Summary over seeds ===")
        for split in ["roc_auc_train", "roc_auc_val", "roc_auc_test"]:
            mean = df_res[split].mean()
            std  = df_res[split].std()
            print(f"{split}: {mean:.4f} ± {std:.4f}")

    # Optional cleanup of extracted images directory
    if extracted_dir and args.cleanup_extracted:
        print(f"[cleanup] Removing extracted images directory: {extracted_dir}")
        safe_rmtree(Path(extracted_dir))


if __name__ == "__main__":
    warnings.simplefilter("once", UserWarning)
    main()

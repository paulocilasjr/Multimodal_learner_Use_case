# run_ag_multimodal_with_zips_tmpclean.py
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    confusion_matrix,
)

# ========= USER CONFIG =========
CSV_PATH       = "../preprocessed_dataset/hancock_multimodal_mult_imgs_recurr_only.csv"
LABEL_COL      = "label_recurrence"             # e.g. ['no_recurrence','recurrence']
POS_LABEL      = "recurrence"                   # positive class name

IMAGE_COLUMNS  = ["image_path", "cd3_image_path", "cd8_image_path"]
IMAGES_ZIPS    = [
    "../datasets/PDL1_images.zip",
    "../datasets/CD3_images.zip",
    "../datasets/CD8_images.zip",
]
IMAGE_FOLDERS  = ["../datasets"]                # used if paths are already valid or to join basenames

RANDOM_SEED    = 42
SPLIT_PROBS    = (0.7, 0.1, 0.2)                # train, val, test
VERBOSITY      = 3
EVAL_METRIC    = "roc_auc"
TIME_LIMIT_S   = None                           # e.g., 3600
# SAVE_PATH    = None                           # let AG choose default save dir

# ========= HELPERS =========
def create_stratified_random_split(
    df: pd.DataFrame,
    label_column: str,
    split_probabilities: Tuple[float, float, float],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p_train, p_val, p_test = split_probabilities
    assert abs(p_train + p_val + p_test - 1.0) < 1e-8, "split_probabilities must sum to 1"

    rng = np.random.RandomState(int(random_state))
    df = df.copy()
    df["_split"] = 0

    for _, grp in df.groupby(label_column, dropna=False):
        idx = grp.sample(frac=1.0, random_state=rng.randint(0, 10**9)).index
        n = len(idx)
        n_train = int(round(n * p_train))
        n_val   = int(round(n * p_val))
        n_train = max(0, min(n, n_train))
        n_val   = max(0, min(n - n_train, n_val))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]
        df.loc[val_idx,  "_split"] = 1
        df.loc[test_idx, "_split"] = 2

    df_train = df[df["_split"] == 0].drop(columns=["_split"])
    df_val   = df[df["_split"] == 1].drop(columns=["_split"])
    df_test  = df[df["_split"] == 2].drop(columns=["_split"])
    return df_train, df_val, df_test

def _specificity_binary(y_true01, y_pred01) -> float:
    cm = confusion_matrix(y_true01, y_pred01, labels=[0,1])
    if cm.shape != (2,2):
        return float("nan")
    tn, fp = cm[0,0], cm[0,1]
    denom = (tn + fp)
    return float(tn / denom) if denom else float("nan")

def compute_metrics_binary_labels(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    pos_label: str,
) -> Dict[str, float]:
    y_true01 = (y_true_labels == pos_label).astype(int)
    y_pred01 = (y_pred_labels == pos_label).astype(int)

    acc  = accuracy_score(y_true01, y_pred01)
    prec = precision_score(y_true01, y_pred01, zero_division=0)
    rec  = recall_score(y_true01, y_pred01, zero_division=0)
    f1   = f1_score(y_true01, y_pred01, zero_division=0)
    mcc  = matthews_corrcoef(y_true01, y_pred01) if np.unique(y_true01).size > 1 else float("nan")
    spec = _specificity_binary(y_true01, y_pred01)

    roc = pr = ll = float("nan")
    if y_pred_proba is not None:
        P = np.asarray(y_pred_proba)
        if P.ndim == 1:
            pos_scores = P
            P2 = np.column_stack([1 - P, P])
        else:
            pos_scores = P[:, -1] if P.shape[1] >= 2 else P.ravel()
            P2 = P if P.shape[1] == 2 else np.column_stack([1 - pos_scores, pos_scores])

        if np.unique(y_true01).size > 1:
            try:
                roc = roc_auc_score(y_true01, pos_scores)
            except Exception:
                roc = float("nan")
            try:
                pr = average_precision_score(y_true01, pos_scores)
            except Exception:
                pr = float("nan")
        try:
            ll = log_loss(y_true01, P2, labels=[0,1])
        except Exception:
            ll = float("nan")

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Specificity": spec,
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "LogLoss": ll,
        "MCC": mcc,
    }

def _fmt_metrics_block(name: str, d: Dict[str, float]) -> str:
    keys = ["Accuracy","Precision","Recall","F1-score","Specificity","ROC-AUC","PR-AUC","LogLoss","MCC"]
    lines = [f"\n=== {name} ==="]
    for k in keys:
        v = d.get(k, float("nan"))
        try:
            lines.append(f"{k:12s}: {v:.6f}")
        except Exception:
            lines.append(f"{k:12s}: {v}")
    return "\n".join(lines)

def _extract_zips_to_temp(zips: List[str]) -> Tuple[str, List[Path]]:
    """Extract each zip to a unique temp subfolder; return (root_temp_dir, list_of_subdirs)."""
    if not zips:
        return "", []
    root = tempfile.mkdtemp(prefix="imgzips_")
    out_dirs: List[Path] = []
    for z in zips:
        zpath = Path(z)
        if not zpath.exists():
            print(f"[WARN] ZIP not found: {z}")
            continue
        dest = Path(root) / zpath.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)
        out_dirs.append(dest)
    return root, out_dirs

def _index_images(roots: List[Path]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    exts = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp"}
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                idx[p.name.lower()] = p.resolve()
    return idx

def _try_make_path(val: str, roots: List[Path]) -> Optional[Path]:
    if not isinstance(val, str) or not val:
        return None
    p = Path(val)
    if p.exists():
        return p.resolve()
    for r in roots:
        cand = (r / val)
        if cand.exists():
            return cand.resolve()
    return None

def _resolve_image_columns(
    df: pd.DataFrame,
    image_cols: List[str],
    zip_roots: List[Path],
    folder_roots: List[str],
) -> pd.DataFrame:
    df = df.copy()
    folder_roots_paths = [Path(fr) for fr in folder_roots if Path(fr).exists()]
    zip_index = _index_images(zip_roots) if zip_roots else {}

    missing_rows = set()
    for col in image_cols:
        if col not in df.columns:
            print(f"[WARN] Image column missing in CSV: {col}")
            continue

        def _resolver(v) -> Optional[str]:
            p = _try_make_path(v, folder_roots_paths)
            if p is not None:
                return str(p)
            base = Path(v).name.lower() if isinstance(v, str) else ""
            if base and base in zip_index:
                return str(zip_index[base])
            return None

        resolved = df[col].astype(str).map(_resolver)
        miss_mask = resolved.isna()
        if miss_mask.any():
            nmiss = int(miss_mask.sum())
            print(f"[WARN] {nmiss} rows with missing image path in column '{col}'. They will be dropped.")
            missing_rows.update(df.index[miss_mask].tolist())
        df[col] = resolved

    if missing_rows:
        df = df.drop(index=list(missing_rows)).reset_index(drop=True)
        print(f"[INFO] Dropped {len(missing_rows)} rows due to missing image paths across image columns.")

    return df

# ========= MAIN =========
def main():
    # 1) Load CSV
    df = pd.read_csv(CSV_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in CSV.")

    # 2) Extract image ZIPs into a *temporary* dir, resolve image paths, and CLEAN UP afterwards.
    temp_root = ""
    extracted_dirs: List[Path] = []
    try:
        temp_root, extracted_dirs = _extract_zips_to_temp(IMAGES_ZIPS)
        df = _resolve_image_columns(df, IMAGE_COLUMNS, extracted_dirs, IMAGE_FOLDERS)

        # 3) Splits
        train_df, val_df, test_df = create_stratified_random_split(
            df=df,
            label_column=LABEL_COL,
            split_probabilities=SPLIT_PROBS,
            random_state=RANDOM_SEED,
        )
        print(f"Rows → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # 4) Train MultiModalPredictor (NO 'presets' passed — uses AutoGluon default)
        from autogluon.multimodal import MultiModalPredictor

        predictor = MultiModalPredictor(
            label=LABEL_COL,
            problem_type="binary",
            eval_metric=EVAL_METRIC,
            verbosity=VERBOSITY,
            # positive_label=POS_LABEL,
            # path=SAVE_PATH,  # let AutoGluon choose default path
        )

        fit_kwargs = dict(
            train_data=train_df,
            tuning_data=val_df,
            time_limit=TIME_LIMIT_S,
            seed=RANDOM_SEED,
        )
        predictor = predictor.fit(**fit_kwargs)

        # 5) Predict helper
        def _predict_block(df_any: pd.DataFrame):
            y_true = df_any[LABEL_COL].to_numpy()
            pred = predictor.predict(df_any)
            y_pred = pred.to_numpy() if hasattr(pred, "to_numpy") else np.asarray(pred)

            proba = predictor.predict_proba(df_any)
            if isinstance(proba, pd.DataFrame):
                cols = list(proba.columns)
                if POS_LABEL in cols:
                    if len(cols) == 2:
                        proba_arr = proba[[c for c in cols if c != POS_LABEL] + [POS_LABEL]].to_numpy()
                    else:
                        proba_arr = proba[POS_LABEL].to_numpy()
                else:
                    proba_arr = proba.to_numpy()
            else:
                proba_arr = np.asarray(proba)

            return y_true, y_pred, proba_arr

        y_tr, yhat_tr, p_tr = _predict_block(train_df)
        y_va, yhat_va, p_va = _predict_block(val_df)
        y_te, yhat_te, p_te = _predict_block(test_df)

        # 6) Metrics
        m_tr = compute_metrics_binary_labels(y_tr, yhat_tr, p_tr, POS_LABEL)
        m_va = compute_metrics_binary_labels(y_va, yhat_va, p_va, POS_LABEL)
        m_te = compute_metrics_binary_labels(y_te, yhat_te, p_te, POS_LABEL)

        print(_fmt_metrics_block("Train", m_tr))
        print(_fmt_metrics_block("Validation", m_va))
        print(_fmt_metrics_block("Test", m_te))

    finally:
        # 7) ALWAYS CLEAN UP the temporary extraction directory (no artifacts left)
        if temp_root and Path(temp_root).exists():
            try:
                shutil.rmtree(temp_root)
                print(f"[CLEANUP] Removed temporary image extraction directory: {temp_root}")
            except Exception as e:
                print(f"[CLEANUP][WARN] Could not remove temp dir {temp_root}: {e}")

if __name__ == "__main__":
    main()


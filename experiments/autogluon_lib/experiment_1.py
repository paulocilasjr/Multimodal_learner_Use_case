#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import roc_auc_score

# ---------- CONFIG (edit if needed) ----------
# Run from: src/
FEATURES_DIR = Path("../../datasets")  # where your train/test paper tables live
TRAIN_FILE   = FEATURES_DIR / "train_paper_recurrence_in_img.csv"  # built earlier (with cd3/cd8_image_path)
TEST_FILE    = FEATURES_DIR / "test_paper_recurrence_in_img.csv"
LABEL        = "target"
PATIENT_COL  = "patient_id"
IMAGE_COLS   = ["cd3_image_path", "cd8_image_path"]  # only CD3/CD8 images
VAL_FRACTION = 0.125    # ~12.5% like earlier runs
SEED         = 42
OVERSAMPLE   = True     # approximate SMOTE by duplicating minority rows in TRAIN only
# ---------------------------------------------

def split_train_val(df_full, label_col, frac=0.125, seed=42):
    return train_test_split(
        df_full, test_size=frac, random_state=seed, stratify=df_full[label_col]
    )

def simple_impute(df: pd.DataFrame, image_cols):
    """Impute numeric=mean, categorical(mode). Leave image paths untouched."""
    df = df.copy()
    # identify numeric vs non-numeric (excluding label/patient/image cols)
    exclude = set([LABEL, PATIENT_COL] + image_cols)
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]

    if num_cols:
        imp_n = SimpleImputer(strategy="mean")
        df[num_cols] = imp_n.fit_transform(df[num_cols])
    if cat_cols:
        imp_c = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_c.fit_transform(df[cat_cols])

    return df

def oversample_minority(df: pd.DataFrame, label=LABEL, seed=SEED):
    """Duplicate minority-class rows to balance classes (train only)."""
    counts = df[label].value_counts()
    if len(counts) != 2:
        return df
    maj = counts.idxmax(); mino = counts.idxmin()
    gap = counts.max() - counts.min()
    if gap <= 0:
        return df
    rng = np.random.RandomState(seed)
    dup = df[df[label] == mino].sample(n=gap, replace=True, random_state=seed)
    out = pd.concat([df, dup], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[info] oversampled minority {int(counts[mino])} → {int(counts[mino]+gap)} (train rows {len(df)} → {len(out)})")
    return out

def pos_proba(proba):
    """Extract positive-class probabilities robustly."""
    if isinstance(proba, pd.DataFrame):
        # columns often '0' and '1' as strings
        if "1" in proba.columns:
            return proba["1"].to_numpy()
        return proba.iloc[:, -1].to_numpy()
    arr = np.asarray(proba)
    if arr.ndim == 1:
        return arr
    return arr[:, -1]

def eval_auc(name, predictor, df):
    y = df[LABEL].astype(int).to_numpy()
    p = pos_proba(predictor.predict_proba(df))
    auc = roc_auc_score(y, p)
    print(f"[{name}] ROC-AUC: {auc:.4f}")

def main():
    # 1) Load your prebuilt tables (already filtered like the paper, and with cd3/cd8 image paths)
    train_full = pd.read_csv(TRAIN_FILE)
    test_df    = pd.read_csv(TEST_FILE)

    # sanity: drop CD3/CD8 count columns if present (we're using images instead)
    drop_cols = {"cd3_z","cd3_inv","cd8_z","cd8_inv"}
    keep = [c for c in train_full.columns if c not in drop_cols]
    train_full = train_full[keep]
    test_df    = test_df[[c for c in test_df.columns if c not in drop_cols]]

    # ensure label type
    train_full[LABEL] = pd.to_numeric(train_full[LABEL], errors="raise").astype(int)
    test_df[LABEL]    = pd.to_numeric(test_df[LABEL], errors="raise").astype(int)

    # 2) Make a small validation split from TRAIN (paper used only train/test; this is just for early stopping)
    train_df, val_df = split_train_val(train_full, LABEL, VAL_FRACTION, SEED)

    # 3) Impute to match PyCaret/simple behavior (mean/mode), images untouched
    train_df = simple_impute(train_df, IMAGE_COLS)
    val_df   = simple_impute(val_df,   IMAGE_COLS)
    test_df  = simple_impute(test_df,  IMAGE_COLS)

    # 4) Oversample minority in TRAIN to approximate SMOTE effect in a multimodal setting
    if OVERSAMPLE:
        train_df = oversample_minority(train_df, LABEL, SEED)

    # 5) Train AutoMM (MultimodalPredictor) with explicit image columns
    predictor = MultiModalPredictor(
        label=LABEL,
        problem_type="binary",
        eval_metric="roc_auc",
        verbosity=3,
    )
    for c in ["cd3_image_path", "cd8_image_path"]:
        train_df[c] = train_df[c].astype(str)
        val_df[c]   = val_df[c].astype(str)
        test_df[c]  = test_df[c].astype(str)

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        seed=SEED,
        time_limit=None,
        hyperparameters=None,  # you can pass a dict to choose a backbone/augmentations
    )

    # 6) Print ROC-AUC on TRAIN / VAL / TEST
    eval_auc("TRAIN", predictor, train_df)
    eval_auc("VAL",   predictor, val_df)
    eval_auc("TEST",  predictor, test_df)

if __name__ == "__main__":
    main()


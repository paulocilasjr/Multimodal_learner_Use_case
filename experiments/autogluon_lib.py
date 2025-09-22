# run_ag_from_single_csv.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict

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
from sklearn.preprocessing import label_binarize

# ====== CONFIG (edit these) ======
CSV_PATH      = "../preprocessed_dataset/hancock_multimodal_mult_imgs_recurr_only.csv"
LABEL_COL     = "label_recurrence"
RANDOM_SEED   = 42
SPLIT_PROBS   = (0.7, 0.1, 0.2)   # train, val, test — must sum to 1.0
VERBOSITY     = 3
PRESETS       = "medium"          # 'medium' (fast prototyping) by default
TIME_LIMIT_S  = None              # e.g. 3600
IMAGE_COLUMNS: Optional[List[str]] = None  # e.g. ["image_path"] to trigger MultiModalPredictor

# ====== Reproducibility ======
def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# ====== UTIL: stratified split like in Multimodal Learner ======
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

    # per-class shuffle and allocation
    for cls, grp in df.groupby(label_column, dropna=False):
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

# ====== METRICS ======
def _specificity_binary(y_true, y_pred) -> float:
    labels = np.unique(y_true)
    if labels.size != 2:
        return float("nan")
    pos = labels.max()
    yb_true = (np.asarray(y_true) != pos).astype(int)  # 1 = negative class
    yb_pred = (np.asarray(y_pred) != pos).astype(int)
    cm = confusion_matrix(yb_true, yb_pred, labels=[0,1])
    tn, fp = cm[1,1], cm[1,0]  # treat "negative" as positive class in this trick
    denom = (tn + fp)
    return float(tn / denom) if denom else float("nan")

def _specificity_multiclass(y_true, y_pred, classes) -> float:
    specs = []
    for c in classes:
        yt = (np.asarray(y_true) == c).astype(int)
        yp = (np.asarray(y_pred) == c).astype(int)
        cm = confusion_matrix(yt, yp, labels=[0,1])
        tn, fp = cm[0,0], cm[0,1]
        denom = (tn + fp)
        specs.append(float(tn / denom) if denom else float("nan"))
    specs = [s for s in specs if not np.isnan(s)]
    return float(np.mean(specs)) if specs else float("nan")

def _proba_to_pos_scores(y_proba: np.ndarray) -> np.ndarray:
    """Return the positive-class scores for binary; else the max class prob for multiclass."""
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        return y_proba.reshape(-1)
    return np.max(y_proba, axis=1)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    is_binary = classes.size == 2
    if is_binary:
        pos_label = classes.max()
        avg = "binary"
        avg_kwargs = dict(pos_label=pos_label)
    else:
        avg = "macro"
        avg_kwargs = {}

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0, **avg_kwargs)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0, **avg_kwargs)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0, **avg_kwargs)
    mcc  = matthews_corrcoef(y_true, y_pred) if classes.size > 1 else float("nan")

    # Specificity
    spec = _specificity_binary(y_true, y_pred) if is_binary else _specificity_multiclass(y_true, y_pred, classes)

    # AUCs & LogLoss need probabilities
    roc = pr = ll = float("nan")
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if is_binary:
            pos = classes.max()
            y_bin = (y_true == pos).astype(int)
            pos_scores = _proba_to_pos_scores(y_proba)
            if np.unique(y_bin).size > 1:
                roc = roc_auc_score(y_bin, pos_scores)
                pr  = average_precision_score(y_bin, pos_scores)
            # for logloss, need 2 columns aligned to label order [neg,pos]
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                pp = np.column_stack([1 - pos_scores, pos_scores])
            else:
                pp = y_proba
            ll = log_loss(y_true, pp, labels=classes)
        else:
            # Multiclass: macro-ovr AUCs
            Y = label_binarize(y_true, classes=classes)
            P = y_proba if y_proba.ndim == 2 else np.column_stack([1 - y_proba, y_proba])
            if P.shape[1] == len(classes):
                try:
                    roc = roc_auc_score(Y, P, average="macro", multi_class="ovr")
                except Exception:
                    roc = float("nan")
                try:
                    pr = average_precision_score(Y, P, average="macro")
                except Exception:
                    pr = float("nan")
                ll = log_loss(y_true, P, labels=classes)

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

# ====== MAIN ======
def main():
    set_seeds(RANDOM_SEED)

    # Load
    df = pd.read_csv(CSV_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in CSV.")

    # Split
    train_df, val_df, test_df = create_stratified_random_split(
        df=df,
        label_column=LABEL_COL,
        split_probabilities=SPLIT_PROBS,
        random_state=RANDOM_SEED,
    )
    print(f"Rows → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Choose predictor: Tabular by default; MultiModal if IMAGE_COLUMNS provided
    use_mmp = IMAGE_COLUMNS is not None and len(IMAGE_COLUMNS) > 0
    if use_mmp:
        from autogluon.multimodal import MultiModalPredictor
        predictor = MultiModalPredictor(
            label=LABEL_COL,
            problem_type=None,
            verbosity=VERBOSITY,
        )
        fit_kwargs = dict(
            train_data=train_df,
            tuning_data=val_df,
            presets=PRESETS,
            seed=RANDOM_SEED,        # valid for MMP.fit
            time_limit=TIME_LIMIT_S,
        )
    else:
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor(
            label=LABEL_COL,
            verbosity=VERBOSITY,
        )
        fit_kwargs = dict(
            train_data=train_df,
            tuning_data=val_df,      # prevents internal re-split
            presets=PRESETS,
            time_limit=TIME_LIMIT_S,
        )

    # Train
    predictor = predictor.fit(**fit_kwargs)

    # Predictions for each split (labels + proba)
    def _predict_split(df_any: pd.DataFrame):
        y_true = df_any[LABEL_COL].values
        pred = predictor.predict(df_any)
        y_pred = pred.values if hasattr(pred, "values") else np.asarray(pred)
        try:
            proba = predictor.predict_proba(df_any)
            y_proba = proba.to_numpy() if hasattr(proba, "to_numpy") else np.asarray(proba)
        except Exception:
            y_proba = None
        return y_true, y_pred, y_proba

    y_tr, yhat_tr, p_tr = _predict_split(train_df)
    y_va, yhat_va, p_va = _predict_split(val_df)
    y_te, yhat_te, p_te = _predict_split(test_df)

    # Compute metric panels
    m_tr = compute_metrics(y_tr, yhat_tr, p_tr)
    m_va = compute_metrics(y_va, yhat_va, p_va)
    m_te = compute_metrics(y_te, yhat_te, p_te)

    # Print
    print(_fmt_metrics_block("Train", m_tr))
    print(_fmt_metrics_block("Validation", m_va))
    print(_fmt_metrics_block("Test", m_te))

if __name__ == "__main__":
    main()

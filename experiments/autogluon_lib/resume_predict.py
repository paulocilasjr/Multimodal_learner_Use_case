#!/usr/bin/env python3
import os, argparse, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from autogluon.multimodal import MultiModalPredictor

def positive_class_proba(y_proba):
    if hasattr(y_proba, "values") and hasattr(y_proba, "columns"):
        cols = list(y_proba.columns)
        # prefer '1' if present, else last column
        pos = "1" if "1" in cols else cols[-1]
        return y_proba[pos].to_numpy()
    arr = np.asarray(y_proba)
    if arr.ndim == 2:
        return arr[:, -1]
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor_path", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--label", default="target")
    ap.add_argument("--out_csv", default="metrics_mm_resume.csv")
    args = ap.parse_args()

    # Keep all caches/temp off NFS
    os.environ.setdefault("TMPDIR", "/tmp")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf")
    os.environ.setdefault("HF_HOME", "/tmp/hf")

    pred = MultiModalPredictor.load(args.predictor_path)

    # Make prediction dataloaders single-threaded to avoid NFS locks
    try:
        pred._config.data.num_workers = 0              # AG 1.3/1.4 style
    except Exception:
        pass
    try:
        pred._config.env.per_gpu_batch_size = max(8, int(pred._config.env.per_gpu_batch_size))
    except Exception:
        pass

    df_tr = pd.read_csv(args.train_csv, sep="\t" if args.train_csv.endswith(".tsv") else ",")
    df_va = pd.read_csv(args.val_csv,   sep="\t" if args.val_csv.endswith(".tsv")   else ",")
    df_te = pd.read_csv(args.test_csv,  sep="\t" if args.test_csv.endswith(".tsv")  else ",")

    p_tr = positive_class_proba(pred.predict_proba(df_tr))
    p_va = positive_class_proba(pred.predict_proba(df_va))
    p_te = positive_class_proba(pred.predict_proba(df_te))

    y_tr = df_tr[args.label].to_numpy()
    y_va = df_va[args.label].to_numpy()
    y_te = df_te[args.label].to_numpy()

    m = {
        "roc_auc_train": float(roc_auc_score(y_tr, p_tr)),
        "roc_auc_val":   float(roc_auc_score(y_va, p_va)),
        "roc_auc_test":  float(roc_auc_score(y_te, p_te)),
    }
    print(m)
    pd.DataFrame([m]).to_csv(args.out_csv, index=False)
    print(f"[info] wrote {args.out_csv}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Multimodal AutoGluon Runner (tabular + CD3 / CD8 images)

Fixes included:
- Preset alias mapping: 'medium'->'medium_quality', 'high'->'high_quality'
- Robust handling of missing/broken image paths via 'data.image.missing_value_strategy=skip'
- Simple imputation (mean for numeric, mode for categorical)
- Seed sweep + metrics saving
- Local tmp dirs to avoid NFS cleanup chatter
- torch.set_float32_matmul_precision('high') for Tensor Cores

Usage:
  python experiment_2.py \
    --train_csv ../../datasets/train_paper_recurrence_in_img_all.csv \
    --test_csv  ../../datasets/test_paper_recurrence_in_img_all.csv \
    --image_cols cd3_image_path cd8_image_path \
    --label target \
    --preset high \
    --val_frac 0.125 \
    --seeds 0 1 2 3 4 \
    --out_dir ./AutogluonModels/ag-mm-high-sweep \
    --save_metrics
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from autogluon.multimodal import MultiModalPredictor

# ------------------------------- Presets & defaults -------------------------------

# Accept legacy names and map to AutoMM internal names
PRESET_ALIASES = {
    "medium": "medium_quality",
    "high": "high_quality",
    "best_quality": "best_quality",
    "medium_quality": "medium_quality",
    "high_quality": "high_quality",
}

# Sensible defaults to avoid training on broken images
DEFAULT_HPARAMS = {
    "data.image.missing_value_strategy": "skip",  # skip samples with missing/unopenable images
    # feel free to uncomment if you want a bit more training
    # "optimization.max_epochs": 15,
    # "optimization.patience": 5,
}

# Columns that sometimes sneak in (but we’re using images instead)
COUNT_COLS_TO_DROP = {"cd3_z", "cd3_inv", "cd8_z", "cd8_inv"}


# ------------------------------- Helpers -------------------------------

def set_env_fast_tmp():
    """Reduce noisy NFS temp warnings by using local tmp when available."""
    os.environ.setdefault("TMPDIR", "/dev/shm")
    os.environ.setdefault("MPLCONFIGDIR", "/dev/shm/mpl")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def pos_proba(y_proba) -> np.ndarray:
    """Extract positive-class probabilities robustly from AutoGluon outputs."""
    if isinstance(y_proba, pd.DataFrame):
        if "1" in y_proba.columns:
            return y_proba["1"].to_numpy()
        return y_proba.iloc[:, -1].to_numpy()
    arr = np.asarray(y_proba)
    if arr.ndim == 1:
        return arr
    return arr[:, -1]


def impute_simple(df: pd.DataFrame, label: str, pid_col: str, image_cols: List[str]) -> pd.DataFrame:
    """Mean-impute numeric, mode-impute non-numeric (images untouched)."""
    df = df.copy()
    protected = set([label, pid_col] + list(image_cols))
    num_cols = [c for c in df.columns if c not in protected and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in protected and not pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    if cat_cols:
        mode_vals = df[cat_cols].mode(dropna=True)
        if not mode_vals.empty:
            df[cat_cols] = df[cat_cols].fillna(mode_vals.iloc[0])
        else:
            for c in cat_cols:
                df[c] = df[c].fillna("")
    return df


def ensure_image_cols(df: pd.DataFrame, image_cols: List[str], split_name: str):
    """Coerce image columns to string and warn if files are missing."""
    import os as _os
    for c in image_cols:
        if c not in df.columns:
            raise KeyError(f"[{split_name}] Missing image column: {c}")
        df[c] = df[c].astype(str)
        missing = (~df[c].apply(_os.path.exists)).sum()
        if missing:
            print(f"[warn] {split_name}: {missing} image paths in '{c}' do not exist on disk.")


def drop_count_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in COUNT_COLS_TO_DROP]
    return df[keep].copy()


def eval_auc(tag: str, predictor: MultiModalPredictor, df: pd.DataFrame, label: str) -> float:
    y_true = df[label].astype(int).to_numpy()
    probs = pos_proba(predictor.predict_proba(df))
    auc = roc_auc_score(y_true, probs)
    print(f"[{tag}] ROC-AUC: {auc:.4f}")
    return float(auc)


def run_once(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Single training/eval run; returns metrics dict."""
    # Better matmul on A30/A100 for speed
    torch.set_float32_matmul_precision("high")

    # Load
    train_full = pd.read_csv(cfg["train_csv"])
    test_df = pd.read_csv(cfg["test_csv"])

    # Clean
    train_full = drop_count_cols(train_full)
    test_df = drop_count_cols(test_df)

    # Label coercion
    if cfg["label"] not in train_full.columns:
        raise KeyError(f"Label column '{cfg['label']}' missing in TRAIN CSV.")
    if cfg["label"] not in test_df.columns:
        raise KeyError(f"Label column '{cfg['label']}' missing in TEST CSV.")
    train_full[cfg["label"]] = pd.to_numeric(train_full[cfg["label"]], errors="raise").astype(int)
    test_df[cfg["label"]] = pd.to_numeric(test_df[cfg["label"]], errors="raise").astype(int)

    # Sanity for image cols
    for c in cfg["image_cols"]:
        if c not in train_full.columns:
            raise KeyError(f"Image column '{c}' missing in TRAIN CSV.")
        if c not in test_df.columns:
            raise KeyError(f"Image column '{c}' missing in TEST CSV.")

    # Stratified train/val split
    train_df, val_df = train_test_split(
        train_full,
        test_size=cfg["val_frac"],
        random_state=cfg["seed"],
        stratify=train_full[cfg["label"]],
    )
    print(f"Rows → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Imputation (mean/mode)
    train_df = impute_simple(train_df, cfg["label"], cfg["pid_col"], cfg["image_cols"])
    val_df = impute_simple(val_df, cfg["label"], cfg["pid_col"], cfg["image_cols"])
    test_df = impute_simple(test_df, cfg["label"], cfg["pid_col"], cfg["image_cols"])

    # Ensure image cols ok (warn on missing paths)
    ensure_image_cols(train_df, cfg["image_cols"], "train")
    ensure_image_cols(val_df, cfg["image_cols"], "val")
    ensure_image_cols(test_df, cfg["image_cols"], "test")

    # Map preset alias to AutoMM
    automm_preset = PRESET_ALIASES.get(cfg["preset"], cfg["preset"])

    predictor = MultiModalPredictor(
        label=cfg["label"],
        problem_type="binary",
        eval_metric="roc_auc",
        path=cfg["out_dir"],
        verbosity=3,
    )

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        seed=cfg["seed"],
        presets=automm_preset,           # mapped preset
        hyperparameters=DEFAULT_HPARAMS, # skip broken/missing images
        time_limit=None,
    )

    auc_train = eval_auc("TRAIN", predictor, train_df, cfg["label"])
    auc_val = eval_auc("VAL", predictor, val_df, cfg["label"])
    auc_test = eval_auc("TEST", predictor, test_df, cfg["label"])

    return {
        "seed": cfg["seed"],
        "preset": cfg["preset"],
        "automm_preset": automm_preset,
        "auc_train": auc_train,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "models_path": str(cfg["out_dir"]),
    }


# ------------------------------- CLI -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run AutoGluon MultiModal on tabular + CD3/CD8 images.")
    p.add_argument("--train_csv", type=Path, required=True, help="Path to train CSV (e.g., *_img_all.csv)")
    p.add_argument("--test_csv", type=Path, required=True, help="Path to test  CSV (e.g., *_img_all.csv)")
    p.add_argument("--label", type=str, default="target", help="Label column (default: target)")
    p.add_argument("--pid_col", type=str, default="patient_id", help="Patient ID column (default: patient_id)")
    p.add_argument(
        "--image_cols",
        type=str,
        nargs="+",
        required=True,
        help="Image column(s), e.g., cd3_image_path cd8_image_path",
    )
    p.add_argument("--val_frac", type=float, default=0.125, help="Validation fraction from TRAIN (default: 0.125)")
    p.add_argument(
        "--preset",
        type=str,
        default="high",
        choices=["medium", "high", "best_quality", "medium_quality", "high_quality"],
        help="Training preset. Aliases: medium->medium_quality, high->high_quality.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (ignored if --seeds provided)")
    p.add_argument("--seeds", type=int, nargs="*", help="Optional multiple seeds, e.g., --seeds 0 1 2 3 4")
    p.add_argument("--out_dir", type=Path, default=Path("AutogluonModels/ag-mm-run"),
                   help="Directory to save models/checkpoints")
    p.add_argument("--save_metrics", action="store_true", help="Save metrics to metrics.csv and metrics.json")
    return p.parse_args()


def main():
    set_env_fast_tmp()
    args = parse_args()

    base_cfg = dict(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        label=args.label,
        pid_col=args.pid_col,
        image_cols=args.image_cols,
        val_frac=args.val_frac,
        preset=args.preset,
    )

    results: List[Dict[str, Any]] = []

    if args.seeds:
        print(f"[info] Running seed sweep: {args.seeds} with preset='{args.preset}'")
        for seed in args.seeds:
            out_dir = args.out_dir / f"{args.preset}-seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = dict(base_cfg, seed=int(seed), out_dir=out_dir)
            res = run_once(cfg)
            results.append(res)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base_cfg, seed=int(args.seed), out_dir=args.out_dir)
        res = run_once(cfg)
        results.append(res)

    # Save metrics if requested
    if args.save_metrics:
        df = pd.DataFrame(results)
        csv_path = args.out_dir / "metrics.csv"
        json_path = args.out_dir / "metrics.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[info] Saved metrics to:\n  {csv_path}\n  {json_path}")

    # Seed sweep summary
    if len(results) > 1:
        print("\nSeed sweep summary (AUC TEST):")
        for r in results:
            print(f"  seed={r['seed']}: {r['auc_test']:.4f}  (train {r['auc_train']:.4f} | val {r['auc_val']:.4f})")
        mean = float(np.mean([r["auc_test"] for r in results]))
        std = float(np.std([r["auc_test"] for r in results]))
        print(f"  mean±std = {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()


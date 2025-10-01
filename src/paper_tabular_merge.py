#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, dtype={"patient_id": str})
    if "patient_id" not in df.columns:
        raise KeyError(f"`patient_id` not found in {path}")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    # Drop duplicate patient rows if any
    df = df.drop_duplicates(subset=["patient_id"], keep="first")
    return df

def build_features_table(features_dir: Path) -> pd.DataFrame:
    clinical      = load_csv(features_dir / "clinical.csv")
    pathological  = load_csv(features_dir / "pathological.csv")
    blood         = load_csv(features_dir / "blood.csv")
    icd           = load_csv(features_dir / "icd_codes.csv")
    cell_density  = load_csv(features_dir / "tma_cell_density.csv")

    # Outer merge to match training script behavior
    df = clinical.merge(pathological, on="patient_id", how="outer") \
                 .merge(blood,        on="patient_id", how="outer") \
                 .merge(icd,          on="patient_id", how="outer") \
                 .merge(cell_density, on="patient_id", how="outer")
    df = df.reset_index(drop=True)
    return df

def apply_target_filters(df_split: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Mirror the filtering/label encoding from the training script.
    - recurrence:
        keep (yes within 3y) OR (no with >=3y follow-up OR survival_status == "living")
        map to 1/0
    - survival_status:
        drop 'deceased not tumor specific'
        map to 1/0
    """
    if target == "recurrence":
        # keep only well-defined positives/negatives
        df = df_split[
            ((df_split["recurrence"] == "yes") & (df_split["days_to_recurrence"] <= 365 * 3)) |
            ((df_split["recurrence"] == "no") &
             ((df_split["days_to_last_information"] > 365 * 3) | (df_split["survival_status"] == "living")))
        ].copy()
        # string -> numeric target
        df["target"] = df["recurrence"].replace({"no": 0, "yes": 1}).astype(int)
        return df[["patient_id", "dataset", "target"]]

    elif target == "survival_status":
        df = df_split[~(df_split["survival_status_with_cause"] == "deceased not tumor specific")].copy()
        df["target"] = df["survival_status"].replace({"living": 0, "deceased": 1}).astype(int)
        return df[["patient_id", "dataset", "target"]]

    else:
        raise KeyError(f"Unsupported target: {target}")

def main():
    ap = argparse.ArgumentParser(description="Build single CSV used for training, with split column (0=train, 2=test).")
    ap.add_argument("features_directory", type=str, help="Path to directory with extracted features (CSVs).")
    ap.add_argument("split_json", type=str, help="Path to ONE dataset split JSON (e.g., dataset_split_in.json).")
    ap.add_argument("output_csv", type=str, help="Where to write the final CSV.")
    ap.add_argument("--target", choices=["recurrence", "survival_status"], required=True, help="Target to prepare.")
    args = ap.parse_args()

    features_dir = Path(args.features_directory)
    split_json   = Path(args.split_json)
    output_csv   = Path(args.output_csv)

    # 1) Build feature matrix (outer merge)
    X = build_features_table(features_dir)

    # 2) Load targets and chosen split JSON
    targets = load_csv(features_dir / "targets.csv")  # dtype & dedup handled
    if not split_json.exists():
        raise FileNotFoundError(f"Split file not found: {split_json}")

    split_df = pd.read_json(split_json, dtype={"patient_id": str})[["patient_id", "dataset"]]
    split_df["patient_id"] = split_df["patient_id"].astype(str).str.strip()

    # 3) Join split + targets, then apply same target-specific filtering as training script
    split_targets = split_df.merge(targets, on="patient_id", how="inner")
    split_targets = apply_target_filters(split_targets, args.target)

    # 4) Map dataset -> split codes (train=0, test=2)
    split_map = {"training": 0, "test": 2}
    split_targets["split"] = split_targets["dataset"].map(split_map).astype("Int64")

    # 5) Keep only patients that are in the chosen split after filtering, and attach features
    final = split_targets[["patient_id", "split", "target"]].merge(X, on="patient_id", how="inner")

    # Reorder columns: patient_id, split, target, then features
    cols = ["patient_id", "split", "target"] + [c for c in final.columns if c not in ("patient_id", "split", "target")]
    final = final[cols]

    # 6) Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)

    # Brief summary
    n_train = int((final["split"] == 0).sum())
    n_test  = int((final["split"] == 2).sum())
    print(f"Wrote {output_csv}")
    print(f"Rows: {len(final)} | Train (split=0): {n_train} | Test (split=2): {n_test}")
    print(f"Features (excluding patient_id/split/target): {final.shape[1] - 3}")

if __name__ == "__main__":
    main()


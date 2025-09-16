# Multimodal Learner · HANCOCK Use Case

> Build a **training-ready multimodal dataset** (tabular + image + free-text) from the HANCOCK layout and train an **AutoGluon** multimodal model.  
> The pipeline is opinionated about **labeling**, **leakage prevention**, **image/text mapping**, and **one-row-per-patient** integrity so it’s reproducible and safe to train.

---

## Table of Contents

- [Dataset Layout](#dataset-layout)
- [Pipeline Overview](#pipeline-overview)
- [Label Definitions](#label-definitions)
- [Differences vs. Paper/Original Repo](#differences-vs-paperoriginal-repo)
- [How to Run](#how-to-run)
- [Output Schema](#output-schema)
- [Troubleshooting](#troubleshooting)
- [Design Choices & Reproducibility](#design-choices--reproducibility)

---

## Dataset Layout

Expected repository structure:

.
├─ datasets/
│ ├─ PDL1_images.zip # images/...patient<digits>.png
│ ├─ StructuredData/
│ │ ├─ blood_data.json
│ │ ├─ clinical_data.json
│ │ └─ pathological_data.json
│ └─ TextData/
│ ├─ histories_english/ # files end with patient id (…patientNNN.txt)
│ ├─ reports_english/
│ ├─ surgery_descriptions_english/
│ ├─ icd_codes/
│ └─ ops_codes/
└─ src/
└─ build_hancock_tabular_raw.py

> We intentionally **exclude TMA cell-density** measurements. Images come from `PDL1_images.zip` and are referenced via a single `image_path` column.

---

## Pipeline Overview

**Script:** `src/build_hancock_tabular_raw.py`

1) **Merge structured JSON**
   - Prefixes columns:  
     `clinical_data.json → clinical__*`  
     `pathological_data.json → path__*`  
     `blood_data.json → long-form labs (see #2)`
   - Normalizes **patient ids** to **trailing digits** across sources (e.g., `CASE001 → 001`).

2) **Pivot blood labs to wide**
   - For each *(patient, analyte)*, chooses the measurement **closest to first treatment day** (*min abs* `blood__days_before_first_treatment`).
   - Produces numeric columns like `blood__Hemoglobin`, `blood__Platelets`, `blood__CRP`, …
   - Drops lab **unit** columns and other long-form fields.

3) **Ingest English-only text (no embeddings)**
   - Uses only English sources:  
     `histories_english → histories_english_text`  
     `reports_english → reports_english_text`  
     `surgery_descriptions_english → surgery_desc_english_text`  
     `icd_codes → icd_codes_text`  
     `ops_codes → ops_codes_text`
   - Concatenates multiple files per patient per source into a single string.
   - **Keeps raw text** in the CSV; embeddings are learned at training time.

4) **Map images from `PDL1_images.zip`**
   - Extracts patient id from filename suffix: `...patient265.png → 265`.
   - Stores **basename** in `image_path` (e.g., `TumorCenter_PDL1_block22_x1_y10_patient265.png`).
   - If multiple tiles exist, picks the **lexicographically first**.
   - **Drops patients without an image** (this pipeline trains image + tabular).

5) **Enforce one row per patient**
   - Structured merges deduplicated by patient id.
   - Blood wide-pivoted (one value per analyte).
   - Text concatenated per source.
   - One image path per patient.

6) **Create labels (two targets, strings)**
   - See [Label Definitions](#label-definitions).

7) **Automatic leakage prevention**
   - Drops **all** `clinical__days_to_*`, `clinical__progress_*`, `clinical__metastasis_*` (+ `*_locations`).
   - Drops raw target sources superseded by labels:  
     `clinical__survival_status`, `clinical__survival_status_with_cause`, `clinical__recurrence`.
   - Drops **`patient_id`** before saving the final CSV.

---

## Label Definitions

Two target columns are produced in the same CSV:

| Column             | Values                             | Rule |
|--------------------|-------------------------------------|------|
| `label_survival`   | `living` / `deceased`               | Derived from `clinical__survival_status`. Rows where `clinical__survival_status_with_cause == "deceased not tumor specific"` are **excluded** (dropped) for survival modeling. |
| `label_recurrence` | `recurrence` / `no_recurrence` / NaN | 3-year endpoint: **recurrence** if `clinical__recurrence == "yes"` **and** `clinical__days_to_recurrence ≤ 1095`. **no_recurrence** if `clinical__recurrence == "no"` **and** `clinical__days_to_last_information ≥ 1095`. Otherwise **NaN** (insufficient follow-up). Filter before training on this target. |

---

## Differences vs. Paper/Original Repo

This pipeline intentionally diverges to simplify **safe multimodal training**:

- **No TMA cell-density**: rely on `PDL1_images.zip` + `image_path`.
- **Text stays raw**: no precomputed embeddings; training handles tokenization/encoding.
- **Both labels in one CSV**: `label_survival` and `label_recurrence` generated together.
- **Cause-of-death filtering**: non-tumor-specific deaths excluded for survival (aligns with paper’s intent).
- **Explicit leakage removal**: timing/event fields and raw target sources are dropped.
- **Strict one-row-per-patient**: deterministic pivoting/concatenation rules.
- **No `patient_id` in output**: avoids identity-based overfitting.

> Need exact parity with the original preprocessing? Re-enable excluded inputs (e.g., TMA), mirror their text handling/embeddings, and match all inclusion/exclusion rules 1:1.

---

## How to Run

### 1) Build the CSV

From repo root:

```bash
python src/build_hancock_tabular_raw.py \
  --out_csv preprocessed_dataset/multimodal_merged_data.csv
```

> This writes a CSV containing:

- image_path

- label_survival (living / deceased)

- label_recurrence (recurrence / no_recurrence / NaN)

- Clinical/pathology features

- Wide blood labs

- English text blocks

> patient_id is dropped automatically in the final CSV.

### 2) (Optional) Filter per-task before training
```
python - <<'PY'
import pandas as pd
p = "preprocessed_dataset/multimodal_merged_data.csv"
df = pd.read_csv(p)
df[df["label_recurrence"].notna()].to_csv(
    "preprocessed_dataset/multimodal_merged_data_recurr_only.csv", index=False
)
df[df["label_survival"].notna()].to_csv(
    "preprocessed_dataset/multimodal_merged_data_survival_only.csv", index=False
)
print("recurrence:", df["label_recurrence"].value_counts(dropna=False).to_dict())
print("survival:", df["label_survival"].value_counts(dropna=False).to_dict())
PY
```

### 3) Train AutoGluon MultiModal
Recurrence example:

```
python gleam/tools/multimodallearner/multimodal_learner.py \
  --input_csv_train preprocessed_dataset/multimodal_merged_data.csv \
  --target_column label_recurrence \
  --image_column image_path \
  --output_csv metrics.csv \
  --output_json results.json \
  --output_html report.html \
  --images_zip datasets/PDL1_images.zip
```

> If you filtered earlier, swap input_csv_train to the *_recurr_only.csv.
Repeat similarly for label_survival.

# Output Schema

### Required for training

- image_path — basename inside PDL1_images.zip (trainer resolves paths)

- label_survival — living / deceased

- label_recurrence — recurrence / no_recurrence / NaN

### Features

- clinical__*, path__*

- blood__* (wide-pivoted analytes)

- Text blocks:

> histories_english_text

> reports_english_text

> surgery_desc_english_text

> icd_codes_text

> ops_codes_text

### Not present by design

- patient_id

- clinical__days_to_*, clinical__progress_*, clinical__metastasis_* (+ locations)

- clinical__survival_status, clinical__survival_status_with_cause, clinical__recurrence

- Any blood__*unit* columns

## Troubleshooting
“Input contains NaN” during training
You’re training on a label with unlabeled rows (e.g., insufficient follow-up for recurrence).
→ Filter to labeled rows (*_recurr_only.csv or *_survival_only.csv) before training.

Image mismatch / many patients dropped
Every image filename in PDL1_images.zip must end with patient<digits>.<ext>, e.g.:
images/TumorCenter_PDL1_block22_x1_y10_patient265.png
The builder extracts the trailing number to match the patient.

Transformers/Torch safetensors warning
If the environment enforces safetensors (older torch), the trainer prefers a safetensors text backbone (e.g., distilbert-base-uncased). Upgrading PyTorch ≥ 2.6 also resolves it.

Empty val/test due to fixed split
The trainer falls back to a stratified random split using --split_probabilities.

## Design Choices & Reproducibility
> Labels
Survival: living / deceased; non–tumor-specific deaths excluded.
Recurrence: 3-year endpoint; negatives require ≥ 3-year follow-up.

> Text
Kept as raw text so the chosen backbone controls tokenization/embedding.

> Labs
Selects measurement nearest to first treatment → standardized baseline.

> Leakage
Drops all post-outcome timing/event fields and raw target sources.
Drops patient_id to prevent identity leakage.

> Determinism
One row per patient by construction; downstream training accepts a fixed random seed.

## Citation & License

Please cite the HANCOCK dataset paper (Nature Communications, 2025) when using this pipeline/dataset in research.

This repository includes a LICENSE. Check the original dataset’s terms for redistribution/usage constraints.

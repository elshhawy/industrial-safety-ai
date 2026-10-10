# SH17 Data Preparation Report
**Industrial Safety AI — Data Engineering (Rizk)**

> Training taxonomy note (2026): this Kaggle exploration used an 8-class
> scheme. The repository training workflow now uses SH17 only with the six
> target classes in `IMPLEMENTATION_SPEC.md` / `training/README.md`
> (`person`, `Face-covering`, `glasses`, `gloves`, `helmet`, `suit`).
> Do not use the 8-class IDs below for `training/prepare_dataset.py`.

Reference: Kaggle Notebook — *Data Processing*, Version 4 (Save & Run All)

---

## 1. Exploration Summary

All steps below were executed on a Kaggle Notebook without downloading the dataset (14 GB compressed) to any local machine. Original data path:

```
/kaggle/input/datasets/mugheesahmad/sh17-dataset-for-ppe-detection/
```

Original structure: `images/`, `labels/` (YOLO), `voc_labels/` (VOC/XML), `meta-data/` (Pexels metadata, not relevant for this project), `train_files.txt`, `val_files.txt`.

---

## 2. Complete Class Map (Extracted Programmatically, Not Assumed)

No `classes.txt` or `data.yaml` file was available in the dataset. The class map was extracted by automated cross-referencing of 1,000 YOLO label files (numeric IDs) against their corresponding VOC/XML files (text names) for the same images:

| ID | Name | Status |
|----|------|--------|
| 0 | person | Required |
| 1 | ear | — |
| 2 | ear-mufs | — |
| 3 | face | — |
| 4 | face-guard | — |
| 5 | face-mask-medical | Required |
| 6 | foot | — |
| 7 | tools | — |
| 8 | glasses | Required |
| 9 | gloves | Required |
| 10 | helmet | Required |
| 11 | hands | — |
| 12 | head | — |
| 13 | medical-suit | Required (merged with 15) |
| 14 | shoes | Required |
| 15 | safety-suit | Required |
| 16 | safety-vest | Required |

---

## 3. Filtering and Merging

Nine irrelevant classes were excluded (`ear, ear-mufs, face, face-guard, foot, tools, hands, head`), retaining 8 final classes (7 PPE items + person). `medical-suit` (157 instances) was merged with `safety-suit` (240 instances) under the unified name **protective-suit** (397 instances, 255 images, with no overlap between the two).

> **Open decision for the Detection team**: "Protective suit" was not part of the original project safety equipment list. Keeping it costs nothing extra, but it can be removed later if the team decides it distracts the model without benefiting violation logic.

### Filtering Results (Corrected)

| Metric | Value |
|---|---|
| Total original SH17 images | 8,099 |
| **Images containing at least one required class** | **7,753** |
| Images fully excluded (no required class present) | 346 |
| Final class count after filtering and merging | 8 (7 PPE + person) |

> **Correction**: A previous report incorrectly stated 7,806. The confirmed number from actual execution is **7,753**. Merging `medical-suit` did not add new images to the valid set, because all `medical-suit` images already contained another required class (most likely `person`).

---

## 4. Instance Distribution per Class (After Final Filtering)

| ID | Class | Instances | Images | Assessment |
|----|-------|-----------|--------|------------|
| 0 | face-mask-medical | 670 | 395 | Weak |
| 1 | glasses | 1,945 | 1,588 | Sufficient |
| 2 | gloves | 2,790 | 1,321 | Sufficient |
| 3 | helmet | 927 | 466 | Weak |
| 4 | shoes | 4,560 | 1,570 | Sufficient |
| 5 | protective-suit | 397 | 255 | Weak |
| 6 | **safety-vest** | 530 | **213** | **Weakest** |
| 7 | person | 13,802 | 7,617 | Very sufficient |

**Key observation**: The most critical items for the project (`helmet` and `safety-vest`) are among the weakest in quantity. The ratio between `person` (7,617 images) and `safety-vest` (213 images) is approximately 36:1. The likely reason is that SH17 was designed for a manufacturing context rather than construction, where hard hats and high-visibility vests are more common. The general Ultralytics recommendation is a minimum of 1,500 images per class — all PPE items here fall below this threshold.

**Recommendation**: Supplement `helmet` and `safety-vest` from CHV or Roboflow PPE Detection later, noting that the `person` class is absent in these alternative sources (auto-labeling may be required before merging).

---

## 5. Quality Check and Cleaning Results

| Check | Sample/Scope | Result |
|---|---|---|
| Visual inspection of random sample (bounding boxes) | 12 images, boxes drawn and reviewed | ✅ Accurate and consistent, no notable errors |
| Duplicate detection (via hashing) | 7,753 images | **0 duplicates** |
| Corrupted image detection | 7,735 images (actually checked) | **0 corrupted images** |

> **Note**: The difference of 18 images between 7,753 (filtering) and 7,735 (corruption check) is due to images with non-`.jpg`/`.jpeg` extensions that were not covered by the check. The proportion is small (0.2%) and is not considered an issue, but warrants later confirmation.

---

## 6. Final Train / Val / Test Split

**Methodology**: The original `train_files.txt` was used as-is for training, while the original `val_files.txt` (1,545 images) was split into val + test using a **stratified** approach (based on the presence of rare classes) to ensure a balanced distribution of critical items (`helmet`, `safety-vest`, `protective-suit`, `face-mask-medical`) between the two sets, rather than relying on a purely random split.

### Final Numbers

| Split | Images | Proportion |
|---|---|---|
| **Train** | 6,208 | 80.1% |
| **Val** | 772 | 10.0% |
| **Test** | 773 | 10.0% |
| **Total** | 7,753 | 100% |

### Rare Class Distribution between Val and Test

| Class | Val | Test |
|---|---|---|
| face-mask-medical | 36 | 39 |
| helmet | 45 | 48 |
| protective-suit | 35 | 23 |
| safety-vest | 20 | 25 |

Distribution is balanced across all classes, with no class falling entirely into a single split.

### Saved File Paths (on Kaggle, Permanent Output — Version 4)

```
/kaggle/working/splits/
├── train/labels/       ← 6,208 label files
├── val/labels/         ← 772 label files
├── test/labels/        ← 773 label files
├── train_files.txt
├── val_files.txt
└── test_files.txt
```

Images themselves were not copied (used directly from `/kaggle/input`) to save space; `*_files.txt` files serve as the index mapping each image to its split.

---

## 7. Overall Status vs. Data Preparation Plan (8 Steps)

| # | Step | Status |
|---|------|--------|
| 1 | Download full SH17 and inspect structure | ✅ Complete |
| 2 | Filter classes to the required seven | ✅ Complete |
| 3 | Compute instances per class | ✅ Complete |
| 4 | Quality check on random sample | ✅ Complete |
| 5 | Data cleaning (Duplicates + Corrupted) | ✅ Complete (0 issues) |
| 6 | Rebuild Train/Val/Test Split | ✅ Complete |
| 7 | Merge supplementary CCTV sample | ⏳ **Action Item** — awaiting team filming |
| 8 | Final documentation | ✅ This document |

---

## 8. Recommendations and Next Steps

1. **Urgent priority**: Supplement `helmet` and `safety-vest` data from CHV or Roboflow before starting final training.
2. **Action item for the team**: Execute the supplementary CCTV filming plan (4 steps: select location, mount camera overhead, record video, extract frames) and merge into `val`/`test` primarily to measure real-world performance.
3. **Decision required from Detection team**: Keep `protective-suit` in the training scope or exclude it.
4. **For training**: Use paths from `/kaggle/working/splits/` directly via `data.yaml` with class ordering: `face-mask-medical(0), glasses(1), gloves(2), helmet(3), shoes(4), protective-suit(5), safety-vest(6), person(7)`.

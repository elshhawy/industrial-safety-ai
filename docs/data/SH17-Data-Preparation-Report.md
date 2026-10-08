# SH17 Data Preparation Report

**Industrial Safety AI: Data Engineering (Rizk)**

All numbers below were measured by running code on Kaggle (Data Processing notebook, Version 4) unless a line says otherwise.

---

## 1. Overview

- **Source dataset:** SH17 (`mugheesahmad/sh17-dataset-for-ppe-detection` on Kaggle), 8,099 images from Pexels, YOLO and VOC labels, 17 classes.
- **Where the work was done:** a Kaggle notebook, using the dataset as an input. The 14 GB archive was never downloaded to a personal machine.
- **Original structure:** `images/`, `labels/` (YOLO), `voc_labels/` (VOC XML), `meta-data/` (Pexels photo metadata, not used), `train_files.txt`, `val_files.txt`.
- **Result:** Dataset V1, with 8 classes and 7,753 images, split into train/val/test.

---

## 2. Class mapping (17 classes)

SH17 ships without a `classes.txt` or `data.yaml`. The mapping below was extracted by comparing 1,000 YOLO label files (numeric IDs) with the matching VOC files (class names) for the same images.

| ID | Name | Used in this project |
|----|------|----------------------|
| 0 | person | Yes |
| 1 | ear | No |
| 2 | ear-mufs | No |
| 3 | face | No |
| 4 | face-guard | No |
| 5 | face-mask-medical | Yes |
| 6 | foot | No |
| 7 | tools | No |
| 8 | glasses | Yes |
| 9 | gloves | Yes |
| 10 | helmet | Yes |
| 11 | hands | No |
| 12 | head | No |
| 13 | medical-suit | Yes (merged into `protective-suit`) |
| 14 | shoes | Yes |
| 15 | safety-suit | Yes |
| 16 | safety-vest | Yes |

---

## 3. Filtering and class merge

Nine classes were dropped (`ear`, `ear-mufs`, `face`, `face-guard`, `foot`, `tools`, `hands`, `head`), leaving 8 classes: 7 PPE classes plus `person`. `medical-suit` (157 instances) and `safety-suit` (240 instances) were merged into one class, `protective-suit` (397 instances in 255 images, no image contains both).

Final class order:

| New ID | Class | Original ID(s) |
|--------|-------|----------------|
| 0 | face-mask-medical | 5 |
| 1 | glasses | 8 |
| 2 | gloves | 9 |
| 3 | helmet | 10 |
| 4 | shoes | 14 |
| 5 | protective-suit | 15, 13 |
| 6 | safety-vest | 16 |
| 7 | person | 0 |

> **Open decision (Detection team):** protective suits were not in the original list of required PPE for the project. Keeping the class costs nothing, but it can be dropped if it does not help the violation logic.

### Images kept and excluded

| Metric | Value |
|---|---|
| Images in SH17 | 8,099 |
| Images with at least one of the 8 classes (kept) | **7,753** |
| Images with none of the 8 classes (excluded) | 346 |

The 346 excluded images contain none of the 8 classes, not even `person` (they most likely contain only classes such as `head`, `face`, `hands` or `tools`; this was not inspected). Where they came from:

| Original list | Excluded | Kept | Original total* | Excluded share |
|---|---|---|---|---|
| `train_files.txt` | 271 | 6,208 | 6,479 | 4.2% |
| `val_files.txt` | 75 | 1,545 | 1,620 | 4.6% |
| Neither list | 0 | 0 | 0 | n/a |

\*Original totals are derived (kept + excluded), not counted separately; they add up to 8,099.

Excluded images were removed from train and val in similar proportions, so the filtering did not bias either side.

---

## 4. Instances per class (after filtering)

| ID | Class | Instances | Images | Assessment |
|----|-------|-----------|--------|------------|
| 0 | face-mask-medical | 670 | 395 | Weak |
| 1 | glasses | 1,945 | 1,588 | OK |
| 2 | gloves | 2,790 | 1,321 | OK |
| 3 | helmet | 927 | 466 | Weak |
| 4 | shoes | 4,560 | 1,570 | OK |
| 5 | protective-suit | 397 | 255 | Weak |
| 6 | **safety-vest** | 530 | **213** | **Weakest** |
| 7 | person | 13,802 | 7,617 | Very strong |

**Key finding:** the two classes that matter most for this project, `helmet` and `safety-vest`, are among the weakest. `person` appears in 7,617 images and `safety-vest` in 213, a ratio of about 36 to 1. The likely cause is that SH17 targets manufacturing scenes, where helmets and high-visibility vests are less common than on construction sites. The Ultralytics training tips suggest 1,500 or more images per class; every PPE class here is below that. Training can still start from pretrained weights, but per-class results must be measured and reported separately.

**Recommendation:** add `helmet` and `safety-vest` images from CHV or Roboflow PPE Detection before final training. Those datasets usually have no `person` boxes, so mixing them in directly would teach the model that workers without a person box are normal. Either use them only for those classes, or auto-label `person` on them first.

---

## 5. Quality checks

| Check | Scope | Result |
|---|---|---|
| Visual inspection of bounding boxes | 12 random images, boxes drawn | Accurate, no obvious errors |
| Duplicate images (MD5 of file content) | 7,753 images | **0 duplicates** |
| Corrupted images (opened and verified) | 7,735 images | **0 corrupted** |

The duplicate check covered all 7,753 images, but the corruption check only covered 7,735. The 18 missing images were skipped because the check looked for `.jpg`/`.jpeg` files only. This is 0.2% of the data, but the cause (probably another extension such as `.png`) has not been confirmed yet.

---

## 6. Train / val / test split

**Method:** train is the original SH17 train list after filtering. The original val list after filtering (1,545 images) is split into val and test. Images containing a rare class (`face-mask-medical`, `helmet`, `protective-suit`, `safety-vest`) are split half and half first, then the remaining images, so rare classes appear in both sets.

| Split | Images | Share |
|---|---|---|
| Train | 6,208 | 80.1% |
| Val | 772 | 10.0% |
| Test | 773 | 10.0% |
| Total | 7,753 | 100% |

Images containing each rare class, in the split produced by the notebook (Version 4):

| Class | Val | Test |
|---|---|---|
| face-mask-medical | 36 | 39 |
| helmet | 45 | 48 |
| protective-suit | 35 | 23 |
| safety-vest | 20 | 25 |

> **Reproducibility note:** the notebook shuffled lists in `os.listdir` order, which is not guaranteed to be the same on every machine. The script `prepare_dataset.py` sorts the lists before shuffling with a fixed seed (42), so its split is deterministic. Split sizes are identical, but **which images land in val and which in test will differ from the notebook**, and the per-class counts in the table above may change slightly. Anyone who started training on the notebook split must be told. The table should be refreshed after the first verified run of the script.

**Image selection:** the notebook saved only label files to `/kaggle/working/splits/`. Images stay in the Kaggle input and are not copied. The script builds the full training folder (images and labels) instead.

---

## 7. Status against the Data Preparation Plan

| # | Step | Status |
|---|------|--------|
| 1 | Download SH17 and inspect structure | Done |
| 2 | Filter classes | Done |
| 3 | Count instances per class | Done |
| 4 | Visual quality check | Done |
| 5 | Clean data (duplicates, corrupted files) | Done (18 images still to confirm) |
| 6 | Train/val/test split | Done (to be regenerated by the script) |
| 7 | Add supplementary CCTV-angle sample | Pending, needs team filming |
| 8 | Documentation | This file |

---

## 8. Next steps

1. Merge the reproducible script (`src/industrial_safety/data_prep/prepare_dataset.py`), run it once, commit the split manifests and record the checksum in its README.
2. Refresh section 6 with the numbers from that run.
3. Add `helmet` and `safety-vest` data from an external source before final training (see section 4).
4. Film a supplementary sample from a CCTV-like angle. SH17 is made of stock photographs, which is a domain gap for fixed ceiling cameras. Use the sample mainly in val/test to measure real-world performance.
5. Detection team to decide whether `protective-suit` stays in the training scope.
6. Confirm the cause of the 18 skipped images from section 5.

**Class order for training:** `face-mask-medical (0), glasses (1), gloves (2), helmet (3), shoes (4), protective-suit (5), safety-vest (6), person (7)`.

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
| Corrupted images (opened and verified) | 7,735 images in the notebook, then all 7,753 prepared images on the script output | **0 corrupted** |

The first corruption check covered 7,735 of the 7,753 images because it only looked for `.jpg`/`.jpeg` files. The other 18 images are `.png` files (counted in the prepared folder), and a second check on the full prepared folder found no corrupted image.

---

## 6. Train / val / test split

**Method:** train is the original SH17 train list after filtering. The original val list after filtering (1,545 images) is split into val and test. Images containing a rare class (`face-mask-medical`, `helmet`, `protective-suit`, `safety-vest`) are split half and half first, then the remaining images, so rare classes appear in both sets.

| Split | Images | Share |
|---|---|---|
| Train | 6,208 | 80.1% |
| Val | 772 | 10.0% |
| Test | 773 | 10.0% |
| Total | 7,753 | 100% |

Images containing each class, per split (measured on a Kaggle run of `prepare_dataset.py`):

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| face-mask-medical | 320 | 44 | 31 | 395 |
| glasses | 1,265 | 155 | 168 | 1,588 |
| gloves | 1,067 | 122 | 132 | 1,321 |
| helmet | 373 | 40 | 53 | 466 |
| shoes | 1,250 | 151 | 169 | 1,570 |
| protective-suit | 197 | 29 | 29 | 255 |
| safety-vest | 168 | 21 | 24 | 213 |
| person | 6,102 | 755 | 760 | 7,617 |

The totals match section 4 exactly, so no image was lost in the split.

The rare classes are balanced between val and test as a group (images with any rare class are divided half and half), not class by class, so single classes can differ (for example `helmet`: 40 in val, 53 in test). `safety-vest` has only 21 val and 24 test images, so per-class results for it will vary a lot between runs and must be reported with that caveat.

**Reproducibility.** The split is produced by `prepare_dataset.py`: lists are sorted and the seed is fixed (42), so it does not depend on the machine. This split is the reference one. It is **not** the split of the earlier notebook (Version 4), which shuffled in `os.listdir` order: sizes are the same, but different images land in val and test. Anyone who trained on the notebook split must be told.

| Item | Value |
|---|---|
| Prepared folder size | about 13 GB (`du -sh`, measured) |
| Raw SH17 images | 14 GB (`du -shL`, measured) |
| Checksum (SHA-256 of file paths and sizes) | `fef1f876bceef8a277c1c37db993e04756a25bf03454ebd300d7ccaf735eaa3b` |
| Measured on | one Kaggle notebook (Linux); a second machine has not confirmed it yet |

The three split manifests (image names without extension) are in `src/industrial_safety/data_prep/splits/`. Images are not stored in the repository; the script copies them from the Kaggle download into the training folder.

---

## 7. Status against the Data Preparation Plan

| # | Step | Status |
|---|------|--------|
| 1 | Download SH17 and inspect structure | Done |
| 2 | Filter classes | Done |
| 3 | Count instances per class | Done |
| 4 | Visual quality check | Done |
| 5 | Clean data (duplicates, corrupted files) | Done |
| 6 | Train/val/test split | Done (reproducible script, checksum recorded) |
| 7 | Add supplementary CCTV-angle sample | Pending, needs team filming |
| 8 | Documentation | This file |

---

## 8. Next steps

1. Merge the script PR. Each teammate runs it once and compares the printed checksum with section 6; a match on a second machine is the proof that the split is reproducible.
2. Add `helmet` and `safety-vest` data from an external source before final training (see section 4).
3. Film a supplementary sample from a CCTV-like angle. SH17 is made of stock photographs, which is a domain gap for fixed ceiling cameras. Use the sample mainly in val/test to measure real-world performance.
4. Detection team to decide whether `protective-suit` stays in the training scope.

**Class order for training:** `face-mask-medical (0), glasses (1), gloves (2), helmet (3), shoes (4), protective-suit (5), safety-vest (6), person (7)`.

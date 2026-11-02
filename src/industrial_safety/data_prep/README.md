# data_prep - SH17 dataset builder

Builds a YOLO-ready PPE dataset (`train/val/test`, Ultralytics layout) from the
public SH17 dataset on Kaggle. **Images and full label sets are never committed**
(this repository is public). Only code, the class list and the split manifests
live in Git.

## Requirements

- The repo environment (`pip install -e ".[dev]"`) plus `pip install kaggle`
- Your **own** Kaggle API token: Kaggle -> Settings -> Create New Token, then save
  it as `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY`).
  Never commit `kaggle.json`.
- About **28 GB** free in `DATA_ROOT` (raw download ~14 GB + prepared copy ~13 GB).

## Run

```bash
export DATA_ROOT=/path/with/free/space        # or pass --data-root
python -m industrial_safety.data_prep.prepare_dataset
```

Steps performed:

1. Download SH17 to `$DATA_ROOT/raw/sh17` (skipped if already there).
2. Keep 8 classes (17 -> 8), merging `medical-suit` into `protective-suit`.
3. Train = original train list. The original val list is split into val/test,
   stratified on the rare classes, with a fixed seed (`SEED = 42`).
4. Write `$DATA_ROOT/sh17_yolo/{train,val,test}/{images,labels}`, `data.yaml`
   and `CHECKSUM.txt`.
5. Write the split manifests to `splits/{train,val,test}_files.txt` (image names
   without extension).

Then train with `$DATA_ROOT/sh17_yolo/data.yaml`.

## Classes

| ID | Name |
|----|------|
| 0 | face-mask-medical |
| 1 | glasses |
| 2 | gloves |
| 3 | helmet |
| 4 | shoes |
| 5 | protective-suit |
| 6 | safety-vest |
| 7 | person |

## Reference checksum

SHA-256 over the relative path and size of every produced file, except
`data.yaml` and `CHECKSUM.txt` (they contain machine-specific paths). It is
printed at the end of each run. A different value means the data or the split
changed.

```
fef1f876bceef8a277c1c37db993e04756a25bf03454ebd300d7ccaf735eaa3b
```

Background, statistics and known limitations (class imbalance, photo-vs-CCTV
domain gap): `docs/data/SH17-Data-Preparation-Report.md`.

"""Build a YOLO-ready SH17 PPE dataset (train/val/test) for Industrial Safety AI.

Usage:
    export DATA_ROOT=/path/to/folder
    python -m industrial_safety.data_prep.prepare_dataset
    # or
    python -m industrial_safety.data_prep.prepare_dataset --data-root /path/to/folder

Each team member uses their own Kaggle API key (see README.md).
Never commit kaggle.json, images or full label sets to this public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path

KAGGLE_DATASET = "mugheesahmad/sh17-dataset-for-ppe-detection"
SEED = 42

# original SH17 class id -> new class id
KEEP_CLASSES: dict[int, int] = {
    5: 0,  # face-mask-medical
    8: 1,  # glasses
    9: 2,  # gloves
    10: 3,  # helmet
    14: 4,  # shoes
    15: 5,  # protective-suit (safety-suit)
    13: 5,  # protective-suit (medical-suit merged)
    16: 6,  # safety-vest
    0: 7,  # person
}

# new ids used to stratify the val/test split
RARE_NEW_IDS = {0, 3, 5, 6}  # face-mask-medical, helmet, protective-suit, safety-vest

CLASS_NAMES = [
    "face-mask-medical",
    "glasses",
    "gloves",
    "helmet",
    "shoes",
    "protective-suit",
    "safety-vest",
    "person",
]


def log(msg: str) -> None:
    print(f"[prepare_dataset] {msg}", flush=True)


def download_raw(raw_dir: Path) -> None:
    """Download and unzip SH17 with the Kaggle API (skipped if already present)."""
    if (raw_dir / "labels").exists():
        log(f"Raw dataset already present at {raw_dir}; skipping download.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise SystemExit(
            "Could not load the Kaggle API. Run `pip install kaggle` and make sure "
            "your own kaggle.json is at ~/.kaggle/kaggle.json (or set the "
            "KAGGLE_USERNAME / KAGGLE_KEY environment variables). "
            "Never commit kaggle.json."
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {KAGGLE_DATASET} into {raw_dir} (this is ~14 GB)...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True)
    log("Download complete.")


def filter_labels(raw_dir: Path) -> dict[str, list[str]]:
    """Map image base name to its remapped label lines (images with >= 1 object)."""
    label_files = sorted((raw_dir / "labels").glob("*.txt"))
    kept: dict[str, list[str]] = {}
    for label_path in label_files:
        new_lines = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            new_id = KEEP_CLASSES.get(int(parts[0]))
            if new_id is not None:
                new_lines.append(" ".join([str(new_id), *parts[1:]]))
        if new_lines:
            kept[label_path.stem] = new_lines
    log(f"Filtering kept {len(kept)} of {len(label_files)} images.")
    return kept


def read_names(path: Path) -> set[str]:
    return {
        Path(line.strip()).stem
        for line in path.read_text().splitlines()
        if line.strip()
    }


def has_rare_class(lines: list[str]) -> bool:
    return any(int(line.split()[0]) in RARE_NEW_IDS for line in lines)


def build_split(raw_dir: Path, kept: dict[str, list[str]]) -> dict[str, list[str]]:
    """Train = original train; original val is split into val/test (seeded)."""
    train_original = read_names(raw_dir / "train_files.txt")
    val_original = read_names(raw_dir / "val_files.txt")
    kept_names = set(kept)

    train = sorted(kept_names & train_original)
    val_pool = sorted(kept_names & val_original)
    unmatched = kept_names - train_original - val_original
    if unmatched:
        log(
            f"Warning: {len(unmatched)} images are in neither original list; "
            "adding them to train."
        )
        train = sorted(set(train) | unmatched)

    rare = [n for n in val_pool if has_rare_class(kept[n])]
    common = [n for n in val_pool if not has_rare_class(kept[n])]

    rng = random.Random(SEED)
    rng.shuffle(rare)
    rng.shuffle(common)

    val = sorted(rare[: len(rare) // 2] + common[: len(common) // 2])
    test = sorted(rare[len(rare) // 2 :] + common[len(common) // 2 :])
    log(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return {"train": train, "val": val, "test": test}


def find_image(images_dir: Path, name: str) -> Path | None:
    for ext in (".jpeg", ".jpg", ".png"):
        candidate = images_dir / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def write_split(
    raw_dir: Path,
    out_root: Path,
    split: str,
    names: list[str],
    kept: dict[str, list[str]],
) -> None:
    images_out = out_root / split / "images"
    labels_out = out_root / split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    missing = 0
    for name in names:
        image = find_image(raw_dir / "images", name)
        if image is None:
            missing += 1
            continue
        shutil.copy2(image, images_out / image.name)
        (labels_out / f"{name}.txt").write_text("\n".join(kept[name]) + "\n")
    if missing:
        log(
            f"Warning: {missing} images in '{split}' had no image file "
            "and were skipped."
        )


def write_data_yaml(out_root: Path) -> None:
    lines = [
        f"path: {out_root.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
    ]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n")


def compute_checksum(out_root: Path) -> str:
    """SHA-256 over every produced file's relative path and size (stable, fast)."""
    digest = hashlib.sha256()
    for path in sorted(out_root.rglob("*")):
        if path.is_file() and path.name != "CHECKSUM.txt":
            digest.update(path.relative_to(out_root).as_posix().encode())
            digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the YOLO-ready SH17 PPE dataset."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"),
        help="Where to store raw + prepared data (or set the DATA_ROOT env var).",
    )
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Provide --data-root or set the DATA_ROOT environment variable.")

    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw" / "sh17"
    out_root = data_root / "sh17_yolo"

    download_raw(raw_dir)
    kept = filter_labels(raw_dir)
    splits = build_split(raw_dir, kept)

    manifest_dir = Path(__file__).parent / "splits"
    manifest_dir.mkdir(exist_ok=True)
    for split, names in splits.items():
        write_split(raw_dir, out_root, split, names, kept)
        (manifest_dir / f"{split}_files.txt").write_text("\n".join(names) + "\n")

    write_data_yaml(out_root)
    checksum = compute_checksum(out_root)
    (out_root / "CHECKSUM.txt").write_text(checksum + "\n")

    log(f"Dataset ready at: {out_root}")
    log(f"Checksum (sha256 of file paths + sizes): {checksum}")
    log("Paste this checksum into the README of this folder.")


if __name__ == "__main__":
    main()

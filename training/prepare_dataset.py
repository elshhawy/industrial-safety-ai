"""Inspect a local SH17-style dataset, remap classes, and write a YOLO split.

This script does not download data. It inspects whatever directory is passed
as --source, reads class names from metadata, and maps those names onto the
six target classes. Source class IDs are never assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, "") and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.taxonomy import (  # noqa: E402
    TARGET_CLASS_NAMES,
    TARGET_NAME_TO_ID,
    build_source_id_to_target_id,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
YAML_CANDIDATES = ("sh17.yaml", "data.yaml", "dataset.yaml", "data.yml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and remap a local SH17 dataset to six PPE classes."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/sh17"),
        help="Extracted source dataset directory (not downloaded by this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/filtered_sh17"),
        help="Output directory for the remapped YOLO dataset.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Keep images that have no retained target objects.",
    )
    parser.add_argument(
        "--use-source-split",
        action="store_true",
        help="Use source train_files.txt / val_files.txt when present.",
    )
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f"PyYAML is required to read {path}. Install project dependencies."
        ) from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _names_from_mapping(names: Any) -> dict[int, str] | None:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return None


def parse_class_names_from_yaml(path: Path) -> dict[int, str] | None:
    data = _load_yaml(path)
    if not isinstance(data, dict):
        return None
    return _names_from_mapping(data.get("names"))


def parse_class_names_from_classes_txt(path: Path) -> dict[int, str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {index: name for index, name in enumerate(names)}


def _xml_object_names(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    names: list[str] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is not None and name_el.text:
            names.append(name_el.text.strip())
    return names


def _yolo_class_ids(label_path: Path) -> list[int]:
    ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        first = line.split()[0]
        try:
            ids.append(int(first))
        except ValueError:
            continue
    return ids


def infer_class_names_from_voc(
    voc_dir: Path, labels_dir: Path, sample_limit: int = 1000
) -> dict[int, str] | None:
    """Infer YOLO id -> name by pairing VOC names with YOLO ids on the same stem."""
    votes: dict[int, Counter[str]] = {}
    xml_files = sorted(voc_dir.rglob("*.xml"))[:sample_limit]
    if not xml_files:
        return None
    label_index = {path.stem: path for path in labels_dir.rglob("*.txt")}
    paired = 0
    for xml_path in xml_files:
        label_path = label_index.get(xml_path.stem)
        if label_path is None:
            continue
        names = _xml_object_names(xml_path)
        ids = _yolo_class_ids(label_path)
        if len(names) != len(ids) or not names:
            continue
        paired += 1
        for class_id, name in zip(ids, names, strict=True):
            votes.setdefault(class_id, Counter())[name] += 1
    if not votes or paired == 0:
        return None
    return {
        class_id: counter.most_common(1)[0][0] for class_id, counter in votes.items()
    }


def find_images_and_labels_dirs(source: Path) -> tuple[Path, Path]:
    images = source / "images"
    labels = source / "labels"
    if images.is_dir() and labels.is_dir():
        return images, labels
    nested = list(source.rglob("images"))
    for image_dir in nested:
        sibling_labels = image_dir.parent / "labels"
        if image_dir.is_dir() and sibling_labels.is_dir():
            return image_dir, sibling_labels
    raise FileNotFoundError(
        f"Could not find paired images/ and labels/ directories under {source}. "
        "Wait until SH17 is extracted and pass --source to that folder."
    )


def load_source_class_names(source: Path, labels_dir: Path) -> dict[int, str]:
    searched: list[str] = []
    for name in YAML_CANDIDATES:
        candidate = source / name
        searched.append(str(candidate))
        if candidate.is_file():
            parsed = parse_class_names_from_yaml(candidate)
            if parsed:
                return parsed
    for candidate in sorted(source.glob("*.yaml")) + sorted(source.glob("*.yml")):
        searched.append(str(candidate))
        parsed = parse_class_names_from_yaml(candidate)
        if parsed:
            return parsed
    classes_txt = source / "classes.txt"
    searched.append(str(classes_txt))
    if classes_txt.is_file():
        return parse_class_names_from_classes_txt(classes_txt)
    voc_dir = source / "voc_labels"
    if voc_dir.is_dir():
        inferred = infer_class_names_from_voc(voc_dir, labels_dir)
        if inferred:
            return inferred
    raise FileNotFoundError(
        "Could not read source class names. Looked for YAML, classes.txt, and "
        f"voc_labels under {source}. Searched: {searched}. Do not assume class "
        "IDs; add metadata or wait until the downloaded dataset is complete."
    )


def iter_image_label_pairs(
    images_dir: Path, labels_dir: Path
) -> tuple[list[tuple[Path, Path]], list[str]]:
    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []
    label_by_rel = {
        path.relative_to(labels_dir).with_suffix("").as_posix().lower(): path
        for path in labels_dir.rglob("*.txt")
    }
    label_by_stem: dict[str, list[Path]] = {}
    for path in labels_dir.rglob("*.txt"):
        label_by_stem.setdefault(path.stem.lower(), []).append(path)

    for image_path in sorted(images_dir.rglob("*")):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue
        rel_key = image_path.relative_to(images_dir).with_suffix("").as_posix().lower()
        label_path = label_by_rel.get(rel_key)
        if label_path is None:
            stem_matches = label_by_stem.get(image_path.stem.lower(), [])
            if len(stem_matches) == 1:
                label_path = stem_matches[0]
        if label_path is None:
            missing_labels.append(str(image_path))
            continue
        pairs.append((image_path, label_path))
    return pairs, missing_labels


def validate_and_remap_annotation(
    parts: list[str], source_id_to_target_id: dict[int, int]
) -> tuple[str | None, str | None]:
    if len(parts) < 5:
        return None, "malformed"
    try:
        source_id = int(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:5])
    except ValueError:
        return None, "malformed"
    if source_id not in source_id_to_target_id:
        return None, "excluded_class"
    values = (x_center, y_center, width, height)
    if not all(math.isfinite(value) for value in values):
        return None, "invalid_coords"
    if width <= 0 or height <= 0:
        return None, "invalid_coords"
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None, "invalid_coords"
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    if x_min < -1e-6 or y_min < -1e-6 or x_max > 1 + 1e-6 or y_max > 1 + 1e-6:
        return None, "invalid_coords"
    target_id = source_id_to_target_id[source_id]
    return f"{target_id} {x_center} {y_center} {width} {height}", None


def remap_label_file(
    label_path: Path, source_id_to_target_id: dict[int, int]
) -> tuple[list[str], Counter[str]]:
    skipped: Counter[str] = Counter()
    kept: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        skipped["unreadable_label"] += 1
        return kept, skipped
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        remapped, reason = validate_and_remap_annotation(
            stripped.split(), source_id_to_target_id
        )
        if reason:
            skipped[reason] += 1
            continue
        assert remapped is not None
        kept.append(remapped)
    return kept, skipped


def unique_output_stem(image_path: Path, used: set[str]) -> str:
    stem = image_path.stem
    if stem not in used:
        used.add(stem)
        return stem
    parent = image_path.parent.name
    candidate = f"{parent}_{stem}"
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{parent}_{stem}_{suffix}"
    used.add(candidate)
    return candidate


def split_pairs(
    pairs: list[tuple[Path, Path, list[str]]],
    train_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path, list[str]]], list[tuple[Path, Path, list[str]]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 retained images to create a train/val split, "
            f"got {len(pairs)}."
        )
    ordered = sorted(pairs, key=lambda item: item[0].as_posix().lower())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_train = int(len(ordered) * train_ratio)
    n_train = min(max(n_train, 1), len(ordered) - 1)
    return ordered[:n_train], ordered[n_train:]


def _match_split_file(
    split_file: Path, available: dict[str, tuple[Path, Path, list[str]]]
):
    selected = []
    missing = []
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        key = Path(line).stem.lower()
        item = available.get(key)
        if item is None:
            missing.append(line)
            continue
        selected.append(item)
    return selected, missing


def apply_source_split(
    retained: list[tuple[Path, Path, list[str]]],
    source: Path,
) -> tuple[list[tuple[Path, Path, list[str]]], list[tuple[Path, Path, list[str]]]]:
    train_file = source / "train_files.txt"
    val_file = source / "val_files.txt"
    if not train_file.is_file() or not val_file.is_file():
        raise FileNotFoundError(
            "--use-source-split requires train_files.txt and val_files.txt in the "
            f"source directory ({source})."
        )
    available = {item[0].stem.lower(): item for item in retained}
    train_items, _ = _match_split_file(train_file, available)
    val_items, _ = _match_split_file(val_file, available)
    train_stems = {item[0].stem.lower() for item in train_items}
    val_items = [item for item in val_items if item[0].stem.lower() not in train_stems]
    if not train_items or not val_items:
        raise ValueError(
            "Source split did not yield both train and val images after filtering."
        )
    return train_items, val_items


def class_instance_counts(items: list[tuple[Path, Path, list[str]]]) -> dict[str, int]:
    counts = Counter()
    for _image, _label, lines in items:
        for line in lines:
            class_id = int(line.split()[0])
            counts[TARGET_CLASS_NAMES[class_id]] += 1
    return {name: int(counts.get(name, 0)) for name in TARGET_CLASS_NAMES}


def write_split(
    items: list[tuple[Path, Path, list[str]]],
    output: Path,
    split: str,
    used_stems: set[str],
) -> list[str]:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for image_path, _label_path, lines in items:
        stem = unique_output_stem(image_path, used_stems)
        dest_image = image_dir / f"{stem}{image_path.suffix.lower()}"
        dest_label = label_dir / f"{stem}.txt"
        shutil.copy2(image_path, dest_image)
        dest_label.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        written.append(stem)
    return written


def write_data_yaml(output: Path) -> Path:
    yaml_path = output / "data.yaml"
    names_block = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(TARGET_CLASS_NAMES)
    )
    content = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(TARGET_CLASS_NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def prepare_dataset(
    source: Path,
    output: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    include_negatives: bool = False,
    use_source_split: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"Source dataset directory does not exist: {source}. "
            "SH17 must be downloaded and extracted separately; pass --source."
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory {output} already exists and is not empty. "
            "Choose another path or remove it explicitly. The source dataset "
            "is never modified."
        )

    images_dir, labels_dir = find_images_and_labels_dirs(source)
    source_names = load_source_class_names(source, labels_dir)
    source_id_to_target_id, mapping_details = build_source_id_to_target_id(source_names)
    pairs, missing_labels = iter_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found under {source}.")

    skip_totals: Counter[str] = Counter()
    retained: list[tuple[Path, Path, list[str]]] = []
    excluded_empty = 0
    person_instances = 0

    for image_path, label_path in pairs:
        lines, skipped = remap_label_file(label_path, source_id_to_target_id)
        skip_totals.update(skipped)
        if not lines and not include_negatives:
            excluded_empty += 1
            continue
        person_instances += sum(
            1 for line in lines if int(line.split()[0]) == TARGET_NAME_TO_ID["person"]
        )
        retained.append((image_path, label_path, lines))

    if not retained:
        raise ValueError("No images remained after class filtering.")
    if person_instances == 0:
        raise ValueError(
            "The source dataset metadata includes a person class, but no usable "
            "person bounding boxes remained after filtering. Person annotations "
            "will not be fabricated. Additional person labels are required."
        )

    if use_source_split:
        train_items, val_items = apply_source_split(retained, source)
        split_method = "source_files"
    else:
        train_items, val_items = split_pairs(retained, train_ratio, seed)
        split_method = "random_seed"

    train_stems = {item[0].resolve().as_posix() for item in train_items}
    val_stems = {item[0].resolve().as_posix() for item in val_items}
    overlap = train_stems & val_stems
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {len(overlap)} paths.")

    output.mkdir(parents=True, exist_ok=True)
    used_stems: set[str] = set()
    write_split(train_items, output, "train", used_stems)
    write_split(val_items, output, "val", used_stems)
    data_yaml = write_data_yaml(output)

    train_counts = class_instance_counts(train_items)
    val_counts = class_instance_counts(val_items)
    absent_in_train = [name for name, count in train_counts.items() if count == 0]
    absent_in_val = [name for name, count in val_counts.items() if count == 0]

    stats: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "data_yaml": str(data_yaml),
        "source_images_with_labels": len(pairs),
        "missing_label_files": len(missing_labels),
        "images_excluded_no_target_objects": excluded_empty,
        "images_retained": len(retained),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "train_ratio": train_ratio,
        "seed": seed,
        "split_method": split_method,
        "include_negatives": include_negatives,
        "person_instances": person_instances,
        "skipped_annotations": dict(skip_totals),
        "train_class_instances": train_counts,
        "val_class_instances": val_counts,
        "classes_absent_from_train": absent_in_train,
        "classes_absent_from_val": absent_in_val,
        "mapping": mapping_details,
        "target_class_names": list(TARGET_CLASS_NAMES),
    }
    stats_path = output / "preparation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    stats["stats_path"] = str(stats_path)
    return stats


def _print_stats(stats: dict[str, Any]) -> None:
    print("SH17 dataset preparation complete")
    print(f"  source: {stats['source']}")
    print(f"  output: {stats['output']}")
    print(f"  retained images: {stats['images_retained']}")
    print(
        f"  excluded (no target objects): {stats['images_excluded_no_target_objects']}"
    )
    print(f"  train: {stats['train_images']}  val: {stats['val_images']}")
    print(f"  split: {stats['split_method']}  seed: {stats['seed']}")
    print(f"  person instances: {stats['person_instances']}")
    print(f"  skipped annotations: {stats['skipped_annotations']}")
    print(f"  train instances: {stats['train_class_instances']}")
    print(f"  val instances: {stats['val_class_instances']}")
    if stats["classes_absent_from_train"]:
        print(f"  absent from train: {stats['classes_absent_from_train']}")
    if stats["classes_absent_from_val"]:
        print(f"  absent from val: {stats['classes_absent_from_val']}")
    print(f"  data yaml: {stats['data_yaml']}")
    print(f"  stats: {stats['stats_path']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = prepare_dataset(
            source=args.source,
            output=args.output,
            train_ratio=args.train_ratio,
            seed=args.seed,
            include_negatives=args.include_negatives,
            use_source_split=args.use_source_split,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

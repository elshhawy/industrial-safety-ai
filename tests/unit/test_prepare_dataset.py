"""GPU-free tests for SH17 class mapping and dataset preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.prepare_dataset import (  # noqa: E402
    prepare_dataset,
    remap_label_file,
    split_pairs,
    validate_and_remap_annotation,
    write_data_yaml,
)
from training.taxonomy import (  # noqa: E402
    TARGET_CLASS_NAMES,
    build_source_id_to_target_id,
)

# Tiny valid JPEG (1x1) so image copy does not depend on OpenCV.
JPEG_1X1 = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
    b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00"
    b"\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00"
    b"\x00?\x00\x7f\x3f\xff\xd9"
)

# Fixture IDs follow the public SH17 yaml published by the dataset authors.
# They are used only in tests and are not read from a real download.
FIXTURE_NAMES = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}


def _write_yaml(path: Path, names: dict[int, str]) -> None:
    lines = ["names:"]
    for key, value in names.items():
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_source(root: Path, names: dict[int, str] | None = None) -> Path:
    names = FIXTURE_NAMES if names is None else names
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    _write_yaml(root / "sh17.yaml", names)

    samples = {
        "a": ["0 0.5 0.5 0.2 0.2", "5 0.3 0.3 0.1 0.1", "14 0.7 0.7 0.1 0.1"],
        "b": ["4 0.4 0.4 0.2 0.2", "10 0.6 0.2 0.1 0.1"],
        "c": ["8 0.5 0.5 0.1 0.1", "9 0.2 0.8 0.1 0.1", "0 0.3 0.3 0.2 0.2"],
        "d": ["13 0.5 0.5 0.3 0.3", "16 0.2 0.2 0.1 0.1"],
        "e": ["15 0.4 0.6 0.2 0.2", "0 0.5 0.5 0.4 0.4"],
        "f": ["1 0.5 0.5 0.1 0.1"],  # only excluded class
    }
    for stem, lines in samples.items():
        (images / f"{stem}.jpg").write_bytes(JPEG_1X1)
        (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class TaxonomyTests(unittest.TestCase):
    def test_source_to_target_mapping_uses_names_not_raw_ids(self) -> None:
        mapping, details = build_source_id_to_target_id(FIXTURE_NAMES)
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[4], 1)
        self.assertEqual(mapping[5], 1)
        self.assertEqual(mapping[8], 2)
        self.assertEqual(mapping[9], 3)
        self.assertEqual(mapping[10], 4)
        self.assertEqual(mapping[13], 5)
        self.assertEqual(mapping[15], 5)
        self.assertEqual(mapping[16], 5)
        self.assertNotIn(1, mapping)
        self.assertNotIn(14, mapping)
        self.assertEqual(details["target_class_names"], TARGET_CLASS_NAMES)

    def test_face_mask_medical_alias(self) -> None:
        names = dict(FIXTURE_NAMES)
        names[5] = "face-mask-medical"
        mapping, _ = build_source_id_to_target_id(names)
        self.assertEqual(mapping[5], 1)

    def test_missing_person_class_fails(self) -> None:
        names = {k: v for k, v in FIXTURE_NAMES.items() if v != "person"}
        with self.assertRaises(ValueError) as ctx:
            build_source_id_to_target_id(names)
        self.assertIn("person", str(ctx.exception).lower())


class AnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping, _ = build_source_id_to_target_id(FIXTURE_NAMES)

    def test_excluded_classes_are_removed(self) -> None:
        line, reason = validate_and_remap_annotation(
            "14 0.5 0.5 0.1 0.1".split(), self.mapping
        )
        self.assertIsNone(line)
        self.assertEqual(reason, "excluded_class")

    def test_annotation_ids_are_remapped(self) -> None:
        line, reason = validate_and_remap_annotation(
            "16 0.5 0.5 0.2 0.2".split(), self.mapping
        )
        self.assertIsNone(reason)
        self.assertEqual(line.split()[0], "5")

    def test_invalid_coordinates_are_rejected(self) -> None:
        cases = [
            "0 0.5 0.5 0.0 0.2",
            "0 1.5 0.5 0.1 0.1",
            "0 nan 0.5 0.1 0.1",
            "0 0.5 0.5 2.0 0.1",
            "x 0.5 0.5 0.1 0.1",
        ]
        for raw in cases:
            line, reason = validate_and_remap_annotation(raw.split(), self.mapping)
            self.assertIsNone(line, raw)
            self.assertIn(reason, {"invalid_coords", "malformed"}, raw)


class SplitTests(unittest.TestCase):
    def test_split_is_disjoint_and_reproducible(self) -> None:
        items = [
            (Path(f"img{i}.jpg"), Path(f"img{i}.txt"), ["0 0.5 0.5 0.1 0.1"])
            for i in range(10)
        ]
        train_a, val_a = split_pairs(items, 0.8, 42)
        train_b, val_b = split_pairs(items, 0.8, 42)
        self.assertEqual([p[0] for p in train_a], [p[0] for p in train_b])
        self.assertEqual([p[0] for p in val_a], [p[0] for p in val_b])
        train_ids = {p[0] for p in train_a}
        val_ids = {p[0] for p in val_a}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(len(train_ids) + len(val_ids), 10)

    def test_generated_yaml_has_six_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            yaml_path = write_data_yaml(output)
            text = yaml_path.read_text(encoding="utf-8")
        self.assertIn("nc: 6", text)
        self.assertIn("train: images/train", text)
        self.assertIn("val: images/val", text)
        for name in TARGET_CLASS_NAMES:
            self.assertIn(name, text)


class PipelineTests(unittest.TestCase):
    def test_prepare_dataset_end_to_end_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = _make_source(tmp_path / "source")
            output = tmp_path / "filtered"
            stats = prepare_dataset(source, output, train_ratio=0.8, seed=42)

            self.assertEqual(stats["images_excluded_no_target_objects"], 1)
            self.assertEqual(stats["images_retained"], 5)
            self.assertEqual(stats["train_images"] + stats["val_images"], 5)
            self.assertGreater(stats["person_instances"], 0)
            self.assertEqual(stats["skipped_annotations"].get("excluded_class", 0), 2)

            yaml_text = Path(stats["data_yaml"]).read_text(encoding="utf-8")
            self.assertIn("nc: 6", yaml_text)
            self.assertIn("Face-covering", yaml_text)

            train_labels = list((output / "labels" / "train").glob("*.txt"))
            val_labels = list((output / "labels" / "val").glob("*.txt"))
            self.assertTrue(train_labels)
            self.assertTrue(val_labels)
            train_stems = {p.stem for p in train_labels}
            val_stems = {p.stem for p in val_labels}
            self.assertFalse(train_stems & val_stems)

            for label in train_labels + val_labels:
                for line in label.read_text(encoding="utf-8").splitlines():
                    class_id = int(line.split()[0])
                    self.assertGreaterEqual(class_id, 0)
                    self.assertLessEqual(class_id, 5)

            stats_json = json.loads(
                Path(stats["stats_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(stats_json["target_class_names"], TARGET_CLASS_NAMES)

    def test_remap_label_file_records_invalid_lines(self) -> None:
        mapping, _ = build_source_id_to_target_id(FIXTURE_NAMES)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text(
                "0 0.5 0.5 0.1 0.1\nnot-a-box\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
            kept, skipped = remap_label_file(path, mapping)
            self.assertEqual(len(kept), 1)
            self.assertEqual(skipped["malformed"], 1)
            self.assertEqual(skipped["excluded_class"], 1)


if __name__ == "__main__":
    unittest.main()

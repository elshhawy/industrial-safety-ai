"""GPU-free tests for training/validation configuration and MLflow logging."""
# ruff: noqa: E402

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.taxonomy import TARGET_CLASS_NAMES  # noqa: E402
from training.train import (  # noqa: E402
    assert_yolov9e,
    build_train_kwargs,
)
from training.train import (
    parse_args as parse_train_args,
)
from training.val import (  # noqa: E402
    parse_args as parse_val_args,
)
from training.val import (
    validation_kwargs,
    verify_model_class_names,
)


class FakeRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="train-run-1")


class FakeMLflow:
    def __init__(self) -> None:
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[str] = []
        self.ended = False

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name: str) -> FakeRun:
        self.run_name = run_name
        return FakeRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def end_run(self) -> None:
        self.ended = True


class FakeYOLO:
    def __init__(self, model: str) -> None:
        self.model = model
        self.names = {index: name for index, name in enumerate(TARGET_CLASS_NAMES)}
        self.train_kwargs = None
        self.val_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return SimpleNamespace(
            results_dict={"metrics/mAP50(B)": 0.1, "metrics/precision(B)": 0.2},
            save_dir="runs/detect/yolov9e_sh17",
        )

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        box = SimpleNamespace(
            mp=0.4,
            mr=0.5,
            map50=0.6,
            map=0.3,
            p=[0.4] * 6,
            r=[0.5] * 6,
            ap50=[0.6] * 6,
        )
        return SimpleNamespace(box=box, seen=3)


class TrainConfigTests(unittest.TestCase):
    def test_default_seed_and_yolov9e(self) -> None:
        args = parse_train_args([])
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.model, "yolov9e.pt")
        self.assertTrue(args.deterministic)
        assert_yolov9e(args.model)
        kwargs = build_train_kwargs(args, "cpu")
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["data"], str(args.data))

    def test_rejects_non_yolov9e_model(self) -> None:
        with self.assertRaises(ValueError):
            assert_yolov9e("yolov8n.pt")

    def test_train_logs_mlflow_params_without_calling_real_yolo(self) -> None:
        from training import train as train_mod

        fake_mlflow = FakeMLflow()
        with tempfile.TemporaryDirectory() as tmp:
            data_yaml = Path(tmp) / "data.yaml"
            data_yaml.write_text("nc: 6\n", encoding="utf-8")
            factory = MagicMock(side_effect=lambda ref: FakeYOLO(ref))
            with patch.object(train_mod, "resolve_device", return_value="cpu"):
                with patch.object(
                    train_mod, "start_run", return_value=(fake_mlflow, FakeRun())
                ):
                    code = train_mod.main(
                        [
                            "--model",
                            "yolov9e.pt",
                            "--data",
                            str(data_yaml),
                            "--epochs",
                            "1",
                            "--device",
                            "cpu",
                            "--allow-cpu",
                        ],
                        model_factory=factory,
                    )
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args[0][0], "yolov9e.pt")
        self.assertEqual(fake_mlflow.params["architecture"], "YOLOv9e")
        self.assertEqual(int(fake_mlflow.params["seed"]), 42)
        self.assertIn("metrics/mAP50_B", fake_mlflow.metrics)
        self.assertTrue(fake_mlflow.ended)


class ValConfigTests(unittest.TestCase):
    def test_validation_targets_val_split(self) -> None:
        args = parse_val_args(["--split", "val"])
        kwargs = validation_kwargs(args, "cpu")
        self.assertEqual(kwargs["split"], "val")
        with self.assertRaises(ValueError):
            args_bad = parse_val_args(["--split", "train"])
            validation_kwargs(args_bad, "cpu")

    def test_class_name_verification(self) -> None:
        verify_model_class_names({i: name for i, name in enumerate(TARGET_CLASS_NAMES)})
        with self.assertRaises(ValueError):
            verify_model_class_names({0: "Hardhat", 1: "NO-Hardhat"})

    def test_val_logs_metrics_and_uses_val_split(self) -> None:
        from training import val as val_mod

        fake_mlflow = FakeMLflow()
        fake_model = FakeYOLO("best.pt")
        with tempfile.TemporaryDirectory() as tmp:
            data_yaml = Path(tmp) / "data.yaml"
            ckpt = Path(tmp) / "best.pt"
            data_yaml.write_text("nc: 6\n", encoding="utf-8")
            ckpt.write_bytes(b"fake")

            def mock_require_run(*, run_name, experiment_name, tracking_uri, tags):
                fake_mlflow.set_tags(tags or {})
                return fake_mlflow, FakeRun()

            with patch.object(val_mod, "resolve_device", return_value="cpu"):
                with patch.object(
                    val_mod, "require_mlflow_run", side_effect=mock_require_run
                ):
                    code = val_mod.main(
                        [
                            "--model",
                            str(ckpt),
                            "--data",
                            str(data_yaml),
                            "--training-run-id",
                            "train-run-1",
                            "--allow-cpu",
                        ],
                        model_factory=lambda _path: fake_model,
                    )
        self.assertEqual(code, 0)
        self.assertEqual(fake_model.val_kwargs["split"], "val")
        self.assertEqual(fake_mlflow.tags["evaluated_split"], "val")
        self.assertEqual(fake_mlflow.tags["training_run_id"], "train-run-1")
        self.assertAlmostEqual(fake_mlflow.metrics["map50"], 0.6)
        self.assertTrue(fake_mlflow.ended)


if __name__ == "__main__":
    unittest.main()

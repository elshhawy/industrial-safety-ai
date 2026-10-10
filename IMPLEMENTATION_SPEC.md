# Implementation Specification: SH17 Dataset Pipeline, YOLOv9e Training, Validation, and MLflow

## 1. Objective

Implement a complete, reproducible PPE detection pipeline for the existing Industrial Safety AI repository.

The pipeline must:

1. Obtain or locate the SH17 dataset.
2. Filter the dataset to the predefined target classes.
3. Merge the specified source classes into the required target classes.
4. Split the filtered dataset into training and validation subsets.
5. Train a YOLOv9e object detection model using the filtered dataset.
6. Validate the trained model using the validation subset.
7. Track training experiments and evaluation results with MLflow.
8. Standardize dependencies using `pyproject.toml`.
9. Document setup, dataset preparation, training, and validation in `training/README.md`.

**Implementation is required.** Inspect the repository, modify the necessary files, run appropriate checks, and report actual results. Do not stop after producing a plan.

## 2. Mandatory Repository Inspection

Before making changes:

- Inspect the repository structure and Git status.
- Read existing training, validation, and dataset preparation scripts.
- Inspect `training/data.yaml`, if present.
- Inspect `.gitignore` and existing dependency files.
- Identify the installed Ultralytics version and the existing model-loading approach.
- Check available hardware and document GPU requirements.
- Reuse working code where practical.
- Preserve existing functionality unless a change is necessary and documented.
- Do not overwrite existing scripts without inspecting their contents.
- Do not run an expensive full training job automatically.

If the dataset location, source class IDs, or model checkpoint cannot be verified, report the uncertainty and implement configurable paths instead of inventing values.

## 3. Target Classes and Class Mapping

The final model must detect exactly six target classes in this order:

| Target ID | Target class | Source classes |
|---|---|---|
| 0 | `person` | `person` |
| 1 | `Face-covering` | `face-guard`, `face-mask` |
| 2 | `glasses` | `glasses` |
| 3 | `gloves` | `gloves` |
| 4 | `helmet` | `helmet` |
| 5 | `suit` | `medical-suit`, `safety-suit`, `safety-vest` |

Treat this mapping as the required target taxonomy. Verify the actual source class names and IDs from the downloaded SH17 dataset metadata or its authoritative documentation before implementing the conversion.

Do not assume that source IDs match target IDs.

### Class Mapping Requirements

- Create an explicit source-to-target class mapping.
- Convert each retained annotation to the corresponding target class ID.
- Merge source classes exactly as specified above.
- Exclude annotations belonging to classes outside the target taxonomy.
- Ensure target class IDs are contiguous integers from `0` to `5`.
- Update image annotations and dataset metadata consistently.
- Do not merely rename class names in YAML while leaving incompatible annotation IDs unchanged.
- Preserve the original dataset without modifying its files.
- Write the filtered dataset to a separate output directory.
- Log source class names, target class names, and filtering statistics.

### Important Person-Annotation Requirement

Verify whether the source dataset contains usable person bounding boxes. If it does not, do not fabricate person annotations or silently treat missing annotations as valid person labels. Report the limitation and explain what additional annotation data is needed.

## 4. Dataset Acquisition and Input Configuration

Implement a dataset preparation pipeline that supports a configurable source dataset directory.

Expected configuration should include:

- Source dataset path.
- Filtered dataset output path.
- Dataset configuration path.
- Train/validation split ratio.
- Random seed.
- Target class mapping.

Support either:

1. A dataset already downloaded and extracted locally, or
2. A documented download procedure from a verified, authorized source.

Do not invent dataset download URLs, access credentials, or licensing permissions.

Do not automatically download large files without documenting the operation and its destination.

Keep downloaded data, generated annotations, and model weights outside version control.

## 5. Dataset Filtering and Preparation

Create a dedicated dataset preparation script, preferably:

`training/prepare_dataset.py`

Adapt the filename if the existing repository has an established structure.

### Required Behavior

1. Locate the source images and their YOLO annotations.
2. Validate the dataset structure.
3. Read and verify the source class metadata.
4. Match source class names to the required target mapping.
5. Retain only the specified classes.
6. Remap retained annotations to target IDs `0–5`.
7. Preserve bounding box coordinates correctly.
8. Exclude images that contain no retained target objects, unless an explicit configuration option enables negative images.
9. Copy or otherwise safely prepare retained images and their transformed annotation files.
10. Detect missing annotations, malformed labels, invalid class IDs, invalid normalized coordinates, and missing image files.
11. Report filtering statistics and any skipped files.
12. Avoid modifying the original dataset.

### Annotation Integrity

For YOLO detection labels, each retained annotation must use this format:

`class_id x_center y_center width height`

Validate that:

- The class ID is an integer in the target range.
- Coordinates are finite numbers.
- Width and height are positive.
- Normalized coordinates and box extents are valid for the selected dataset format.
- Each annotation corresponds to the correct image.

Do not silently repair malformed annotations without recording the action. Prefer failing with a useful error message when a correction could change annotation meaning.

### Output Dataset Structure

Generate a dataset in the following YOLO-compatible layout:

```text
data/
└── filtered_sh17/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

The generated `data.yaml` must contain:

- Correct training image path.
- Correct validation image path.
- `nc: 6`.
- Class names in the exact target order defined in Section 3.

Use valid YAML and paths compatible with the actual training environment.

## 6. Train/Validation Split

Split the filtered dataset into training and validation subsets before training.

### Requirements

- Use a configurable split ratio, defaulting to `80%` training and `20%` validation.
- Use a fixed random seed, defaulting to `42`.
- Ensure the split is reproducible.
- Ensure every retained image belongs to exactly one split.
- Keep each image and its corresponding label file together.
- Avoid train/validation leakage.
- If multiple images originate from the same source scene, video, or sequence, group them appropriately when the dataset metadata supports it.
- Inspect class frequencies before and after splitting.
- Report classes that are absent from either split.
- Do not duplicate images merely to force class balance.
- Do not claim stratification unless the implemented algorithm actually performs it.

If the dataset is too small or lacks sufficient examples for a reliable split, report the issue rather than silently producing misleading evaluation data.

### Acceptance Criteria

- The train and validation file lists do not overlap.
- The split uses the configured seed.
- All labels use the final six-class taxonomy.
- The split counts and class distribution are reported.
- Re-running preparation with the same source data and seed produces the same split, provided the inputs and implementation remain unchanged.

## 7. YOLOv9e Model Training

Implement or update the training script:

`training/train.py`

Train the filtered dataset using **YOLOv9e**, not YOLOv8n or another architecture.

### Model Requirements

- Use a verified YOLOv9e-compatible Ultralytics model checkpoint or a documented compatible implementation.
- Do not assume a checkpoint is available locally.
- Make the model checkpoint configurable.
- Document how the checkpoint is obtained.
- Verify that the selected implementation supports the required training and validation operations.
- Report incompatibilities rather than silently substituting a different model.

### Training Configuration

Make the following settings configurable:

- Model checkpoint.
- Filtered dataset YAML path.
- Epoch count.
- Image size.
- Batch size.
- Learning rate, where supported.
- Random seed.
- Deterministic training setting.
- Device selection.
- Worker count.
- Output directory.
- Experiment name.
- MLflow tracking configuration.

Use the following illustrative defaults only when compatible with the existing project and hardware:

- Seed: `42`
- Image size: `640`
- Train/validation ratio: `80/20`

Do not launch an unnecessarily expensive training run just to test code integration.

### Reproducibility

- Pass the configured seed to the training API.
- Enable deterministic behavior where supported.
- Record all important training parameters.
- Document that identical results are not guaranteed across different hardware and software environments.

### Training Outputs

Preserve and document the location of:

- Best checkpoint: `best.pt`
- Last checkpoint: `last.pt`
- Training plots.
- Training logs.
- Experiment metadata.

Do not commit large checkpoint files to Git.

### Acceptance Criteria

- Training uses the filtered dataset configuration.
- The model is YOLOv9e as explicitly configured.
- The final model has six target classes in the required order.
- The configured seed and training parameters are passed to the model.
- Training outputs are saved in a predictable location.
- Errors such as missing checkpoints, invalid YAML, and unavailable CUDA devices produce useful messages.

## 8. Model Validation

Implement or update:

`training/val.py`

The validation script must evaluate the trained YOLOv9e checkpoint against the **validation subset created by the dataset preparation pipeline**.

### Requirements

- Load the configured trained checkpoint, preferably the best checkpoint.
- Load the generated filtered dataset YAML.
- Explicitly validate using the `val` split.
- Do not evaluate against the training subset by mistake.
- Do not use the test set as a substitute for validation.
- Make the checkpoint path, dataset YAML path, confidence threshold, IoU threshold, and device configurable where supported.
- Save evaluation outputs and plots.
- Report errors when the checkpoint or validation data is missing.
- Verify that the model's class names and IDs match the target taxonomy.

### Metrics

Report actual available values for:

- Precision.
- Recall.
- mAP@50.
- mAP@50-95.
- Per-class metrics, where supported.
- Number of evaluated images and instances, where available.

Do not hardcode example metrics or fabricate results.

### Acceptance Criteria

- Validation runs against the generated `val` split.
- Metrics are produced by the actual model evaluation.
- The results identify the checkpoint and dataset configuration used.
- Validation outputs can be traced to the corresponding training run.

## 9. MLflow Integration

Integrate MLflow Tracking into the preparation, training, and validation workflow where appropriate.

### Training Run

Record:

- Model architecture and checkpoint identifier.
- Dataset configuration.
- Target class mapping.
- Train/validation split ratio.
- Random seed.
- Epoch count.
- Image size.
- Batch size.
- Learning rate, if applicable.
- Deterministic setting.
- Training duration, where available.
- Actual available training and validation metrics.

### Validation Run

Record:

- Checkpoint identifier.
- Dataset configuration.
- Evaluated split.
- Precision.
- Recall.
- mAP@50.
- mAP@50-95.
- Relevant evaluation artifacts.

Associate validation with the corresponding training run using MLflow run IDs, tags, or another explicit mechanism.

Avoid creating duplicate runs accidentally.

### Artifacts

Log useful, reasonably sized artifacts, such as:

- Training configuration.
- Filtered dataset metadata and split statistics.
- Validation reports.
- Confusion matrix and evaluation plots, when available.

Do not upload the full dataset or unnecessary large model files by default.

### Configuration

Support configurable settings, such as:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`

Provide a documented local setup and allow training to run without a remote MLflow server when configured to do so.

## 10. Dependency Management: `pyproject.toml`

Create or update the root `pyproject.toml`.

Requirements:

- Declare project metadata and supported Python versions.
- Declare the dependencies used by the actual implementation.
- Include compatible versions of Ultralytics and MLflow.
- Include other direct dependencies only when needed.
- Select and document compatible PyTorch and torchvision versions.
- Do not unintentionally force CPU-only PyTorch on NVIDIA GPU users.
- Document the supported CUDA/PyTorch installation procedure.
- Pin tested versions or use explicit compatibility constraints.
- Keep dependency documentation consistent with the actual code.

Verify dependency metadata and installation instructions where the environment permits.

## 11. Training Documentation: `training/README.md`

Create practical documentation covering:

1. Project prerequisites.
2. Environment setup and dependency installation.
3. NVIDIA GPU and PyTorch/CUDA configuration.
4. SH17 dataset acquisition from a verified source.
5. Dataset license and usage restrictions.
6. Required source dataset structure.
7. Filtering and class remapping.
8. Train/validation split configuration.
9. Target classes and their source-class mapping.
10. Dataset preparation command.
11. YOLOv9e checkpoint setup.
12. Training command.
13. Validation command.
14. MLflow server startup and UI access.
15. Output locations.
16. Troubleshooting common errors.

Document commands that match the implemented scripts.

Example command sequence, subject to the actual implementation:

```powershell
python -m pip install -e .
python training/prepare_dataset.py
python training/train.py
python training/val.py
```

If the implementation requires additional arguments, environment variables, or a separate MLflow startup command, document them accurately.

## 12. Repository Hygiene

Update `.gitignore` only when necessary.

Ensure the following are not committed accidentally:

- Original and filtered datasets.
- Dataset images and annotations.
- Large model checkpoints.
- Generated training runs.
- Video outputs.
- Local MLflow databases and artifacts.
- Secrets and machine-specific paths.

Keep scripts, configuration files, documentation, and lightweight tests in Git.

Do not remove existing ignore rules without a clear reason.

## 13. Testing and Verification

Implement appropriate lightweight tests for:

- Source-to-target class mapping.
- Removal of excluded classes.
- Annotation ID remapping.
- Annotation coordinate validation.
- Train/validation split disjointness.
- Split reproducibility.
- Generated dataset YAML correctness.
- Training configuration and fixed seed.
- MLflow parameter and metric logging.
- Validation configuration targeting the `val` split.

Use a small synthetic fixture for unit tests where possible. Do not require the full SH17 dataset or GPU for basic preprocessing tests.

After implementation:

1. Check Python syntax.
2. Validate YAML and TOML files.
3. Run available unit tests.
4. Verify that documented commands match the implementation.
5. Verify model and dataset class compatibility.
6. Report checks that could not be completed because of missing dependencies, data, network access, or hardware.

Do not launch a full YOLOv9e training job automatically unless explicitly requested.

Never fabricate successful test results or model metrics.

## 14. Expected Deliverables

Create or update the following, adapting to the existing repository:

- [ ] `training/prepare_dataset.py`
- [ ] `training/train.py`
- [ ] `training/val.py`
- [ ] Generated dataset configuration for the filtered six-class dataset
- [ ] Root `pyproject.toml`
- [ ] MLflow integration
- [ ] `training/README.md`
- [ ] `.gitignore`, if necessary
- [ ] Lightweight tests for dataset preparation and configuration

Avoid creating duplicate scripts when equivalent functionality already exists.

## 15. Final Report

After implementation, report:

1. Files created or modified.
2. Dataset source and verified source-to-target class mapping.
3. Number of images retained and excluded.
4. Train/validation image counts and class distributions, if the dataset was available.
5. YOLOv9e checkpoint and implementation used.
6. Dependency installation commands.
7. Dataset preparation, training, validation, and MLflow commands.
8. Tests executed and their actual results.
9. Any unresolved dataset, licensing, model compatibility, or hardware limitations.

Clearly distinguish implemented functionality from functionality that remains untested.

**Execution order:** Inspect repository → verify dataset metadata → filter and remap annotations → split dataset → verify generated YAML → train YOLOv9e → validate on the validation subset → log results to MLflow → run checks → document results.

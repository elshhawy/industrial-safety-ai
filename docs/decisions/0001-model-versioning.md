# 0001: Model versioning (proposal)

Status: **proposal**. The team has not reviewed it yet.

## Why

Anyone must be able to say which model the system runs, how it was
trained, and how to go back to the previous one.

## Proposal

1. **One registered model name per task.** Suggested name: `ppe-detector`.
   Model family and class set are still open team decisions.
2. **Every model version comes from a tracked MLflow run.** The run records:
   Git commit, dataset version (tag or pointer hash), parameters, metrics,
   and where it ran (Colab, Kaggle or a local machine).
3. **A model version contains:** the weights file, the ordered class list
   (`data.yaml`), the input size, and a link to its run.
4. **Aliases:**
   - `champion`: the version the real-time pipeline uses.
   - `challenger`: a candidate under evaluation.
5. **The pipeline loads by alias** (`models:/ppe-detector@champion`),
   never by a file path on someone's laptop.
6. **Promotion** means moving `champion` to a new version. It needs measured
   metrics on the same held-out test set as the current champion, with the
   data version and hardware stated. No measured numbers, no promotion.
7. **Rollback** means moving `champion` back to the previous version.
   No code change is needed.

## Tested so far

In a throwaway sandbox (MLflow 3.16.1, local SQLite): two versions were
registered, `champion` was set on version 2, and the model was loaded by
alias. The models and numbers there were fake and prove only the mechanism.

## Still pending

- Where MLflow runs when training happens on different machines.
- Where model weights are stored (MLflow artifacts or a DVC remote).
- Model family, class set, and final test set.
- Whether an FPS/latency "benchmark card" is stored with each version
  (only measured values, with the hardware named).

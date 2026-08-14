# Cataract EfficientNetB0 Baseline

This project contains a reproducible, fixed-split baseline for binary slit-lamp
image classification:

- `Normal` -> label `0`
- `Cataract` -> label `1`
- `Other` -> excluded

The source workbooks and images under `../Fixed Dataset/Clean` are read-only.
No random split, relabeling, cleaning, or dataset mutation is performed.

## Baseline design

- EfficientNetB0 with ImageNet weights and `include_top=False`
- frozen backbone, Global Average Pooling, 64-unit dense layer, dropout, sigmoid
- Binary Crossentropy and Adam (`1e-3` initial learning rate)
- validation-loss checkpointing, early stopping, and learning-rate reduction
- fixed seed `2026` for Python, NumPy, and TensorFlow
- deterministic validation/test ordering
- training-only conservative rotation, zoom, brightness, and contrast augmentation
- no horizontal flip and no fine-tuning in this first experiment

Images are decoded as RGB, resized to `224 x 224`, and retained as float32 values
in `[0, 255]`. TensorFlow/Keras EfficientNetB0 contains the single internal
`Rescaling(1/255)` operation, so no external normalization or
`preprocess_input` is applied.

## Structure

```text
configs/baseline.yaml       Normal-vs-All-Cataract experiment
configs/mild_cataract.yaml  Controlled Normal-vs-Mild-Cataract experiment
src/audit.py                Dataset integrity audit
src/task_audit.py           Label-filtered counts/leakage/duplicate audit
src/data.py                 Metadata selection and tf.data construction
src/model.py                EfficientNetB0 and conservative augmentation
src/train.py                Train/validation-only fitting and checkpoint selection
src/evaluate.py             Validation report and locked test evaluation
src/metrics.py              Binary metrics and diagnostic plots
src/sanity.py               Hard pre-training gates
src/compare.py              Completed-baseline comparison report
src/utils.py                Seeds, paths, hashes, and configuration
```

## Environment

The completed run used Python 3.11, TensorFlow 2.17.1, and Keras 3.15.1 on CPU.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproduction order

```powershell
.\.venv\Scripts\python.exe src\audit.py
.\.venv\Scripts\python.exe src\sanity.py --config configs\baseline.yaml
.\.venv\Scripts\python.exe src\train.py --config configs\baseline.yaml
.\.venv\Scripts\python.exe src\evaluate.py --config configs\baseline.yaml
```

`train.py` constructs only the official training and validation datasets. The
test workbook is used by the separate evaluator only after checkpoint selection.
The evaluator refuses to overwrite an existing locked-test result unless the
operator explicitly supplies `--allow-repeat`.

## Completed Normal-vs-All-Cataract run

- Best frozen-backbone checkpoint: epoch 3 of 9 run
- Validation: accuracy 66.7%, ROC-AUC 0.760
- Locked test: accuracy 71.4%, sensitivity 77.8%, specificity 50.0%, ROC-AUC 0.676
- Confusion matrix: TN=4, FP=4, FN=6, TP=21
- Verdict: `BASELINE WORKS WITH WARNINGS`

See `outputs/reports/baseline_results.txt` for the full broad-cataract report.

## Controlled Normal-vs-Mild-Cataract run

The Mild task changes only the label filter and experiment-specific output
paths. Its training protocol is identical to the first run.

```powershell
.\.venv\Scripts\python.exe src\task_audit.py --config configs\mild_cataract.yaml --reference-config configs\baseline.yaml
.\.venv\Scripts\python.exe src\sanity.py --config configs\mild_cataract.yaml
.\.venv\Scripts\python.exe src\train.py --config configs\mild_cataract.yaml
.\.venv\Scripts\python.exe src\evaluate.py --config configs\mild_cataract.yaml
.\.venv\Scripts\python.exe src\compare.py
```

- Train: 45 Normal + 54 Mild Cataract = 99
- Validation: 5 Normal + 7 Mild Cataract = 12
- Locked test: 8 Normal + 21 Mild Cataract = 29
- Best checkpoint: epoch 11 of 17 run
- Locked test: accuracy 44.8%, sensitivity 38.1%, specificity 62.5%, ROC-AUC 0.524
- Confusion matrix: TN=5, FP=3, FN=13, TP=8
- Verdict: `BASELINE UNRELIABLE`

See `outputs/mild_cataract/reports/baseline_results.txt` for every Mild-task
error and `outputs/reports/baseline_comparison.txt` for the controlled
comparison. This project still does not implement ROI localization,
segmentation, Grad-CAM, explainability, uncertainty estimation, fine-tuning,
architecture comparisons, optimizer comparisons, or threshold tuning.

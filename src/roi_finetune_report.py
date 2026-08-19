"""Assemble the final frozen-vs-fine-tuned ROI-224 controlled report."""

from __future__ import annotations

from pathlib import Path

from utils import PROJECT_ROOT, read_json


ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_finetune"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "frozen_vs_finetuned_report.txt"
VERDICT = "PARTIAL FINE-TUNING DOES NOT IMPROVE ROI BASELINE"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.1f} pp"


def metric_line(label: str, values: dict) -> str:
    key = label.casefold().replace("-", "_")
    return f"{label}: {values[key]:.4f} ({pct(values[key])})"


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite final report: {OUTPUT}")
    setup = read_json(REPORTS / "fine_tuning_setup.json")
    training = read_json(REPORTS / "training_summary.json")
    results = read_json(REPORTS / "locked_test_metrics.json")
    comparison = read_json(REPORTS / "frozen_vs_finetuned_metrics.json")
    transitions = read_json(REPORTS / "case_transition_summary.json")
    probability = read_json(REPORTS / "probability_analysis.json")
    gradcam = read_json(ROOT / "gradcam" / "finetuned_gradcam_summary.json")
    frozen = {row["metric"]: row["frozen_roi_224"] for row in comparison}
    fine = {row["metric"]: row["finetuned_roi_224"] for row in comparison}
    changes = {row["metric"]: row["absolute_difference"] for row in comparison}
    environment = setup["environment"]
    model = setup["model_setup"]
    validation = results["validation"]
    test = results["test"]
    trainable_layers = "\n".join(f"- {name}" for name in model["trainable_backbone_layer_names"])
    probability_text = (
        "Fine-tuning did not increase class separation. Both classes shifted slightly toward "
        "lower Mild probabilities, with a larger downward shift for Mild cases. Mean class "
        "separation decreased by 0.0030 and mean distance from the 0.5 threshold decreased by "
        "0.0036, so predictions became marginally less extreme. Histogram overlap decreased "
        "by 0.0595, but identical ROC-AUC/ranking performance and identical case labels mean "
        "this coarse-bin change should not be interpreted as improved discrimination."
    )
    gradcam_text = (
        "No map was lens/pupil dominant in either model. Fine-tuning changed only one map from "
        "diffuse/unclear to partially lens-related; reflection/illumination dominance remained "
        "6 cases and eyelid/background dominance remained 14. The +0.0024 central-mass and "
        "-0.0066 border-mass changes are negligible. Activation therefore remains dominated by "
        "border/eyelid, illumination/reflection, and diffuse patterns, with no persuasive shift "
        "toward lens-centered evidence."
    )
    report = f"""FROZEN ROI 224 VS PARTIALLY FINE-TUNED ROI 224
====================================================

RESEARCH QUESTION
-----------------
Does limited fine-tuning of the upper EfficientNetB0 layers improve Mild Cataract
discrimination beyond the frozen ROI baseline without substantially increasing
overfitting?

ENVIRONMENT
-----------
Operating system: {environment['ubuntu']} under WSL2 ({environment['kernel']})
Interpreter: {environment['interpreter']}
TensorFlow: {environment['tensorflow_version']}
CUDA build: {environment['cuda_build']} (CUDA {environment['cuda_version']}, cuDNN {environment['cudnn_version']})
GPU: {environment['gpu_details']['device_name']} at {environment['gpu_device']}
Verified GPU operation device: {environment['gpu_operation_device']}
NVIDIA status: {environment['nvidia_smi']}

EXPERIMENTAL CONTROLS
---------------------
The official train.xlsx, val.xlsx, and test.xlsx assignments, subject IDs, label
mapping, deterministic fixed center-square ROI, 224x224 input, EfficientNetB0
backbone, ImageNet initialization, learned classification head, augmentation,
batch size 8, seed 2026, binary crossentropy, threshold 0.5, callbacks, and
validation-loss checkpoint selection were retained from the frozen ROI baseline.
No class weighting, focal loss, threshold tuning, optimizer comparison, new
architecture, segmentation, resolution change, ROI change, or test-set tuning was
introduced. The saved frozen checkpoint was loaded successfully, preserving the
classification-head weights before selective unfreezing. The test set was not
loaded during training and received one performance prediction pass only after
validation-loss model selection.

SELECTIVE UNFREEZING AND PARAMETERS
-----------------------------------
Total backbone layers: {model['total_backbone_layers']}
Frozen backbone layers: {model['frozen_backbone_layers']} ({100*model['frozen_backbone_fraction']:.1f}%)
Trainable backbone layers: {model['trainable_backbone_layers']}
First trainable backbone layer: {model['first_trainable_backbone_layer']}
Last trainable backbone layer: {model['last_trainable_backbone_layer']}
First/last weighted trainable layers: {model['first_trainable_weighted_backbone_layer']} / {model['last_trainable_weighted_backbone_layer']}
Total parameters: {model['total_parameters']:,}
Trainable parameters: {model['trainable_parameters']:,}
Non-trainable parameters: {model['non_trainable_parameters']:,}

Exact trainable backbone layers:
{trainable_layers}

BATCHNORMALIZATION STRATEGY
---------------------------
{model['batch_normalization_implementation']}
All {model['batch_normalization_layer_count']} BatchNormalization layers remained frozen; zero were
trainable. A pre-training training-mode probe and a post-training comparison both
confirmed unchanged moving means and variances.

OPTIMIZATION AND TRAINING BEHAVIOR
----------------------------------
Optimizer: Adam
Initial learning rate: {training['initial_learning_rate']:.8f}
Final learning rate: {training['final_learning_rate']:.8f}
Loss: binary crossentropy
Maximum epochs: {training['epochs_requested']}
Epochs run: {training['epochs_run']}
Best epoch: {training['best_epoch']}
Minimum validation loss: {training['minimum_validation_loss']:.6f}
EarlyStopping patience: {training['early_stopping_patience']}, restore_best_weights=True
ReduceLROnPlateau patience: {training['reduce_lr_patience']}, factor=0.2, minimum LR=0.000001

Validation loss was best at epoch 1 and never improved afterward. ReduceLROnPlateau
reduced the LR after epochs 4 and 7, and EarlyStopping restored epoch 1. The earlier
5%-divergence heuristic did not flag overfitting because validation loss changed
only modestly, but rising/variable training discrimination without validation-loss
improvement indicates immediate saturation and possible small-sample overfitting.

SELECTED-CHECKPOINT VALIDATION METRICS
--------------------------------------
Accuracy: {validation['accuracy']:.4f} ({pct(validation['accuracy'])})
Precision: {validation['precision']:.4f} ({pct(validation['precision'])})
Sensitivity: {validation['sensitivity']:.4f} ({pct(validation['sensitivity'])})
Specificity: {validation['specificity']:.4f} ({pct(validation['specificity'])})
F1: {validation['f1']:.4f} ({pct(validation['f1'])})
ROC-AUC: {validation['roc_auc']:.4f}
TN={validation['tn']}, FP={validation['fp']}, FN={validation['fn']}, TP={validation['tp']}

LOCKED TEST METRICS AT THRESHOLD 0.5
------------------------------------
Accuracy: {test['accuracy']:.4f} ({pct(test['accuracy'])})
Precision: {test['precision']:.4f} ({pct(test['precision'])})
Sensitivity: {test['sensitivity']:.4f} ({pct(test['sensitivity'])})
Specificity: {test['specificity']:.4f} ({pct(test['specificity'])})
F1: {test['f1']:.4f} ({pct(test['f1'])})
ROC-AUC: {test['roc_auc']:.4f}
TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}

FROZEN VS FINE-TUNED COMPARISON
-------------------------------
Metric          Frozen ROI 224   Fine-tuned ROI 224   Absolute / pp change
Accuracy        {pct(frozen['accuracy']):<16} {pct(fine['accuracy']):<20} {changes['accuracy']:+.4f} / {pp(changes['accuracy'])}
Precision       {pct(frozen['precision']):<16} {pct(fine['precision']):<20} {changes['precision']:+.4f} / {pp(changes['precision'])}
Sensitivity     {pct(frozen['sensitivity']):<16} {pct(fine['sensitivity']):<20} {changes['sensitivity']:+.4f} / {pp(changes['sensitivity'])}
Specificity     {pct(frozen['specificity']):<16} {pct(fine['specificity']):<20} {changes['specificity']:+.4f} / {pp(changes['specificity'])}
F1              {pct(frozen['f1']):<16} {pct(fine['f1']):<20} {changes['f1']:+.4f} / {pp(changes['f1'])}
ROC-AUC         {frozen['roc_auc']:<16.4f} {fine['roc_auc']:<20.4f} {changes['roc_auc']:+.4f} / {pp(changes['roc_auc'])}
TN              {int(frozen['tn']):<16} {int(fine['tn']):<20} {int(changes['tn']):+d}
FP              {int(frozen['fp']):<16} {int(fine['fp']):<20} {int(changes['fp']):+d}
FN              {int(frozen['fn']):<16} {int(fine['fn']):<20} {int(changes['fn']):+d}
TP              {int(frozen['tp']):<16} {int(fine['tp']):<20} {int(changes['tp']):+d}

Primary interpretation: ROC-AUC, sensitivity, false-negative count, specificity,
and F1 were all exactly unchanged. Accuracy is also unchanged but is not the basis
of the verdict. Limited fine-tuning produced no measurable locked-test improvement.

CASE-LEVEL TRANSITIONS
----------------------
FN→TP: {transitions.get('FN→TP', 0)}
TP→FN: {transitions.get('TP→FN', 0)}
FP→TN: {transitions.get('FP→TN', 0)}
TN→FP: {transitions.get('TN→FP', 0)}
Unchanged correct: {transitions.get('unchanged correct', 0)}
Unchanged incorrect: {transitions.get('unchanged incorrect', 0)}
Every thresholded prediction was unchanged. There were no gains offset by
regressions and no case-level evidence of improved consistency.

PROBABILITY-DISTRIBUTION ANALYSIS
---------------------------------
Frozen Normal/Mild means: {probability['frozen']['normal_mean']:.4f} / {probability['frozen']['mild_mean']:.4f}
Fine-tuned Normal/Mild means: {probability['finetuned']['normal_mean']:.4f} / {probability['finetuned']['mild_mean']:.4f}
Normal paired mean shift: {probability['normal_mean_paired_probability_shift']:+.4f}
Mild paired mean shift: {probability['mild_mean_paired_probability_shift']:+.4f}
Class-mean separation change: {probability['mean_class_separation_change']:+.4f}
Histogram-overlap change: {probability['histogram_overlap_change']:+.4f}
Mean absolute margin-from-0.5 change: {probability['mean_absolute_margin_change']:+.4f}
{probability_text}

GRAD-CAM COMPARISON
-------------------
Method: {gradcam['methodology']}.
Frozen category counts: {gradcam['frozen_roi_category_counts']}
Fine-tuned category counts: {gradcam['finetuned_roi_category_counts']}
Category changes: {gradcam['category_count_changes']}
Mean central-proxy attention change: {gradcam['central_attention_change']:+.4f}
Mean border attention change: {gradcam['border_attention_change']:+.4f}
{gradcam_text}
Grad-CAM was generated after test evaluation and was not used for model selection.
Its fixed center/border/brightness heuristics and 7x7 map are descriptive, not
anatomical ground truth or evidence of clinical causality.

LIMITATIONS
-----------
- The training set is small: 99 usable images (45 Normal, 54 Mild Cataract).
- The validation set is very small: 12 images; one case changes a metric sharply.
- The locked test set is small: 29 images, including only 8 Normal cases.
- This is one random seed, so stochastic fine-tuning variability is unknown.
- The ROI is heuristic and partially truncates anatomy in previously audited cases.
- Illumination, reflection, eyelid, border, and acquisition cues remain possible shortcuts.
- The research workflow has repeated study-level exposure to the same test set across
  completed experiments. Although this run used it once only after selection, the broader
  sequence increases researcher-overfitting risk and limits the strength of confirmatory claims.
- Grad-CAM is coarse and post-hoc; it cannot prove that clinically causal lens features were used.

FINAL VERDICT
-------------
{VERDICT}

The verdict is driven primarily by identical ROC-AUC, sensitivity, false-negative
count, specificity, and F1; zero case transitions; slightly weaker probability
separation; no meaningful anatomical-attention shift; and validation loss selecting
the first fine-tuning epoch with no later improvement.

EXACTLY ONE NEXT CONTROLLED EXPERIMENT
--------------------------------------
Recommend one pre-registered five-seed replication of the identical frozen-versus-
partial-fine-tuning ROI-224 protocol, with all hyperparameters fixed in advance and
paired mean/dispersion reported; do not inspect the current locked-test outcomes
between seeds. This directly tests whether the single-seed null result is stable.
This recommendation is not implemented here.
"""
    OUTPUT.write_text(report, encoding="utf-8")
    print(VERDICT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

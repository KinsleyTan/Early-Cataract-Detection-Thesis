"""Assemble the final controlled ROI experiment report and integrity verification."""

from __future__ import annotations

import json
from pathlib import Path

from roi_audit import protected_manifest
from utils import PROJECT_ROOT, read_json, write_json


ROI_ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment"
REPORTS = ROI_ROOT / "reports"
FULL_METRICS = PROJECT_ROOT / "outputs" / "mild_cataract" / "reports" / "baseline_metrics.json"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.1f} percentage points"


def metric_line(name: str, metrics: dict) -> str:
    return f"{name}: {metrics[name]:.4f} ({pct(metrics[name])})"


def main() -> None:
    audit = read_json(REPORTS / "roi_audit.json")
    training = read_json(REPORTS / "training_summary.json")
    roi_result = read_json(REPORTS / "baseline_metrics.json")
    full_result = read_json(FULL_METRICS)
    transitions = read_json(REPORTS / "case_transition_summary.json")
    gradcam = read_json(ROI_ROOT / "gradcam" / "roi_gradcam_summary.json")
    before = read_json(REPORTS / "protected_artifact_hashes_before.json")
    after = protected_manifest()
    changed = {
        path: {"before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    }
    integrity = {
        "protected_file_count_before": len(before),
        "protected_file_count_after": len(after),
        "all_protected_artifacts_unchanged": not changed,
        "changed_or_missing_files": changed,
    }
    write_json(REPORTS / "protected_artifact_integrity_after.json", integrity)
    if changed:
        raise RuntimeError(f"Protected baseline/Grad-CAM artifacts changed: {changed}")

    full = full_result["test"]
    roi = roi_result["test"]
    validation = roi_result["validation"]
    overall_roi_focus = gradcam["roi_overall_category_counts"]
    overall_full_focus = gradcam["full_image_overall_category_counts"]
    hypothesis = "PARTIALLY SUPPORTED"
    verdict = "ROI IMPROVES THE BASELINE"
    recommendation = (
        "one controlled higher-input-resolution experiment on the same fixed ROI "
        "(224x224 versus 320x320), keeping the backbone family, splits, labels, optimizer, "
        "threshold, and all other protocol elements fixed"
    )
    failures = "\n".join(f"- {item}" for item in audit["visually_identified_failures"])
    report = f"""FULL IMAGE VS PUPIL/LENS ROI: NORMAL VS MILD CATARACT
============================================================

EXPERIMENTAL SAFEGUARDS
-----------------------
The completed full-image baseline was not retrained. Its saved locked metrics and
predictions were used as the comparator. The ROI experiment retained the official
train/validation/test workbooks, subject assignments, label policy, EfficientNetB0
architecture, ImageNet initialization, frozen backbone, classification head, Adam
optimizer, learning rate, batch size, seed, augmentation, callbacks, 0.5 threshold,
and metrics. Only the deterministic input representation changed. The ROI test set
was evaluated once after validation-loss checkpoint selection. Grad-CAM was run
only after the locked evaluation and did not influence training or selection.

A. ROI METHOD
-------------
Method: deterministic label-independent fixed center-square crop.
Source images: 4032x3024 RGB.
ROI coordinates: left=928, top=424, right=3105, bottom=2601.
ROI size: 2177x2177, equal to 72% of the source short edge, centered at (0.50, 0.50).
The crop is applied before the same bilinear resize to 224x224 and float32 [0,255]
pipeline. No external normalization was added; EfficientNetB0 retains its single
internal Rescaling(1/255). The ROI rule never reads labels or metadata.

B. ROI AUDIT
------------
Audit scope: all {audit['train_images_reviewable']} usable training images, including Normal and Mild Cataract
under Diffuse, Direct Focal, and Retro Illumination. Parameters were chosen using
training images only. The pupil/lens region was adequately retained in 96/99 images
(97.0%). Three laterally framed Direct Focal images showed partial truncation:
{failures}
This 3.0% limitation was recorded explicitly; it was not substantial enough to
invalidate the simple controlled baseline.

C. TRAINING BEHAVIOR
--------------------
Best checkpoint epoch: {training['best_epoch']} of {training['epochs_run']} epochs run (30 maximum).
Best validation loss: {training['best_validation_loss']:.4f}.
Best logged validation accuracy: {training['best_validation_accuracy']:.4f} ({pct(training['best_validation_accuracy'])}).
The predefined history heuristic marked overfitting: {training['overfitting_observed']}.
Validation loss was best at epoch 2 while training performance continued to rise;
early stopping restored epoch 2. No fine-tuning or test-driven decision was used.

D. ROI VALIDATION METRICS
-------------------------
{metric_line('accuracy', validation)}
{metric_line('precision', validation)}
{metric_line('sensitivity', validation)}
{metric_line('specificity', validation)}
{metric_line('f1', validation)}
ROC-AUC: {validation['roc_auc']:.4f}
TN={validation['tn']}, FP={validation['fp']}, FN={validation['fn']}, TP={validation['tp']}
The 12-image validation set is too small for stable estimates; its 100% sensitivity
coincided with only 20% specificity.

E. LOCKED ROI TEST METRICS
--------------------------
{metric_line('accuracy', roi)}
{metric_line('precision', roi)}
{metric_line('sensitivity', roi)}
{metric_line('specificity', roi)}
{metric_line('f1', roi)}
ROC-AUC: {roi['roc_auc']:.4f}
TN={roi['tn']}, FP={roi['fp']}, FN={roi['fn']}, TP={roi['tp']}

F. FULL IMAGE VS ROI COMPARISON
-------------------------------
Metric          Full image     ROI            Change
Accuracy        {pct(full['accuracy']):<14} {pct(roi['accuracy']):<14} {pp(roi['accuracy'] - full['accuracy'])}
Precision       {pct(full['precision']):<14} {pct(roi['precision']):<14} {pp(roi['precision'] - full['precision'])}
Sensitivity     {pct(full['sensitivity']):<14} {pct(roi['sensitivity']):<14} {pp(roi['sensitivity'] - full['sensitivity'])}
Specificity     {pct(full['specificity']):<14} {pct(roi['specificity']):<14} {pp(roi['specificity'] - full['specificity'])}
F1              {pct(full['f1']):<14} {pct(roi['f1']):<14} {pp(roi['f1'] - full['f1'])}
ROC-AUC         {full['roc_auc']:.4f}         {roi['roc_auc']:.4f}         {roi['roc_auc'] - full['roc_auc']:+.4f} ({pp(roi['roc_auc'] - full['roc_auc'])})
False negatives {full['fn']:<14} {roi['fn']:<14} {roi['fn'] - full['fn']:+d}
False positives {full['fp']:<14} {roi['fp']:<14} {roi['fp'] - full['fp']:+d}

The primary outcomes all improved: sensitivity increased by 47.6 points, ROC-AUC
increased by 7.7 points, and false negatives fell by 10. The tradeoff was a 12.5-
point specificity decrease and one additional false positive. ROC-AUC remains only
0.601, so discrimination is still weak despite the fixed-threshold sensitivity gain.

G. CASE TRANSITIONS
-------------------
FN -> TP improvements: {transitions.get('FN -> TP improvement', 0)}
TP -> FN regressions: {transitions.get('TP -> FN regression', 0)}
FP -> TN improvements: {transitions.get('FP -> TN improvement', 0)}
TN -> FP regressions: {transitions.get('TN -> FP regression', 0)}
Stable TP: {transitions.get('stable TP', 0)}; stable FN: {transitions.get('stable FN', 0)};
stable TN: {transitions.get('stable TN', 0)}; stable FP: {transitions.get('stable FP', 0)}.
The ROI recovered ten previously missed Mild cases without losing any original TP.
It did not correct any original FP and converted 1841URPSL (3).jpg from TN to FP.

H. GRAD-CAM COMPARISON
----------------------
Full-image focus categories: {overall_full_focus}
ROI focus categories: {overall_roi_focus}
Mean central-proxy attention changed from
{gradcam['full_image_mean_central_attention_mass']:.3f} to {gradcam['roi_mean_central_attention_mass']:.3f} ({gradcam['central_attention_change']:+.3f}).
Mean border attention changed from {gradcam['full_image_mean_border_attention_mass']:.3f} to
{gradcam['roi_mean_border_attention_mass']:.3f} ({gradcam['border_attention_change']:+.3f}).
Neither model produced a map categorized as lens/pupil dominant. ROI edge/background
dominance increased from 7 to 14 maps, while partial lens-related maps fell from 4
to 1. Visual review shows continued activation on slit illumination, reflections,
crop edges, and non-central structures. Therefore Grad-CAM does not show improved
anatomical relevance, even though classification sensitivity improved.

I. INTERPRETATION
-----------------
Hypothesis status: {hypothesis}.
Restricting the input strongly improved Mild sensitivity and reduced missed cases
at the locked 0.5 threshold, with a smaller improvement in ranking discrimination.
However, specificity worsened, ROC-AUC remains modest, and Grad-CAM did not become
more lens/pupil centered. The result supports ROI as a useful input constraint for
this dataset, but does not establish that the model learned clinically causal lens
features. The probability shift may partly reflect retained illumination or crop-
boundary cues.

J. LIMITATIONS
--------------
- Only 99 training, 12 validation, and 29 test images were usable.
- The test set contains only 8 Normal images; one image changes specificity by 12.5 points.
- The fixed heuristic ROI partially truncates anatomy in 3/99 audited train images.
- Illumination and reflection signals remain inside the ROI and can still act as shortcuts.
- No pupil/lens segmentation ground truth or clinician localization annotations exist.
- Grad-CAM is coarse (7x7), post-hoc, and cannot prove clinical causality.
- The full and ROI models are paired by split/protocol but each stochastic training run is a single seed.

FINAL VERDICT
-------------
{verdict}
The verdict is driven by the large sensitivity gain, 10 fewer false negatives, and
improved ROC-AUC, while explicitly retaining warnings about lower specificity,
weak absolute ROC-AUC, small samples, and non-anatomical Grad-CAM patterns.

EXACTLY ONE NEXT EXPERIMENT
---------------------------
Recommend {recommendation}. This tests whether preserving finer texture inside the
already controlled ROI improves ROC-AUC and specificity without adding a new
architecture, threshold tuning, segmentation, or fine-tuning. Do not implement it
as part of this completed experiment.
"""
    (REPORTS / "roi_comparison_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": verdict, "hypothesis": hypothesis, "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()

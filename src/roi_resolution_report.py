"""Write the final controlled ROI 224x224 versus ROI 320x320 experiment report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from utils import PROJECT_ROOT, read_json


ROOT_224 = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment"
ROOT_320 = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_resolution_320"
REPORTS = ROOT_320 / "reports"


def rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def metric_block(metrics: dict[str, Any]) -> list[str]:
    return [
        f"Accuracy: {metrics['accuracy']:.4f} ({rate(metrics['accuracy'])})",
        f"Precision: {metrics['precision']:.4f} ({rate(metrics['precision'])})",
        f"Sensitivity: {metrics['sensitivity']:.4f} ({rate(metrics['sensitivity'])})",
        f"Specificity: {metrics['specificity']:.4f} ({rate(metrics['specificity'])})",
        f"F1: {metrics['f1']:.4f} ({rate(metrics['f1'])})",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}",
    ]


def names_by_transition(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        result[row["transition_category"]].append(row["filename"])
    return dict(result)


def main() -> None:
    environment = read_json(REPORTS / "environment_check.json")
    audit = read_json(REPORTS / "resolution_audit.json")
    compatibility = read_json(REPORTS / "model_compatibility.json")
    metrics_224 = read_json(ROOT_224 / "reports" / "baseline_metrics.json")
    metrics_320 = read_json(REPORTS / "baseline_metrics.json")
    comparison = read_json(REPORTS / "metric_comparison_rows.json")
    transitions = read_json(REPORTS / "case_transition_rows.json")
    transition_summary = read_json(REPORTS / "case_transition_summary.json")
    training = read_json(REPORTS / "training_comparison.json")
    gradcam = read_json(ROOT_320 / "gradcam" / "roi320_gradcam_summary.json")
    names = names_by_transition(transitions)

    comparison_by_metric = {row["metric"]: row for row in comparison}
    t224 = training["roi_224"]
    t320 = training["roi_320"]
    b224 = t224["best_epoch_metrics"]
    b320 = t320["best_epoch_metrics"]
    f224 = t224["final_epoch_metrics"]
    f320 = t320["final_epoch_metrics"]
    test224 = metrics_224["test"]
    test320 = metrics_320["test"]
    val320 = metrics_320["validation"]

    verdict = "320x320 DOES NOT IMPROVE ROI BASELINE"
    lines = [
        "ROI 224x224 VS ROI 320x320 CONTROLLED EXPERIMENT",
        "=" * 88,
        f"FINAL VERDICT: {verdict}",
        "",
        "A. EXPERIMENTAL SETUP",
        "-" * 88,
        "The only intended scientific variable was the ROI resize: 224x224 -> 320x320 RGB.",
        "The crop itself remained exactly (left=928, top=424, right=3105, bottom=2601).",
        "Official train/validation/test workbooks, subjects, labels, EfficientNetB0/ImageNet ",
        "frozen-backbone policy, classification head, Adam learning rate, batch size, seed,",
        "augmentation, callbacks, validation-loss selection, and threshold 0.5 were unchanged.",
        f"Protocol audit passed: {audit['only_intended_variable_changed']}.",
        "The ROI-224 model was not retrained; its saved locked result was used directly.",
        "No test data was used during fitting or model selection.",
        f"Model input/output: {compatibility['input_shape']} -> {compatibility['output_shape']}.",
        f"Parameters unchanged: {compatibility['actual_parameter_counts_roi_320']}.",
        "Preprocessing: fixed ROI -> bilinear resize -> float32 [0,255]; EfficientNetB0's",
        "internal Rescaling(1/255) is the only normalization, so normalization was not duplicated.",
        "",
        "B. ENVIRONMENT",
        "-" * 88,
        f"Execution mode: {'native Windows' if environment['native_windows'] else 'WSL2' if environment['wsl2'] else environment['operating_system']['system']}.",
        f"Python: {environment['python_version'].splitlines()[0]}",
        f"TensorFlow: {environment['tensorflow_version']}; Keras: {environment['keras_version']}.",
        f"TensorFlow devices: {environment['tensorflow_physical_devices']}",
        f"TensorFlow GPU devices: {environment['tensorflow_gpu_devices']}",
        f"NVIDIA device visible to OS: {environment['nvidia_gpu_name']} (driver {environment['nvidia_driver_version']}, 6141 MiB VRAM).",
        "The run used CPU because TensorFlow 2.17.1 has no native-Windows CUDA support and",
        "this installed TensorFlow build is not CUDA-enabled. No environment was switched.",
        "",
        "C. TRAINING BEHAVIOR",
        "-" * 88,
        f"ROI-224: best epoch {t224['best_epoch']}/{t224['epochs_run']}; at best epoch train acc/loss={b224['training_accuracy']:.4f}/{b224['training_loss']:.4f}, val acc/loss={b224['validation_accuracy']:.4f}/{b224['validation_loss']:.4f}.",
        f"ROI-320: best epoch {t320['best_epoch']}/{t320['epochs_run']}; at best epoch train acc/loss={b320['training_accuracy']:.4f}/{b320['training_loss']:.4f}, val acc/loss={b320['validation_accuracy']:.4f}/{b320['validation_loss']:.4f}.",
        f"Best-epoch accuracy divergence (train-val): 224={b224['accuracy_divergence_train_minus_validation']:+.4f}, 320={b320['accuracy_divergence_train_minus_validation']:+.4f}.",
        f"Best-epoch loss divergence (val-train): 224={b224['loss_divergence_validation_minus_train']:+.4f}, 320={b320['loss_divergence_validation_minus_train']:+.4f}.",
        f"Final loss divergence (val-train): 224={f224['loss_divergence_validation_minus_train']:+.4f}, 320={f320['loss_divergence_validation_minus_train']:+.4f}.",
        "Both runs triggered the predefined overfitting heuristic. At the selected epoch, ROI-320",
        "already had a larger train-validation gap and worse validation loss than ROI-224.",
        "Higher resolution increased neither stable validation discrimination nor generalization.",
        "",
        "D. ROI-320 VALIDATION METRICS",
        "-" * 88,
        *metric_block(val320),
        "Validation has only 12 usable images (5 Normal, 7 Mild Cataract), so these values are unstable.",
        "",
        "E. LOCKED ROI-320 TEST METRICS",
        "-" * 88,
        *metric_block(test320),
        "The locked test set was evaluated once after checkpoint selection.",
        "",
        "F. ROI-224 VS ROI-320 METRIC COMPARISON",
        "-" * 88,
        "metric        ROI-224     ROI-320     absolute change     percentage-point change",
    ]
    for metric in ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"):
        row = comparison_by_metric[metric]
        lines.append(
            f"{metric:<12} {row['roi_224']:>9.4f}   {row['roi_320']:>9.4f}   "
            f"{row['absolute_change']:>+15.4f}   {row['percentage_point_change']:>+9.2f} pp"
        )
    lines.extend(
        [
            f"False negatives: {test224['fn']} -> {test320['fn']} ({test320['fn'] - test224['fn']:+d}).",
            f"False positives: {test224['fp']} -> {test320['fp']} ({test320['fp'] - test224['fp']:+d}).",
            "Primary endpoints all fail to support improvement: sensitivity decreased 19.05 pp,",
            "ROC-AUC decreased 0.006, and false negatives increased by four.",
            "",
            "G. CASE-LEVEL TRANSITIONS",
            "-" * 88,
            f"Transition counts: {transition_summary}",
            f"FN->TP improvements ({len(names.get('FN -> TP improvement', []))}): {names.get('FN -> TP improvement', []) or 'none'}",
            f"TP->FN regressions ({len(names.get('TP -> FN regression', []))}): {names.get('TP -> FN regression', []) or 'none'}",
            f"FP->TN improvements ({len(names.get('FP -> TN improvement', []))}): {names.get('FP -> TN improvement', []) or 'none'}",
            f"TN->FP regressions ({len(names.get('TN -> FP regression', []))}): {names.get('TN -> FP regression', []) or 'none'}",
            "The dominant harmful transition is TP->FN: five previously detected Mild cases fell",
            "below threshold, while only one previous false negative became a true positive.",
            "",
            "H. GRAD-CAM COMPARISON",
            "-" * 88,
            f"Layer: EfficientNetB0 top_conv; feature grid 224={gradcam['roi_224_layer_shape']}, 320={gradcam['roi_320_layer_shape']}.",
            f"ROI-224 focus categories: {gradcam['roi_224_overall_category_counts']}",
            f"ROI-320 focus categories: {gradcam['roi_320_overall_category_counts']}",
            f"Mean central attention: {gradcam['roi_224_mean_central_attention_mass']:.3f} -> {gradcam['roi_320_mean_central_attention_mass']:.3f} ({gradcam['central_attention_change_320_minus_224']:+.3f}).",
            f"Mean border attention: {gradcam['roi_224_mean_border_attention_mass']:.3f} -> {gradcam['roi_320_mean_border_attention_mass']:.3f} ({gradcam['border_attention_change_320_minus_224']:+.3f}).",
            "Border attention decreased, but central attention also decreased and lens/pupil-dominant",
            "cases remained zero. ROI-320 had more diffuse/unclear maps and still substantial",
            "reflection/illumination or eyelid/background categories. This is not meaningful evidence",
            "of improved anatomical focus or reduced shortcut reliance.",
            "Grad-CAM was generated only after locked evaluation and was not used for tuning.",
            "",
            "I. INTERPRETATION",
            "-" * 88,
            verdict,
            "At the fixed threshold, higher input resolution materially worsened Mild Cataract recall",
            "and increased missed cases. ROC-AUC was essentially unchanged but slightly lower, so the",
            "result is not merely a threshold shift with better ranking. Case transitions, validation",
            "behavior, and Grad-CAM provide no consistent compensating benefit. The main failure pattern",
            "is regression of previously detected Mild cases to false negatives, accompanied by continued",
            "non-anatomical or diffuse attention. Accuracy is not used as the deciding measure.",
            "",
            "J. LIMITATIONS",
            "-" * 88,
            "- Validation is very small: 12 usable images.",
            "- Locked test is small and imbalanced: 29 images (8 Normal, 21 Mild Cataract).",
            "- The ROI is a fixed heuristic crop, not expert pupil/lens segmentation.",
            "- Shortcut learning from illumination, reflection, eyelids, or acquisition artifacts remains possible.",
            "- Grad-CAM is coarse (7x7 vs 10x10 final feature grids), exploratory, and not causal proof.",
            "- Only one random seed was evaluated; no uncertainty interval or repeated-seed estimate is available.",
            "",
            "EXACTLY ONE RECOMMENDED NEXT CONTROLLED EXPERIMENT",
            "-" * 88,
            "Run controlled fine-tuning of only the final EfficientNetB0 blocks using the better ROI-224",
            "configuration as the fixed comparator, with all splits, ROI, threshold, and other protocol",
            "elements held constant. This tests whether domain adaptation—not more input pixels—is needed",
            "for subtle Mild Cataract features. Do not implement this recommendation in the current task.",
        ]
    )
    output = REPORTS / "roi_224_vs_320_report.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

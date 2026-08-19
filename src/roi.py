"""Deterministic, label-independent pupil/lens ROI utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RoiBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def validate_roi_config(roi_config: dict[str, Any]) -> None:
    if roi_config.get("method") != "fixed_center_square":
        raise ValueError("Only the controlled fixed_center_square ROI is supported")
    side_fraction = float(roi_config["side_fraction_of_short_edge"])
    center_x = float(roi_config["center_x_fraction"])
    center_y = float(roi_config["center_y_fraction"])
    if not 0.25 <= side_fraction <= 1.0:
        raise ValueError("ROI side fraction must be between 0.25 and 1.0")
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError("ROI center fractions must be in [0, 1]")


def roi_box_for_dimensions(
    width: int, height: int, roi_config: dict[str, Any]
) -> RoiBox:
    """Calculate the same integer square crop used by Pillow and TensorFlow."""
    validate_roi_config(roi_config)
    side = max(
        1,
        int(round(min(width, height) * float(roi_config["side_fraction_of_short_edge"]))),
    )
    center_x = float(roi_config["center_x_fraction"]) * width
    center_y = float(roi_config["center_y_fraction"]) * height
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return RoiBox(left=left, top=top, right=left + side, bottom=top + side)


def crop_pil(image: Image.Image, roi_config: dict[str, Any]) -> tuple[Image.Image, RoiBox]:
    box = roi_box_for_dimensions(image.width, image.height, roi_config)
    return image.crop(box.as_tuple()), box


def apply_roi_tensor(image, roi_config: dict[str, Any]):
    """Apply the fixed integer crop to an HxWxC TensorFlow image tensor."""
    import tensorflow as tf

    validate_roi_config(roi_config)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    side = tf.cast(
        tf.round(
            tf.cast(tf.minimum(width, height), tf.float32)
            * float(roi_config["side_fraction_of_short_edge"])
        ),
        tf.int32,
    )
    center_x = tf.cast(
        tf.round(tf.cast(width, tf.float32) * float(roi_config["center_x_fraction"])),
        tf.int32,
    )
    center_y = tf.cast(
        tf.round(tf.cast(height, tf.float32) * float(roi_config["center_y_fraction"])),
        tf.int32,
    )
    left = tf.clip_by_value(center_x - side // 2, 0, width - side)
    top = tf.clip_by_value(center_y - side // 2, 0, height - side)
    return tf.image.crop_to_bounding_box(image, top, left, side, side)


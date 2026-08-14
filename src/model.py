"""EfficientNetB0 baseline model construction."""

from __future__ import annotations

from typing import Any


def build_augmentation(config: dict[str, Any]):
    import tensorflow as tf

    aug = config["augmentation"]
    seed = int(config["experiment"]["seed"])
    layers = []
    if aug.get("enabled", True):
        layers.extend(
            [
                tf.keras.layers.RandomRotation(
                    factor=float(aug["rotation_factor"]),
                    fill_mode="reflect",
                    seed=seed + 1,
                    name="small_rotation",
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=(-float(aug["zoom_factor"]), float(aug["zoom_factor"])),
                    width_factor=(-float(aug["zoom_factor"]), float(aug["zoom_factor"])),
                    fill_mode="reflect",
                    seed=seed + 2,
                    name="small_zoom",
                ),
                tf.keras.layers.RandomBrightness(
                    factor=float(aug["brightness_factor"]),
                    value_range=(0.0, 255.0),
                    seed=seed + 3,
                    name="mild_brightness",
                ),
                tf.keras.layers.RandomContrast(
                    factor=float(aug["contrast_factor"]),
                    value_range=(0.0, 255.0),
                    seed=seed + 4,
                    name="mild_contrast",
                ),
            ]
        )
        if aug.get("horizontal_flip", False):
            layers.append(
                tf.keras.layers.RandomFlip(
                    mode="horizontal", seed=seed + 5, name="horizontal_flip"
                )
            )
    return tf.keras.Sequential(layers, name="training_augmentation")


def build_model(config: dict[str, Any]):
    import tensorflow as tf

    image_height, image_width = [int(value) for value in config["data"]["image_size"]]
    model_cfg = config["model"]
    if model_cfg["architecture"] != "EfficientNetB0":
        raise ValueError("This baseline supports only EfficientNetB0")
    if model_cfg.get("include_top", False):
        raise ValueError("The baseline requires include_top=False")

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=model_cfg["weights"],
        input_shape=(image_height, image_width, 3),
    )
    backbone.trainable = bool(model_cfg.get("backbone_trainable", False))

    inputs = tf.keras.Input((image_height, image_width, 3), name="rgb_image_0_255")
    x = build_augmentation(config)(inputs)
    # EfficientNetB0 in TensorFlow/Keras 2.17/3.15 includes Rescaling(1/255)
    # internally. Calling the frozen backbone with training=False also keeps all
    # BatchNormalization layers in inference mode.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(
        int(model_cfg["dense_units"]),
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(
            float(model_cfg["l2_regularization"])
        ),
        name="classification_dense",
    )(x)
    x = tf.keras.layers.Dropout(
        float(model_cfg["dropout_rate"]), name="classification_dropout"
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="cataract_probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_baseline")
    return model, backbone


def compile_model(model, config: dict[str, Any]) -> None:
    import tensorflow as tf

    train_cfg = config["training"]
    if train_cfg["optimizer"] != "Adam":
        raise ValueError("This baseline supports only the Adam optimizer")
    threshold = float(train_cfg["threshold"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(train_cfg["learning_rate"])),
        loss=tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy"),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
            tf.keras.metrics.Precision(name="precision", thresholds=threshold),
            tf.keras.metrics.Recall(name="sensitivity", thresholds=threshold),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        ],
    )


def parameter_counts(model) -> tuple[int, int]:
    import tensorflow as tf

    trainable = int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(
        sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)
    )
    return trainable, non_trainable


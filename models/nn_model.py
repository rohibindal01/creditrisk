"""
models/nn_model.py
Deep Neural Network for tabular credit risk classification.
Includes:
  - Configurable architecture with residual connections
  - Keras Tuner HyperBand search
  - Class-weight handling for imbalanced data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras_tuner as kt
import os


# ─────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────

def residual_block(x, units, dropout_rate, l2_reg):
    shortcut = x
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut if dimensions mismatch
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, use_bias=False)(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    return x


# ─────────────────────────────────────────────
# Fixed model builder
# ─────────────────────────────────────────────

def build_model(
    n_features: int,
    units_1: int = 256,
    units_2: int = 128,
    units_3: int = 64,
    dropout: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
    use_residual: bool = True,
) -> Model:
    inputs = keras.Input(shape=(n_features,), name="features")

    x = layers.Dense(units_1, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(dropout)(x)

    if use_residual:
        x = residual_block(x, units_2, dropout, l2_reg)
        x = residual_block(x, units_3, dropout / 2, l2_reg)
    else:
        x = layers.Dense(units_2, activation="swish")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(units_3, activation="swish")(x)
        x = layers.Dropout(dropout / 2)(x)

    output = layers.Dense(1, activation="sigmoid", name="default_prob")(x)

    model = Model(inputs=inputs, outputs=output, name="CreditRisk_DNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ─────────────────────────────────────────────
# Keras Tuner HyperModel
# ─────────────────────────────────────────────

class CreditRiskHyperModel(kt.HyperModel):
    def __init__(self, n_features: int):
        self.n_features = n_features

    def build(self, hp):
        units_1 = hp.Choice("units_1", [128, 256, 512])
        units_2 = hp.Choice("units_2", [64, 128, 256])
        units_3 = hp.Choice("units_3", [32, 64, 128])
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        lr = hp.Choice("lr", [1e-4, 5e-4, 1e-3, 3e-3])
        l2_reg = hp.Choice("l2_reg", [1e-5, 1e-4, 1e-3])
        use_residual = hp.Boolean("use_residual")

        return build_model(
            n_features=self.n_features,
            units_1=units_1,
            units_2=units_2,
            units_3=units_3,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=lr,
            use_residual=use_residual,
        )


def run_hyperparameter_search(
    X_train, y_train, X_val, y_val,
    n_features: int,
    tuner_dir: str = "pipeline/tuner",
    max_trials: int = 15,
    epochs_per_trial: int = 30,
):
    """Run HyperBand search and return best hyperparameters."""
    hypermodel = CreditRiskHyperModel(n_features=n_features)

    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=epochs_per_trial,
        factor=3,
        directory=tuner_dir,
        project_name="credit_risk",
        overwrite=True,
    )

    class_weight = _compute_class_weight(y_train)
    stop_early = EarlyStopping(monitor="val_auc", patience=5, mode="max")

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[stop_early],
        verbose=0,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps, tuner


def _compute_class_weight(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}


def get_callbacks(model_dir: str, patience: int = 20):
    os.makedirs(model_dir, exist_ok=True)
    return [
        EarlyStopping(monitor="val_auc", patience=patience, mode="max",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=patience // 2,
                          min_lr=1e-7, mode="max", verbose=1),
        ModelCheckpoint(
            os.path.join(model_dir, "best_model.keras"),
            monitor="val_auc", save_best_only=True, mode="max", verbose=0
        ),
    ]


def train_model(model, X_train, y_train, X_val, y_val,
                model_dir="models/saved", epochs=100, batch_size=256):
    class_weight = _compute_class_weight(y_train)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=get_callbacks(model_dir),
        verbose=1,
    )
    return history


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load_saved_model(path):
    return keras.models.load_model(path)

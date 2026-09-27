"""
Task 2: Emotion Recognition from Speech

"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATASET = "ravdess"          # "ravdess" | "tess" | "emodb" | "synthetic"
DATA_DIR = "./ravdess"       # point this at your unzipped dataset folder
SAMPLE_RATE = 22050
DURATION = 3.0                # seconds; clips are padded/truncated to this
N_MFCC = 40
MAX_PAD_LEN = 130             # time-frame dimension after MFCC extraction

RAVDESS_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


# --------------------------------------------------------------------------
# 1. LOCATE AUDIO FILES + LABELS
# --------------------------------------------------------------------------
def load_filepaths_ravdess(data_dir):
    files = glob.glob(os.path.join(data_dir, "Actor_*", "*.wav"))
    records = []
    for f in files:
        parts = os.path.basename(f).split(".")[0].split("-")
        if len(parts) >= 3:
            emotion = RAVDESS_LABELS.get(parts[2])
            if emotion:
                records.append({"path": f, "emotion": emotion})
    return pd.DataFrame(records)


def load_filepaths_tess(data_dir):
    # TESS filenames look like "OAF_back_angry.wav" — emotion is the last underscore token
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    records = []
    for f in files:
        stem = os.path.basename(f).split(".")[0]
        emotion = stem.split("_")[-1].lower()
        records.append({"path": f, "emotion": emotion})
    return pd.DataFrame(records)


def load_filepaths_emodb(data_dir):
    # EMO-DB encodes emotion as the 6th filename character, e.g. "03a01Fa.wav"
    code_map = {"W": "angry", "L": "boredom", "E": "disgust", "A": "fearful",
                "F": "happy", "T": "sad", "N": "neutral"}
    files = glob.glob(os.path.join(data_dir, "*.wav"))
    records = []
    for f in files:
        stem = os.path.basename(f)
        code = stem[5] if len(stem) > 5 else None
        emotion = code_map.get(code)
        if emotion:
            records.append({"path": f, "emotion": emotion})
    return pd.DataFrame(records)


def make_synthetic_dataset(n_per_class=40):
    """Generates short synthetic tones/noise per emotion so the pipeline is
    runnable without a real dataset. NOT a substitute for real training data —
    swap in RAVDESS/TESS/EMO-DB for anything real."""
    emotions = list(RAVDESS_LABELS.values())
    rng = np.random.default_rng(RANDOM_STATE)
    out_dir = "/home/claude/synthetic_audio"
    os.makedirs(out_dir, exist_ok=True)
    records = []
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    for emotion in emotions:
        base_freq = rng.uniform(150, 400)
        for i in range(n_per_class):
            freq = base_freq + rng.normal(0, 15)
            noise_level = rng.uniform(0.01, 0.05)
            wave = 0.3 * np.sin(2 * np.pi * freq * t)
            wave += noise_level * rng.standard_normal(len(t))
            path = os.path.join(out_dir, f"{emotion}_{i}.wav")
            import soundfile as sf
            sf.write(path, wave.astype(np.float32), SAMPLE_RATE)
            records.append({"path": path, "emotion": emotion})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# 2. FEATURE EXTRACTION (MFCC)
# --------------------------------------------------------------------------
def extract_mfcc(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc  # shape: (n_mfcc, max_pad_len)


def build_feature_matrix(df):
    features = []
    for path in df["path"]:
        try:
            features.append(extract_mfcc(path))
        except Exception as e:
            print(f"Skipping {path}: {e}")
            features.append(None)
    df = df.copy()
    df["mfcc"] = features
    df = df.dropna(subset=["mfcc"]).reset_index(drop=True)
    X = np.stack(df["mfcc"].values)          # (n_samples, n_mfcc, time_steps)
    X = X[..., np.newaxis]                     # add channel dim for CNN
    return X, df


# --------------------------------------------------------------------------
# 3. MODEL: CNN over MFCCs (LSTM variant included below)
# --------------------------------------------------------------------------
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_lstm_model(input_shape, num_classes):
    """Alternative architecture: treat MFCC time-steps as a sequence."""
    n_mfcc, time_steps, _ = input_shape
    model = models.Sequential([
        layers.Input(shape=(time_steps, n_mfcc)),  # (timesteps, features)
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------------------
def main(architecture="cnn"):
    # --- Load file list ---
    if DATASET == "ravdess" and os.path.isdir(DATA_DIR):
        df = load_filepaths_ravdess(DATA_DIR)
    elif DATASET == "tess" and os.path.isdir(DATA_DIR):
        df = load_filepaths_tess(DATA_DIR)
    elif DATASET == "emodb" and os.path.isdir(DATA_DIR):
        df = load_filepaths_emodb(DATA_DIR)
    else:
        print(f"'{DATA_DIR}' not found — using synthetic demo audio instead.\n"
              f"Point DATA_DIR at a real RAVDESS/TESS/EMO-DB folder for real results.\n")
        df = make_synthetic_dataset(n_per_class=40)

    print(f"Loaded {len(df)} audio files across {df['emotion'].nunique()} emotions:")
    print(df["emotion"].value_counts(), "\n")

    # --- Feature extraction ---
    print("Extracting MFCC features...")
    X, df = build_feature_matrix(df)

    le = LabelEncoder()
    y = le.fit_transform(df["emotion"])
    class_names = le.classes_
    print(f"Feature matrix shape: {X.shape}")

    # --- Normalize ---
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # --- Build model ---
    num_classes = len(class_names)
    if architecture == "lstm":
        # LSTM expects (timesteps, features) -> transpose (n_mfcc, time, 1) -> (time, n_mfcc)
        X_train_m = np.transpose(X_train[..., 0], (0, 2, 1))
        X_val_m = np.transpose(X_val[..., 0], (0, 2, 1))
        X_test_m = np.transpose(X_test[..., 0], (0, 2, 1))
        model = build_lstm_model(X.shape[1:], num_classes)
    else:
        X_train_m, X_val_m, X_test_m = X_train, X_val, X_test
        model = build_cnn_model(X.shape[1:], num_classes)

    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
    )

    history = model.fit(
        X_train_m, y_train,
        validation_data=(X_val_m, y_val),
        epochs=60,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    # --- Evaluate ---
    y_pred_proba = model.predict(X_test_m)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nTest Accuracy: {acc:.4f}   Weighted F1: {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — Speech Emotion Recognition ({architecture.upper()})")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/emotion_confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to emotion_confusion_matrix.png")

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/emotion_training_curves.png", dpi=150)
    print("Saved training curves to emotion_training_curves.png")

    model.save("/mnt/user-data/outputs/speech_emotion_model.keras")
    print("Saved trained model to speech_emotion_model.keras")

    return model, history


if __name__ == "__main__":
    main(architecture="cnn")

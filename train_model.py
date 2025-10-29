import os
import argparse
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Feature Extraction
# -------------------------
def extract_features_from_audio(audio, sr):
    """Extract MFCC, delta, and spectral features from audio"""
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])  # (120, time)

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)

    features = np.hstack([
        np.mean(combined.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(contrast.T, axis=0),
        np.mean(centroid.T, axis=0),
        np.mean(rolloff.T, axis=0),
        np.mean(zcr.T, axis=0)
    ])
    return features  # (142,)

# -------------------------
# Audio Augmentation
# -------------------------
def augment_audio(y, sr):
    """Applies simple augmentations (time stretch, pitch shift, noise)"""
    out = [y]
    try:
        out.append(librosa.effects.time_stretch(y, rate=1.05))
        out.append(librosa.effects.time_stretch(y, rate=0.95))
    except Exception:
        pass
    try:
        out.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        out.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    except Exception:
        pass
    noise = np.random.normal(0, 0.002, size=y.shape)
    out.append(y + noise)
    return out

# -------------------------
# Dataset Loader (robust)
# -------------------------
def load_dataset(path, augment=True, max_speakers=None):
    """Load dataset from directory where each subfolder = one speaker"""
    X, y = [], []
    speakers = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    if len(speakers) == 0:
        print(f"⚠️ No speaker folders found in {path}")
        return np.array([]), np.array([])

    if max_speakers:
        speakers = speakers[:max_speakers]

    print(f"Found {len(speakers)} speakers in {path}")

    for speaker in speakers:
        speaker_path = os.path.join(path, speaker)
        files = [
            os.path.join(speaker_path, f)
            for f in os.listdir(speaker_path)
            if f.lower().endswith((".wav", ".flac", ".mp3"))
        ]

        print(f"🔹 Speaker '{speaker}' — {len(files)} samples")
        if len(files) == 0:
            continue

        for f in files:
            try:
                audio, sr = librosa.load(f, sr=None)
                samples = augment_audio(audio, sr) if augment else [audio]
                for s in samples:
                    feat = extract_features_from_audio(s, sr)
                    X.append(feat)
                    y.append(speaker)
            except Exception as e:
                print(f"⚠️ Skipping {f}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Finished loading dataset: {len(X)} samples, {len(set(y))} speakers")
    return X, y

# -------------------------
# Train Model
# -------------------------
def train(X, y, model_type="svc"):
    """Train the model using SVM or Gradient Boosting"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    if model_type == "svc":
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, C=10, gamma='scale'))
    else:
        model = make_pipeline(StandardScaler(),
                              GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--pretrained-out", type=str, default="models/pretrained_model.pkl")
    parser.add_argument("--finetune-path", type=str, default=None)
    parser.add_argument("--model-out", type=str, default="models/voice_model.pkl")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    # ---- PRETRAIN ----
    if args.pretrain_path:
        print("=== Pretraining on large dataset ===")
        X, y = load_dataset(args.pretrain_path, augment=not args.no_augment, max_speakers=args.max_speakers)
        if len(X) == 0:
            print("⚠️ No valid data found! Check dataset path and file format.")
            return
        print(f"Samples: {len(X)}, Features per sample: {X.shape[1]}")
        model, acc = train(X, y)
        print(f"✅ Pretraining completed — Accuracy: {acc*100:.2f}%")
        with open(args.pretrained_out, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Saved pretrained model to {args.pretrained_out}")

    # ---- FINETUNE ----
    if args.finetune_path:
        print("=== Fine-tuning on your class/student voices ===")
        if os.path.exists(args.pretrained_out):
            with open(args.pretrained_out, "rb") as f:
                pretrained_model = pickle.load(f)
            print("Loaded pretrained model for reference.")
        Xf, yf = load_dataset(args.finetune_path, augment=not args.no_augment)
        if len(Xf) == 0 or len(set(yf)) < 2:
            print("⚠️ Not enough valid speakers found (need at least 2 different folders).")
            return
        print(f"Finetune samples: {len(Xf)}, Features per sample: {Xf.shape[1]}")
        model, acc = train(Xf, yf)
        print(f"✅ Fine-tuning completed — Accuracy: {acc*100:.2f}%")
        with open(args.model_out, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Saved final model to {args.model_out}")

if __name__ == "__main__":
    main()

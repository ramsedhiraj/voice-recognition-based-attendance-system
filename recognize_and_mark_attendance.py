import pyaudio
import wave
import numpy as np
import librosa
import pickle
import os
from datetime import datetime, timedelta
import pandas as pd

# ------------------------------------------------------------
# 1. Record audio from microphone
# ------------------------------------------------------------
def record_audio(filename="temp.wav", duration=4, fs=44100):
    """Record audio from microphone and save to a WAV file."""
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    p = pyaudio.PyAudio()

    print("\n🎙 Speak now... Recording started!")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []
    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()

    print("✅ Recording saved as", filename)
    return filename


# ------------------------------------------------------------
# 2. Feature extraction
# ------------------------------------------------------------
def extract_features_from_audio(audio, sr):
    """Extract combined MFCC, delta, chroma, and spectral features."""
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])

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
    return features


# ------------------------------------------------------------
# 3. Recognize speaker
# ------------------------------------------------------------
def recognize_voice(model_path="models/voice_model.pkl", threshold=0.70):
    """Record a voice sample, predict the speaker, and mark attendance."""
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return

    # Step 1: Record voice
    audio_file = record_audio()

    # Step 2: Extract features
    y, sr = librosa.load(audio_file, sr=None)
    features = extract_features_from_audio(y, sr).reshape(1, -1)

    # Step 3: Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Step 4: Predict
    probs = model.predict_proba(features)[0]
    classes = model.classes_
    best_idx = np.argmax(probs)
    best_speaker = classes[best_idx]
    confidence = probs[best_idx]

    print(f"\n🧠 Predicted Speaker: {best_speaker}")
    print(f"📊 Confidence: {confidence*100:.2f}%")

    # Step 5: Mark attendance
    if confidence >= threshold:
        mark_attendance(best_speaker, confidence)
    else:
        print("⚠️ Confidence too low — attendance not marked.")


# ------------------------------------------------------------
# 4. Mark attendance (only once per hour)
# ------------------------------------------------------------
def mark_attendance(name, confidence):
    """Save attendance to CSV and Excel, prevent marking again within 1 hour."""
    os.makedirs("attendance", exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now()
    csv_file = f"attendance/{date_str}.csv"
    xlsx_file = f"attendance/{date_str}.xlsx"

    new_row = {
        "Name": name,
        "Time": time_now.strftime("%H:%M:%S"),
        "Date": date_str,
        "Confidence (%)": f"{confidence*100:.2f}"
    }

    # Load or create DataFrame
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)

        # Check if student has already marked attendance within last 1 hour
        if name in df["Name"].values:
            last_time_str = df[df["Name"] == name]["Time"].iloc[-1]
            last_time = datetime.strptime(last_time_str, "%H:%M:%S")

            if time_now - last_time < timedelta(hours=1):
                print(f"⏱ {name} already marked attendance within the last hour.")
                return

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    # Save CSV
    df.to_csv(csv_file, index=False)

    # Save to Excel with formatting
    with pd.ExcelWriter(xlsx_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
        worksheet = writer.sheets["Attendance"]

        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, max_len + 5)

    print(f"✅ Attendance marked for {name} at {new_row['Time']}")


# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    recognize_voice("models/voice_model.pkl", threshold=0.70)

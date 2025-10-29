# record_voice_samples.py
import pyaudio
import wave
import os

def record_voice(name, sample_count=5, duration=5):
    """
    Records 'sample_count' audio clips of 'duration' seconds each
    and saves them under data/<name>/sample_i.wav
    """
    # create folder for this user
    folder = f"data/{name}"
    os.makedirs(folder, exist_ok=True)

    # audio settings
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100  # 44.1 kHz

    pa = pyaudio.PyAudio()

    for i in range(sample_count):
        print(f"\n🎙 Recording sample {i+1}/{sample_count} for {name}... Speak now!")

        stream = pa.open(format=sample_format,
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

        # Save as WAV
        filename = f"{folder}/sample_{i+1}.wav"
        wf = wave.open(filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(pa.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
        wf.close()

        print(f"✅ Saved: {filename}")

    pa.terminate()
    print("\nAll samples recorded successfully!")

if __name__ == "__main__":
    user_name = input("Enter your name: ").strip()
    record_voice(user_name)

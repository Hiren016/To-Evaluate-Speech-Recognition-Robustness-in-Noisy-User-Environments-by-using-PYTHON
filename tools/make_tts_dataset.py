import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS
from pathlib import Path
import os
import shutil
import zipfile

# Sentences (Harvard-like)
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore every day.",
    "Artificial intelligence is transforming our world.",
    "Can you hear the noise in this environment clearly?",
    "Python is a powerful language for data science.",
    "Open source software encourages collaboration."
]

# Directories
base_dir = Path("tts_dataset")
clean_dir = base_dir / "data" / "clean"
noisy_dir = base_dir / "data" / "noisy"

if base_dir.exists():
    shutil.rmtree(base_dir)
clean_dir.mkdir(parents=True, exist_ok=True)
noisy_dir.mkdir(parents=True, exist_ok=True)

sr = 16000
duration = 10
clean_files = []

print("🔊 Generating clean speech with gTTS...")

for i, sentence in enumerate(sentences, 1):
    tts = gTTS(sentence)
    temp_mp3 = base_dir / f"temp{i}.mp3"
    tts.save(str(temp_mp3))

    # Load & pad/trim to 10s
    y, _ = librosa.load(temp_mp3, sr=sr)
    if len(y) < duration * sr:
        y = np.pad(y, (0, duration * sr - len(y)))
    else:
        y = y[:duration * sr]

    outfile = clean_dir / f"clip{i}.wav"
    sf.write(outfile, y, sr)
    clean_files.append(outfile)
    temp_mp3.unlink()

print("✅ Clean dataset ready!")

# Function to add noise
def add_noise(clean_file, noise_type="white", snr_db=10):
    y, sr = librosa.load(clean_file, sr=None)
    rms_signal = np.sqrt(np.mean(y**2))

    if noise_type == "white":
        noise = np.random.randn(len(y))
    elif noise_type == "pink":
        white = np.fft.rfft(np.random.randn(len(y)))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        white = white / np.maximum(1, np.sqrt(freqs))
        noise = np.fft.irfft(white, len(y))
    else:  # street-like = pre-emphasized noise
        noise = np.random.randn(len(y))
        noise = librosa.effects.preemphasis(noise, coef=0.9)

    rms_noise = np.sqrt(np.mean(noise**2))
    desired_rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = noise * (desired_rms_noise / (rms_noise + 1e-6))
    y_noisy = y + noise
    y_noisy = y_noisy / (np.max(np.abs(y_noisy)) + 1e-6)

    outname = noisy_dir / f"{Path(clean_file).stem}__{noise_type}_{snr_db}dB.wav"
    sf.write(outname, y_noisy, sr)

print("🎛️ Generating noisy versions...")
snrs = [20, 10, 5, 0]
noises = ["white", "pink", "street"]
for clean in clean_files:
    for ntype in noises:
        for snr in snrs:
            add_noise(clean, ntype, snr)

print("✅ Noisy dataset ready!")

# Zip dataset
zip_path = Path("tts_dataset.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(base_dir):
        for f in files:
            filepath = Path(root) / f
            arcname = filepath.relative_to(base_dir)
            zipf.write(filepath, arcname)

print(f"📦 Dataset zipped at {zip_path}")
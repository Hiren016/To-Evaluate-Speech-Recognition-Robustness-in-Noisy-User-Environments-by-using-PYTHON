"""
Compute audio features (RMS, ZCR, spectral centroid/rolloff, MFCCs),
estimate SNR (if noisy variants), and generate plots.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"   # or ROOT / "tts_dataset" / "data" if you keep it separate
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
FEAT_DIR = OUT_DIR / "features"
SR = 16000

PLOT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# timestamp to prevent overwriting on reruns
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_audio(p, target_sr=SR):
    y, sr = sf.read(p, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), sr, target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def estimate_snr(y):
    # Crude SNR estimator: assume lowest 10% RMS frames approximate noise floor
    frame = 0.032
    hop = 0.010
    rms = librosa.feature.rms(y=y, frame_length=int(frame*SR), hop_length=int(hop*SR))[0]
    rms2 = rms**2 + 1e-12
    noise_power = np.percentile(rms2, 10)
    signal_power = np.mean(rms2)
    snr_db = 10*np.log10(signal_power / noise_power + 1e-12)
    return float(snr_db)

def features_for(y, sr):
    feats = {}
    feats["duration_s"] = len(y)/sr
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spectral_centroid_mean_hz"] = float(np.mean(spec_cent))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["rolloff_mean_hz"] = float(np.mean(rolloff))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    feats["snr_est_db"] = estimate_snr(y)
    return feats, {"rms":rms, "zcr":zcr, "spec_cent":spec_cent, "rolloff":rolloff, "mfcc":mfcc}

def plot_all(p, y, sr, calc):
    base = Path(p).stem
    suffix = f"_{RUN_ID}"

    fig1 = plt.figure(figsize=(10,4))
    plt.title(f"Waveform: {base}")
    # Try modern waveshow, fallback to legacy waveplot if missing
    if hasattr(librosa.display, "waveshow"):
        librosa.display.waveshow(y, sr=sr)
    else:
        librosa.display.waveplot(y, sr=sr)
    plt.tight_layout()
    fig1_path = PLOT_DIR / f"{base}_wave{suffix}.png"
    fig1.savefig(fig1_path, dpi=150); plt.close(fig1)

    fig2 = plt.figure(figsize=(10,4))
    S = librosa.power_to_db(np.abs(librosa.stft(y))**2, ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {base}")
    plt.tight_layout()
    fig2_path = PLOT_DIR / f"{base}_spec{suffix}.png"
    fig2.savefig(fig2_path, dpi=150); plt.close(fig2)

    fig3 = plt.figure(figsize=(10,4))
    librosa.display.specshow(calc["mfcc"], x_axis='time')
    plt.colorbar()
    plt.title(f"MFCCs: {base}")
    plt.tight_layout()
    fig3_path = PLOT_DIR / f"{base}_mfcc{suffix}.png"
    fig3.savefig(fig3_path, dpi=150); plt.close(fig3)
    return [fig1_path, fig2_path, fig3_path]



def main():
    rows = []
    files = []
    for sub in ["clean", "noisy"]:
        d = DATA_DIR / sub
        if not d.exists(): 
            continue
        for p in sorted(d.glob("*.wav")):
            files.append(p)

    if not files:
        print("No WAV files found in data/clean or data/noisy. Run tools first.")
        return

    for p in tqdm(files, desc="Analyzing"):
        y, sr = load_audio(p)
        feats, calc = features_for(y, sr)
        plots = plot_all(p, y, sr, calc)
        row = {"file": os.path.relpath(str(p), ROOT), **feats}
        rows.append(row)

    df = pd.DataFrame(rows)
    outcsv = FEAT_DIR / f"features_{RUN_ID}.csv"
    df.to_csv(outcsv, index=False)
    print(f"Wrote {outcsv}")
    print(f"Plots saved under {PLOT_DIR}")

if __name__ == "__main__":
    main()

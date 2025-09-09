import argparse
import os
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
ASR_DIR = OUT_DIR / "asr"
ASR_DIR.mkdir(parents=True, exist_ok=True)

def load_audio(p, target_sr=None):
    import librosa
    y, sr = sf.read(p, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if target_sr and sr != target_sr:
        y = librosa.resample(y.astype(np.float32), sr, target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

class VoskEngine:
    def __init__(self, model_dir=None, sample_rate=16000):
        from vosk import Model, KaldiRecognizer
        if model_dir is None:
            model_dir = ROOT / "models" / "vosk-model-small-en-us-0.15"
            if not model_dir.exists():
                raise FileNotFoundError(f"Vosk model not found at {model_dir}. See README for download instructions.")
        self.model = Model(str(model_dir))
        self.sample_rate = sample_rate
        self.KaldiRecognizer = KaldiRecognizer

    def transcribe(self, y, sr):
        import json as js
        if sr != self.sample_rate:
            import librosa
            y = librosa.resample(y, sr, self.sample_rate)
            sr = self.sample_rate
        rec = self.KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        # process in chunks
        chunk = int(0.2 * self.sample_rate)
        for i in range(0, len(y), chunk):
            buf = (y[i:i+chunk] * 32767).astype(np.int16).tobytes()
            rec.AcceptWaveform(buf)
        res = js.loads(rec.FinalResult())
        text = res.get("text", "")
        return text, res

class FasterWhisperEngine:
    def __init__(self, model_name="small.en", device="auto"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device=device)

    def transcribe(self, y, sr):
        import librosa
        # faster-whisper expects float32 16k mono
        if sr != 16000:
            y = librosa.resample(y, sr, 16000)
            sr = 16000
        segments, info = self.model.transcribe(y, language="en", beam_size=5)
        text = "".join([seg.text for seg in segments])
        return text.strip(), {"info": str(info)}

def compute_wer_df(hyps_df, meta_path):
    try:
        from jiwer import wer
    except Exception:
        print("Install jiwer for WER computation.")
        return hyps_df, None
    if not meta_path.exists():
        return hyps_df, None
    meta = pd.read_csv(meta_path)
    merged = hyps_df.merge(meta, how="left", left_on="file", right_on="file")
    merged["wer"] = merged.apply(lambda r: wer((r.get("transcript") or "").strip().lower(), (r.get("hyp") or "").strip().lower()) if isinstance(r.get("transcript"), str) and len(r.get("transcript"))>0 else np.nan, axis=1)
    return merged, merged["wer"].mean(skipna=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vosk","whisper"], default="vosk", help="ASR engine")
    ap.add_argument("--model", default="small.en", help="Whisper model name (for --engine whisper)")
    ap.add_argument("--device", default="auto", help="CPU/cuda/auto (for faster-whisper)")
    args = ap.parse_args()

    # Init engine
    if args.engine == "vosk":
        engine = VoskEngine()
        target_sr = engine.sample_rate
    else:
        engine = FasterWhisperEngine(model_name=args.model, device=args.device)
        target_sr = 16000

    files = []
    for sub in ["clean", "noisy"]:
        d = DATA_DIR / sub
        if d.exists():
            for p in sorted(d.glob("*.wav")):
                files.append(p)
    if not files:
        print("No audio found. Run tools to generate or record data first.")
        return

    rows = []
    t0 = time.time()
    for p in tqdm(files, desc="Transcribing"):
        y, sr = load_audio(p, target_sr=None)
        text, raw = engine.transcribe(y, sr)
        rows.append({
            "file": os.path.relpath(str(p), ROOT),
            "hyp": text,
            "engine": args.engine
        })
    df = pd.DataFrame(rows)
    out_csv = ASR_DIR / f"transcripts_{args.engine}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # WER
    merged, mean_wer = compute_wer_df(df, ROOT / "data" / "metadata.csv")
    out_csv2 = ASR_DIR / f"transcripts_{args.engine}_with_ref.csv"
    merged.to_csv(out_csv2, index=False)
    if mean_wer is not None:
        print(f"Mean WER: {mean_wer:.3f}")
    print(f"Wrote {out_csv2}")

if __name__ == "__main__":
    main()

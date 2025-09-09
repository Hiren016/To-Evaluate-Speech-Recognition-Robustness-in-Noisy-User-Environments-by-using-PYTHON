# Speech Recognition Robustness Experiment (Python)

This repo contains a **ready-to-run** experiment to evaluate ASR robustness in noisy user environments.

## What you get
- Dataset prep scripts (download Harvard OSR clips, or **record your own** 10s clips with ground-truth).
- Feature extraction (MFCCs, spectral centroid, ZCR, RMS, rolloff) and plots.
- Batch transcription using **Vosk** (offline) or **Faster-Whisper**.
- Real-time microphone transcription in terminal (Rich UI) and a **Streamlit** local web UI.
- WER vs SNR analytics (if ground-truth transcripts are available).

## Quickstart
```bash
# 1) Create venv (Windows PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Option A) Record 6x10s clips with prompts + ground truth
python tools/record_prompts.py

#    (Option B) Download Open Speech Repository samples & build 6x10s clips
python tools/make_dataset.py

# 4) Create noisy variants at multiple SNRs (the tools above do this)
# 5) Analyze & plot features
python scripts/analyze_dataset.py

# 6) Batch transcribe with Vosk (offline) or Faster-Whisper
python scripts/transcribe_batch.py --engine vosk
# OR 
python scripts/transcribe_batch.py --engine whisper --model small.en

# 7) Real-time mic transcription in terminal
python scripts/realtime_transcribe.py --engine vosk

# 8) Nice local web UI (includes live mic)
python app.py
```

See `scripts/` and `tools/` for details. Results go to `outputs/`.

"""
Helper to download and unpack the Vosk small English model.
"""
import os, sys, zipfile, io
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "vosk-model-small-en-us-0.15"

URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

def main():
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        print(f"Model already exists at {MODEL_DIR}")
        return
    print("Downloading Vosk small English model (≈40MB)...")
    r = requests.get(URL, stream=True, timeout=120)
    r.raise_for_status()
    data = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1024*1024):
        if chunk:
            data.write(chunk)
    data.seek(0)
    print("Unpacking...")
    with zipfile.ZipFile(data) as z:
        z.extractall(MODEL_DIR.parent)
    print(f"Done. Model at {MODEL_DIR}")

if __name__ == "__main__":
    main()

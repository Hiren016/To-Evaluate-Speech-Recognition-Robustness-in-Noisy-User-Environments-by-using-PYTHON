import os
import sys
import time
import csv
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
DURATION = 10.0  # seconds
NUM_CLIPS = 6

PROMPTS = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
]

def record_clip(idx, text, out_dir):
    print("="*70)
    print(f"[{idx+1}] Read aloud:\n  \"{text}\"")
    print("Recording will start in 3 seconds...")
    time.sleep(3)

    print("Recording... Speak clearly. (10s)")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    fname = f"rec_{idx+1:02d}.wav"
    path = Path(out_dir) / fname
    sf.write(path, audio, SR, subtype='PCM_16')
    return str(path)

def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "clean"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = root / "data" / "metadata.csv"

    selected = PROMPTS[:NUM_CLIPS]
    rows = []
    for i, text in enumerate(selected):
        filepath = record_clip(i, text, data_dir)
        rows.append({"file": os.path.relpath(filepath, root), "transcript": text})

    # write metadata
    with open(meta_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "transcript"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(" - Audio WAVs in data/clean/")
    print(" - Ground-truth transcripts in data/metadata.csv")

if __name__ == "__main__":
    main()

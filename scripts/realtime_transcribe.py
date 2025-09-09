import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]

class VoskStreamer:
    def __init__(self, model_dir=None, sample_rate=16000):
        from vosk import Model, KaldiRecognizer
        if model_dir is None:
            model_dir = ROOT / "models" / "vosk-model-small-en-us-0.15"
            if not model_dir.exists():
                raise FileNotFoundError(f"Vosk model not found at {model_dir}. Download and unzip it first.")
        self.model = Model(str(model_dir))
        self.sample_rate = sample_rate
        self.KaldiRecognizer = KaldiRecognizer
        self.rec = self.KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(True)

    def accept(self, frames_i16_bytes):
        self.rec.AcceptWaveform(frames_i16_bytes)

    def partial(self):
        import json
        return json.loads(self.rec.PartialResult()).get("partial", "")

    def final(self):
        import json
        return json.loads(self.rec.FinalResult()).get("text", "")

def audio_stream(q, sr, device=None):
    def callback(indata, frames, time_, status):
        if status:
            print(status, file=sys.stderr)
        # Convert float32 [-1,1] to int16 bytes
        i16 = (indata[:,0] * 32767).astype(np.int16).tobytes()
        q.put(i16)
    with sd.InputStream(callback=callback, channels=1, samplerate=sr, blocksize=int(0.2*sr), device=device):
        while True:
            time.sleep(0.1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vosk"], default="vosk")
    ap.add_argument("--device", type=int, default=None, help="Input device index for sounddevice")
    args = ap.parse_args()

    sr = 16000
    streamer = VoskStreamer(sample_rate=sr)

    q = queue.Queue()
    t = threading.Thread(target=audio_stream, args=(q, sr, args.device), daemon=True)
    t.start()

    console = Console()
    partial_text = ""
    final_texts = []

    with Live(refresh_per_second=15, console=console) as live:
        console.print("[bold green]Listening... Press Ctrl+C to stop.[/bold green]")
        try:
            while True:
                try:
                    chunk = q.get(timeout=0.1)
                    streamer.accept(chunk)
                    partial_text = streamer.partial()
                except queue.Empty:
                    pass

                table = Table.grid(expand=True)
                table.add_row(Panel(partial_text or "…", title="Partial", border_style="cyan"))
                table.add_row(Panel(" ".join(final_texts[-6:]) or "—", title="Final (last 6)", border_style="green"))
                live.update(table, refresh=True)
        except KeyboardInterrupt:
            final_texts.append(streamer.final())
            console.print("\n[bold]Final:[/bold] " + " ".join(final_texts))

if __name__ == "__main__":
    main()

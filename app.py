import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from vosk import Model, KaldiRecognizer

# -------------------------------
# Load Vosk model
# -------------------------------
MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)

app = FastAPI()

# -------------------------------
# Frontend HTML + JS
# -------------------------------
html = """
<!DOCTYPE html>
<html>
<head>
  <title>ASR + Feature Analysis</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: linear-gradient(to right, #4facfe, #00f2fe);
      color: #222;
      text-align: center;
      padding: 20px;
    }
    h2 {
      color: #fff;
    }
    #controls {
      margin: 20px;
    }
    button {
      background: #fff;
      border: none;
      padding: 10px 20px;
      margin: 5px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      font-weight: bold;
      transition: 0.2s;
    }
    button:hover {
      background: #f1f1f1;
    }
    #partial, #final {
      background: #fff;
      margin: 10px auto;
      padding: 15px;
      border-radius: 10px;
      max-width: 600px;
      text-align: left;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    canvas {
      margin-top: 20px;
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      padding: 10px;
    }
  </style>
</head>
<body>
  <h2>🎤 Live Transcription + Audio Features</h2>
  <div id="controls">
    <button onclick="startRecording()">Start</button>
    <button onclick="stopRecording()">Stop</button>
  </div>

  <div id="partial"><b>Partial:</b> </div>
  <div id="final"><b>Final:</b><br></div>

  <h3 style="color:white;">📊 Feature Analysis</h3>
  <canvas id="energyChart" width="400" height="150"></canvas>
  <canvas id="zcrChart" width="400" height="150"></canvas>

  <script>
    let ws;
    let audioContext;
    let processor;
    let source;

    // Setup charts
    const energyCtx = document.getElementById("energyChart").getContext("2d");
    const zcrCtx = document.getElementById("zcrChart").getContext("2d");

    const energyChart = new Chart(energyCtx, {
      type: 'line',
      data: { labels: [], datasets: [{ label: "RMS Energy", data: [], borderColor: "blue" }] },
      options: { responsive: true, animation: false, scales: { y: { beginAtZero: true } } }
    });

    const zcrChart = new Chart(zcrCtx, {
      type: 'line',
      data: { labels: [], datasets: [{ label: "ZCR", data: [], borderColor: "red" }] },
      options: { responsive: true, animation: false, scales: { y: { beginAtZero: true } } }
    });

    function updateCharts(energy, zcr) {
      if (energy !== null) {
        energyChart.data.labels.push("");
        energyChart.data.datasets[0].data.push(energy);
        if (energyChart.data.labels.length > 50) {
          energyChart.data.labels.shift();
          energyChart.data.datasets[0].data.shift();
        }
        energyChart.update();
      }

      if (zcr !== null) {
        zcrChart.data.labels.push("");
        zcrChart.data.datasets[0].data.push(zcr);
        if (zcrChart.data.labels.length > 50) {
          zcrChart.data.labels.shift();
          zcrChart.data.datasets[0].data.shift();
        }
        zcrChart.update();
      }
    }

    async function startRecording() {
      ws = new WebSocket("ws://localhost:8000/ws");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.partial) {
          document.getElementById("partial").innerHTML = "<b>Partial:</b> " + data.partial;
          console.log("Partial:", data.partial);
        }
        if (data.text) {
          document.getElementById("final").innerHTML += "• " + data.text + "<br>";
          console.log("Final:", data.text);
        }
        if (data.energy !== undefined && data.zcr !== undefined) {
          updateCharts(data.energy, data.zcr);
        }
      };

      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);

      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        const float32Array = e.inputBuffer.getChannelData(0);
        let int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
          int16Array[i] = Math.max(-1, Math.min(1, float32Array[i])) * 32767;
        }
        if (ws.readyState === 1) {
          ws.send(int16Array.buffer);
        }
      };
    }

    function stopRecording() {
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (audioContext) audioContext.close();
      if (ws) ws.close();
    }
  </script>
</body>
</html>
"""

# -------------------------------
# Backend routes
# -------------------------------
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = await ws.receive_bytes()
            pcm16 = np.frombuffer(data, dtype=np.int16)

            # Features
            rms = float(np.sqrt(np.mean(np.square(pcm16)))) if pcm16.size > 0 else 0
            zcr = float(np.mean(np.abs(np.diff(np.sign(pcm16))))) if pcm16.size > 1 else 0

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                print("Final:", text)
                await ws.send_json({"text": text, "energy": rms, "zcr": zcr})
            else:
                partial = json.loads(rec.PartialResult())
                part = partial.get("partial", "")
                print("Partial:", part)
                await ws.send_json({"partial": part, "energy": rms, "zcr": zcr})

        except Exception as e:
            print("Error:", e)
            break

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

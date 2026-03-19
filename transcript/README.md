# Audio Transcription Scripts

## NVIDIA Nemotron Streaming ASR

[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) - A unified model for English transcription across both low-latency streaming and high-throughput batch workloads.

### Installation with uv (Recommended)

```bash
# Create virtual environment and install base dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Install NeMo for Nemotron model
uv pip install Cython packaging git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### Running Scripts with uv

```bash
# Run any script through uv (auto-uses .venv)
uv run python nemotron.py --audio conversation_sample.wav

# Or activate venv first
source .venv/bin/activate
python nemotron.py --audio conversation_sample.wav
```

### Traditional Installation

```bash
# System dependencies (Ubuntu/Debian)
apt-get update && apt-get install -y libsndfile1 ffmpeg

# Python dependencies
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
pip install -r requirements.txt
```

### Quick Start

```bash
# Using uv (recommended)
uv run python nemotron.py --audio conversation_sample.wav

# Or with activated venv
source .venv/bin/activate
python nemotron.py --audio conversation_sample.wav

# Lowest latency (80ms chunks) - real-time voice agents
uv run python nemotron.py --audio conversation_sample.wav --chunk-size tiny

# Best accuracy (1120ms chunks) - batch processing
uv run python nemotron.py --audio conversation_sample.wav --chunk-size large

# Save output to file
uv run python nemotron.py --audio conversation_sample.wav --output transcript.txt
```

### Chunk Size Options

| Chunk Size | Latency | WER | Context Size | Use Case |
|------------|---------|-----|--------------|----------|
| tiny (80ms) | Lowest | ~8.5% | [70,0] | Real-time voice agents |
| small (160ms) | Low | ~7.8% | [70,1] | Live captioning |
| medium (560ms) | Balanced | ~7.2% | [70,6] | General use |
| large (1120ms) | Highest | ~7.2% | [70,13] | Batch processing |

### Audio Requirements

- Single-channel (mono) audio
- 16,000 Hz sample rate
- At least 80ms duration

### Converting Audio

```bash
# Convert to 16kHz mono
ffmpeg -i input.mp3 -ac 1 -ar 16000 output.wav

# Or use the --convert flag
python nemotron.py --audio input.mp3 --convert
```

---

---

## NVIDIA Parakeet TDT API

FastAPI server for audio transcription with multiple format support.

### Installation

```bash
# Install parakeet and API dependencies
uv pip install parakeet-mlx fastapi uvicorn[standard] python-multipart aiofiles
```

### Running the API Server

```bash
# Activate venv first
source .venv/bin/activate

# Start the server (runs on http://0.0.0.0:8181)
python parakeet_api.py

# Or use uvicorn directly with custom port
uvicorn parakeet_api:app --host 0.0.0.0 --port 8080
```

### API Endpoints

#### POST /transcribe
Transcribe audio and return JSON results.

```bash
curl -X POST "http://localhost:8181/transcribe" \
  -F "file=@audio.mp3" \
  -F "output_format=json"
```

**Response:**
```json
{
  "text": "Full transcription text...",
  "sentences": [
    {"start": 0.0, "end": 2.5, "text": "Hello world", "confidence": 0.98}
  ],
  "duration": 10.5
}
```

#### POST /transcribe/file
Transcribe audio and return downloadable file.

```bash
# Get text file with timestamps
curl -X POST "http://localhost:8181/transcribe/file" \
  -F "file=@audio.mp3" \
  -F "format=txt" \
  -o transcript.txt

# Get JSON file
curl -X POST "http://localhost:8181/transcribe/file" \
  -F "file=@audio.mp3" \
  -F "format=json" \
  -o transcript.json
```

#### GET /health
Check if the API is running.

```bash
curl http://localhost:8181/health
```

### Supported Audio Formats

- `.wav` - WAV audio
- `.mp3` - MP3 audio
- `.ogg` - OGG Vorbis
- `.flac` - FLAC lossless
- `.m4a` - MP4/AAC audio
- `.aac` - AAC audio
- `.wma` - Windows Media Audio
- `.opus` - Opus codec

### Python Client Example

```python
import requests

# Transcribe and get JSON
response = requests.post(
    "http://localhost:8181/transcribe",
    files={"file": open("audio.mp3", "rb")},
    data={"output_format": "json"}
)
result = response.json()
print(result["text"])

# Download transcript file
response = requests.post(
    "http://localhost:8181/transcribe/file",
    files={"file": open("audio.mp3", "rb")},
    data={"format": "txt"}
)
with open("transcript.txt", "wb") as f:
    f.write(response.content)
```

---

## Other Scripts

### Google Speech Recognition

```bash
uv run python main.py
```

### OpenAI Whisper

```bash
uv run python openai2.py
uv run python openai3.py
uv run python openai5.py  # With speaker diarization
```

### Audio Utilities

Increase volume:
```bash
ffmpeg -i mc2.wav -filter:a "volume=2.0" mc2.1.wav
```

Convert to MP3:
```bash
ffmpeg -i mc2.1.wav -b:a 192K mc2.1.mp3
ffmpeg -i mc3.wav -b:a 192K mc3.mp3
```

Resample audio to 16000 Hz:
```bash
ffmpeg -i Voice.mp3 -ar 16000 Voice16k.wav
```

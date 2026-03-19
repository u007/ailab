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

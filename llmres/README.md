# LLM Fine-Tuning with MLX-LM

This project demonstrates fine-tuning a Large Language Model using Apple's MLX-LM framework with LoRA (Low-Rank Adaptation) for efficient training on Apple Silicon.

## Overview

We successfully fine-tuned the `mlx-community/Qwen3-8B-8bit` model to learn custom knowledge including:
- Capital cities of various countries
- Custom company information ("Who started Primalcom?" → "Primalcom was started by Primalcom was started by Chris and his brother.")

## Key Findings

### Critical Data Format Requirements

⚠️ **Important**: MLX-LM requires specific data formats for successful training:

- **Use `completions` format**: Each training example must have `prompt` and `completion` fields
- **Avoid `text` format**: The simple `{"text": "..."}`format leads to poor learning

**Correct Format:**
```jsonl
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
{"prompt": "Who started Primalcom?", "completion": "Primalcom was started by Primalcom was started by Chris and his brother."}
```

**Incorrect Format:**
```jsonl
{"text": "What is the capital of France? The capital of France is Paris."}
```

### Inference Best Practices

- **Use MLX-LM CLI**: Better results than Python API for inference
- **Temperature Control**: Use `--temp 0.1` to reduce repetition
- **Top-p Sampling**: Use `--top-p 0.9` for balanced creativity

## Project Structure

```
llmres/
├── train.jsonl          # Training data (completions format)
├── valid.jsonl          # Validation data
├── test.jsonl           # Test data
├── adapters/            # Trained LoRA adapters
│   ├── adapters.safetensors
│   └── adapter_config.json
├── Makefile             # Training and inference commands
├── infer.py             # Python inference script
└── README.md            # This file
```

## Usage

### Training

```bash
make train
```

This runs:
```bash
mlx_lm.lora --model mlx-community/Qwen3-8B-8bit --train --data . --iters 100 --batch-size 1 --num-layers 4 --learning-rate 1e-4
```

### Inference

**Test the Primalcom question:**
```bash
make run
```

**Test a capital city question:**
```bash
make infer
```

**Custom inference:**
```bash
mlx_lm.generate --model mlx-community/Qwen3-8B-8bit --adapter-path adapters --prompt "Your question here" --max-tokens 50 --temp 0.1 --top-p 0.9
```

## Training Results

- **Final validation loss**: 0.310
- **Final training loss**: 0.337
- **Training time**: ~100 iterations
- **Peak memory usage**: 8.8 GB

## Model Performance

✅ **"Who started Primalcom?"** → **"Primalcom was started by Chris and his brother."**
✅ **"What is the capital of Japan?"** → **"The capital of Japan is Tokyo."**
✅ **"What is the capital of France?"** → **"The capital of France is Paris."**

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- MLX-LM package
- ~9GB RAM for training

## Installation

```bash
pip install mlx-lm
```

## Troubleshooting

### Common Issues

1. **Repetitive outputs**: Use temperature control (`--temp 0.1`) in CLI
2. **Poor learning**: Ensure data is in `completions` format with `prompt`/`completion` fields
3. **Memory issues**: Reduce batch size or number of layers

### Data Format Validation

Ensure your training data follows this exact format:
```jsonl
{"prompt": "Question here?", "completion": "Answer here."}
```

## References

- [MLX-LM Documentation](https://github.com/ml-explore/mlx-lm)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MLX Framework](https://github.com/ml-explore/mlx)
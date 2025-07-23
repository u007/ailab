# MLX Vision-Language Model Training

This project has been optimized for Apple Silicon (M1/M2/M3) using MLX, Apple's machine learning framework.

## What is MLX?

MLX is Apple's machine learning framework designed specifically for Apple Silicon. It provides:

- **Native M1/M2/M3 optimization**: Leverages the unified memory architecture and Neural Engine
- **Memory efficiency**: Optimized memory usage for Apple Silicon's unified memory
- **Performance**: Significantly faster training and inference on Apple Silicon
- **Energy efficiency**: Lower power consumption compared to other frameworks

## Key Optimizations for M1 Training

### 1. Model Architecture
- Reduced model size (2048 hidden dimensions vs 4096)
- Fewer transformer layers (16 vs 32) for faster training
- Optimized attention mechanisms for Apple Silicon

### 2. Training Configuration
- Small batch sizes (2) optimized for unified memory
- Reduced epochs (3) for efficient training
- Memory-efficient data processing

### 3. MLX-Specific Features
- Native MLX arrays instead of PyTorch tensors
- MLX neural network modules
- Optimized gradient computation with `nn.value_and_grad`
- Efficient model checkpointing with SafeTensors

## Installation

```bash
# Install MLX dependencies
pip install -r requirements.txt

# Verify MLX installation
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
```

## Usage

### Training

```bash
# Start training with MLX optimization
python train.py
```

The training script will:
- Load and preprocess your vision-language datasets
- Initialize the MLX-optimized model
- Train using Apple Silicon-optimized routines
- Save checkpoints and final model

### Inference

```bash
# Run inference with trained MLX model
python infer_mlx.py
```

## Performance Benefits on Apple Silicon

### Memory Efficiency
- **Unified Memory**: MLX leverages Apple Silicon's unified memory architecture
- **Lower Memory Usage**: Optimized tensor operations reduce memory footprint
- **No GPU Memory Transfer**: Direct access to unified memory eliminates transfer overhead

### Speed Improvements
- **Native Optimization**: MLX is built specifically for Apple Silicon
- **Neural Engine**: Utilizes the dedicated ML accelerator when available
- **Efficient Operations**: Optimized matrix operations and convolutions

### Energy Efficiency
- **Lower Power Consumption**: Apple Silicon's efficiency cores and optimized operations
- **Thermal Management**: Better heat distribution and management
- **Battery Life**: Longer training sessions on MacBook without external power

## Model Architecture

The MLX model includes:

1. **Vision Encoder**: Convolutional layers for image processing
2. **Text Embedding**: Token embeddings for text input
3. **Transformer Layers**: Multi-head attention and feed-forward networks
4. **Output Projection**: Final layer for token generation

## File Structure

```
llmres/
├── train.py              # MLX training script
├── infer_mlx.py          # MLX inference script
├── requirements.txt      # MLX dependencies
├── data/                 # Training datasets
│   ├── billboard/        # Billboard images
│   ├── gantry-samples/   # Gantry images
│   ├── train.jsonl       # Training text data
│   └── valid.jsonl       # Validation text data
└── results/              # Model outputs
    └── final_model/      # Trained MLX model
        ├── model.safetensors
        └── config.json
```

## Training Configuration

```python
# Optimized for M1/M2/M3
model_config = {
    'vocab_size': 32000,
    'hidden_size': 2048,      # Reduced for M1 efficiency
    'num_layers': 16,         # Reduced for M1 efficiency
    'num_attention_heads': 16,
    'max_position_embeddings': 512
}

# Training settings
num_epochs = 3              # Reduced for M1 efficiency
batch_size = 2              # Small batch size for M1
learning_rate = 1e-4        # Optimized learning rate
```

## Troubleshooting

### Common Issues

1. **MLX not found**: Ensure you're on Apple Silicon and have installed MLX
2. **Memory errors**: Reduce batch size or model dimensions
3. **Slow training**: Check that MLX is using the Neural Engine

### Performance Tips

1. **Close other applications** to free up unified memory
2. **Use smaller batch sizes** for better memory efficiency
3. **Monitor Activity Monitor** to check memory and CPU usage
4. **Enable low power mode** for longer training sessions on battery

## Comparison: PyTorch vs MLX

| Feature | PyTorch | MLX |
|---------|---------|-----|
| Apple Silicon Optimization | Limited | Native |
| Memory Usage | Higher | Lower |
| Training Speed | Slower | Faster |
| Energy Efficiency | Lower | Higher |
| Setup Complexity | Higher | Lower |
| Model Size Support | Larger | Optimized |

## Future Improvements

- [ ] Add proper tokenizer integration
- [ ] Implement more sophisticated vision encoder
- [ ] Add support for larger models
- [ ] Optimize for M3 Pro/Max Neural Engine
- [ ] Add distributed training support

## Contributing

Contributions are welcome! Please ensure all changes maintain Apple Silicon optimization and MLX compatibility.

## License

This project is optimized for Apple Silicon using MLX framework.
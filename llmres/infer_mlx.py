#!/usr/bin/env python3
"""
MLX Inference Script for Vision-Language Model
Optimized for Apple Silicon (M1/M2/M3)
"""

import os
import json
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path

class MLXVisionLanguageModel(nn.Module):
    """MLX-optimized Vision-Language Model for M1 inference."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = config.get('vocab_size', 32000)
        self.hidden_size = config.get('hidden_size', 2048)
        self.num_layers = config.get('num_layers', 16)
        self.num_heads = config.get('num_attention_heads', 16)
        
        # Vision encoder (simplified)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, self.hidden_size)
        )
        
        # Text embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Transformer layers
        self.layers = [self._make_layer() for _ in range(self.num_layers)]
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        
    def _make_layer(self):
        """Create a transformer layer optimized for MLX."""
        return nn.Sequential(
            nn.MultiHeadAttention(self.hidden_size, self.num_heads),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
    
    def __call__(self, input_ids: mx.array, pixel_values: mx.array = None):
        # Text embeddings
        text_embeds = self.token_embedding(input_ids)
        
        # Vision embeddings (if provided)
        if pixel_values is not None:
            vision_embeds = self.vision_encoder(pixel_values)
            # Combine text and vision embeddings
            combined_embeds = mx.concatenate([vision_embeds[:, None, :], text_embeds], axis=1)
        else:
            combined_embeds = text_embeds
        
        # Pass through transformer layers
        hidden_states = combined_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        return logits

def load_mlx_model(model_path: str):
    """Load MLX model from checkpoint."""
    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "model.safetensors")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = MLXVisionLanguageModel(config)
    
    # Load weights
    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))
    
    return model, config

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image for MLX inference."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    pixel_values = np.array(image).astype(np.float32) / 255.0
    pixel_values = np.transpose(pixel_values, (2, 0, 1))  # CHW format
    return pixel_values

def tokenize_text(text: str, max_length: int = 512) -> list:
    """Simple tokenization (replace with proper tokenizer in production)."""
    tokens = [ord(c) % 32000 for c in text[:max_length]]
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    return tokens

def generate_response(model, prompt: str, image_path: str = None, max_tokens: int = 100):
    """Generate response using MLX model."""
    # Tokenize prompt
    input_ids = tokenize_text(prompt)
    input_ids = mx.array([input_ids])
    
    # Process image if provided
    pixel_values = None
    if image_path and os.path.exists(image_path):
        pixel_values = preprocess_image(image_path)
        pixel_values = mx.array([pixel_values])
    
    # Generate tokens
    generated_tokens = []
    
    for _ in range(max_tokens):
        # Forward pass
        logits = model(input_ids, pixel_values)
        
        # Get next token (simple greedy decoding)
        next_token = mx.argmax(logits[0, -1, :]).item()
        
        # Stop if end token
        if next_token == 0:  # Assuming 0 is end token
            break
            
        generated_tokens.append(next_token)
        
        # Update input_ids for next iteration
        next_token_array = mx.array([[next_token]])
        input_ids = mx.concatenate([input_ids, next_token_array], axis=1)
    
    # Convert tokens back to text (simplified)
    response = ''.join([chr(token % 128) for token in generated_tokens if token > 0])
    return response

def main():
    """Main inference function."""
    model_path = "/Users/james/www/ailab/llmres/results/final_model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    print("Loading MLX model...")
    try:
        model, config = load_mlx_model(model_path)
        print(f"Model loaded successfully with config: {config}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example inference
    print("\n=== MLX Vision-Language Model Inference ===")
    
    # Text-only inference
    prompt = "What type of outdoor media is shown in this image?"
    print(f"\nPrompt: {prompt}")
    response = generate_response(model, prompt)
    print(f"Response: {response}")
    
    # Image + text inference (if image exists)
    image_path = "/Users/james/www/ailab/llmres/data/billboard/sample.jpg"
    if os.path.exists(image_path):
        print(f"\nPrompt with image: {prompt}")
        print(f"Image: {image_path}")
        response = generate_response(model, prompt, image_path)
        print(f"Response: {response}")
    else:
        print(f"\nNo sample image found at {image_path}")
    
    print("\n=== Inference completed ===")

if __name__ == '__main__':
    main()
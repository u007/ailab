import os
import json
from PIL import Image
from datasets import load_dataset, Dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.utils import load_config
import numpy as np
from typing import Dict, List, Any, Optional
from download_utils import ResumableDownloader, progress_bar
from huggingface_hub import snapshot_download
import shutil
from pathlib import Path
import time

def create_image_dataset(data_path: str, split: str = 'train') -> Dataset:
    """Create dataset from image directories for outdoor media classification."""
    image_data = []
    
    # Billboard images
    billboard_dir = os.path.join(data_path, 'billboard')
    if os.path.exists(billboard_dir):
        for img_file in os.listdir(billboard_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_data.append({
                    'image_path': os.path.join(billboard_dir, img_file),
                    'media_type': 'billboard',
                    'prompt': 'What type of outdoor media is shown in this image?',
                    'completion': 'billboard'
                })
    
    # Gantry samples (if any exist)
    gantry_dir = os.path.join(data_path, 'gantry-samples')
    if os.path.exists(gantry_dir):
        for img_file in os.listdir(gantry_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_data.append({
                    'image_path': os.path.join(gantry_dir, img_file),
                    'media_type': 'gantry',
                    'prompt': 'What type of outdoor media is shown in this image?',
                    'completion': 'gantry'
                })
    
    # Split data for train/validation
    if split == 'train':
        image_data = image_data[:int(0.8 * len(image_data))]
    else:
        image_data = image_data[int(0.8 * len(image_data)):]
    
    return Dataset.from_list(image_data)

def load_combined_datasets(data_path: str):
    """Load both text-only and image datasets."""
    # Load existing text datasets
    text_train = load_dataset('json', data_files=os.path.join(data_path, 'train.jsonl'), split='train')
    text_valid = load_dataset('json', data_files=os.path.join(data_path, 'valid.jsonl'), split='train')
    
    # Load image datasets
    image_train = create_image_dataset(data_path, 'train')
    image_valid = create_image_dataset(data_path, 'valid')
    
    # Add image_path column to text datasets (None for text-only)
    text_train = text_train.add_column('image_path', [None] * len(text_train))
    text_valid = text_valid.add_column('image_path', [None] * len(text_valid))
    text_train = text_train.add_column('media_type', [None] * len(text_train))
    text_valid = text_valid.add_column('media_type', [None] * len(text_valid))
    
    # Combine datasets
    from datasets import concatenate_datasets
    combined_train = concatenate_datasets([text_train, image_train])
    combined_valid = concatenate_datasets([text_valid, image_valid])
    
    return combined_train, combined_valid

def download_mlx_model(model_name: str, cache_dir: str = None):
    """Download and convert model for MLX with resume capability."""
    print(f"Downloading MLX-compatible model: {model_name}")
    
    try:
        # Use HuggingFace's built-in resume capability
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,  # Enable resume
            local_files_only=False
        )
        print(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

class MLXVisionLanguageModel(nn.Module):
    """MLX-optimized Vision-Language Model for M1 training."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = config.get('vocab_size', 32000)
        self.hidden_size = config.get('hidden_size', 4096)
        self.num_layers = config.get('num_layers', 32)
        self.num_heads = config.get('num_attention_heads', 32)
        
        # Vision encoder (simplified for MLX compatibility)
        # Use adaptive linear layers that can handle variable input sizes
        self.vision_linear1 = nn.Linear(512, 256)  # Reduced input size
        self.vision_linear2 = nn.Linear(256, self.hidden_size)
        self.vision_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Text embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Transformer layers
        self.layers = [self._make_layer() for _ in range(self.num_layers)]
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        
    def _make_layer(self):
        """Create a transformer layer optimized for MLX."""
        class TransformerLayer(nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.attention = nn.MultiHeadAttention(hidden_size, num_heads)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm2 = nn.LayerNorm(hidden_size)
            
            def __call__(self, x):
                # Self-attention with residual connection
                attn_out = self.attention(x, x, x)  # query, key, value
                x = self.norm1(x + attn_out)
                
                # MLP with residual connection
                mlp_out = self.mlp(x)
                x = self.norm2(x + mlp_out)
                
                return x
        
        return TransformerLayer(self.hidden_size, self.num_heads)
    
    def __call__(self, input_ids: mx.array, pixel_values: Optional[mx.array] = None):
        # Text embeddings
        text_embeds = self.token_embedding(input_ids)
        
        # Ensure text_embeds has the right shape [batch, seq_len, hidden_size]
        if len(text_embeds.shape) == 2:
            text_embeds = mx.expand_dims(text_embeds, axis=1)  # Add sequence dimension
        
        # Vision embeddings (if provided)
        if pixel_values is not None:
            # Get batch size from text_embeds to ensure consistency
            batch_size = text_embeds.shape[0]
            
            # Handle pixel_values - ensure it matches batch size
            if pixel_values.shape[0] != batch_size:
                # If pixel_values has different batch size, repeat or truncate
                if pixel_values.shape[0] == 1:
                    # Repeat single image for all batch items
                    pixel_values = mx.broadcast_to(pixel_values, (batch_size,) + pixel_values.shape[1:])
                else:
                    # Take first batch_size items
                    pixel_values = pixel_values[:batch_size]
            
            # MLX vision processing with adaptive sizing
            x = mx.reshape(pixel_values, (batch_size, -1))  # Flatten
            # Reduce dimensionality to match our linear layer input
            if x.shape[1] > 512:
                # Simple downsampling by taking every nth element
                step = x.shape[1] // 512
                x = x[:, ::step][:, :512]  # Take every step-th element, limit to 512
            elif x.shape[1] < 512:
                # Pad with zeros if too small
                padding = mx.zeros((batch_size, 512 - x.shape[1]))
                x = mx.concatenate([x, padding], axis=1)
            
            x = mx.maximum(self.vision_linear1(x), 0)  # ReLU activation
            x = mx.maximum(self.vision_linear2(x), 0)  # ReLU activation
            vision_features = self.vision_projection(x)
            # Expand vision features to match sequence length [batch, 1, hidden_size]
            vision_features = mx.expand_dims(vision_features, axis=1)
            combined_embeds = mx.concatenate([vision_features, text_embeds], axis=1)
        else:
            combined_embeds = text_embeds
        
        # Pass through transformer layers
        hidden_states = combined_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        return logits

def download_dataset_files(urls_and_paths: list, data_dir: str):
    """Download dataset files with resume capability."""
    downloader = ResumableDownloader()
    
    def progress_callback(downloaded, total):
        progress_bar(downloaded, total)
    
    os.makedirs(data_dir, exist_ok=True)
    
    for url, filename in urls_and_paths:
        filepath = os.path.join(data_dir, filename)
        print(f"\nDownloading {filename}...")
        
        success = downloader.download_file(
            url=url,
            filepath=filepath,
            progress_callback=progress_callback
        )
        
        if not success:
            print(f"\nFailed to download {filename}")
            return False
        print(f"\nCompleted: {filename}")
    
    return True

def preprocess_for_mlx(dataset):
    """Preprocess dataset for MLX training."""
    processed_data = []
    
    for example in dataset:
        # Extract text data
        if 'prompt' in example and 'completion' in example:
            prompt = example['prompt']
            completion = example['completion']
        elif 'text' in example:
            # Split text into prompt and completion if needed
            text = example['text']
            if 'Assistant:' in text:
                parts = text.split('Assistant:', 1)
                prompt = parts[0].replace('User:', '').strip()
                completion = parts[1].strip()
            else:
                prompt = text[:len(text)//2]
                completion = text[len(text)//2:]
        else:
            continue  # Skip if no valid text data
        
        # Simple tokenization (in practice, use proper tokenizer)
        text = f"{prompt} {completion}"
        tokens = [ord(c) % 32000 for c in text[:512]]  # Simple char-level tokenization
        
        # Pad to fixed length
        if len(tokens) < 512:
            tokens.extend([0] * (512 - len(tokens)))
        
        # Process image if available
        pixel_values = None
        img_path = example.get('image_path')
        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize((224, 224))
                pixel_values = np.array(image).astype(np.float32) / 255.0
                pixel_values = np.transpose(pixel_values, (2, 0, 1))  # CHW format
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                pixel_values = np.zeros((3, 224, 224), dtype=np.float32)
        else:
            pixel_values = np.zeros((3, 224, 224), dtype=np.float32)
        
        processed_data.append({
            'input_ids': tokens,
            'pixel_values': pixel_values,
            'labels': tokens  # For causal LM
        })
    
    return processed_data

def train_step(model, batch, optimizer):
    """Single training step optimized for MLX."""
    def loss_fn(model):
        # Handle input_ids - properly stack batch items
        if isinstance(batch['input_ids'], list):
            # Stack all items in the batch
            input_ids = mx.array(batch['input_ids'])  # Shape: [batch_size, seq_len]
        else:
            input_ids = mx.array(batch['input_ids'])
            if len(input_ids.shape) == 1:
                input_ids = mx.expand_dims(input_ids, axis=0)  # Add batch dimension
        
        # Handle labels - properly stack batch items
        if isinstance(batch['labels'], list):
            # Stack all items in the batch
            labels = mx.array(batch['labels'])  # Shape: [batch_size, seq_len]
        else:
            labels = mx.array(batch['labels'])
            if len(labels.shape) == 1:
                labels = mx.expand_dims(labels, axis=0)  # Add batch dimension
        
        # Handle pixel_values properly
        pixel_values = None
        if batch['pixel_values'] is not None:
            if isinstance(batch['pixel_values'], mx.array):
                pixel_values = batch['pixel_values']
            else:
                # Convert numpy array to MLX array
                pixel_values = mx.array(batch['pixel_values'])
        
        logits = model(input_ids, pixel_values)
        
        # Handle sequence length mismatch due to vision tokens
        if pixel_values is not None:
            # Skip vision tokens in logits (first token is vision, rest are text)
            text_logits = logits[:, 1:, :]  # Skip first vision token
            # Ensure text_logits matches labels length
            seq_len = min(text_logits.shape[1], labels.shape[1] if len(labels.shape) > 1 else labels.shape[0])
            text_logits = text_logits[:, :seq_len, :]
            if len(labels.shape) == 1:
                target_labels = labels[:seq_len]
            else:
                target_labels = labels[:, :seq_len]
        else:
            text_logits = logits
            target_labels = labels
        
        # Compute cross-entropy loss
        loss = nn.losses.cross_entropy(text_logits.reshape(-1, text_logits.shape[-1]), target_labels.reshape(-1))
        # Ensure loss is a scalar by taking the mean
        loss = mx.mean(loss)
        return loss
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model)
    
    # Update parameters
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    return loss

def main():
    # Define paths
    data_path = './data/'
    output_dir = './results'
    model_cache_dir = './models_cache'

    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)

    # MLX model configuration optimized for M1
    model_config = {
        'vocab_size': 32000,
        'hidden_size': 2048,  # Reduced for M1 efficiency
        'num_layers': 16,     # Reduced for M1 efficiency
        'num_attention_heads': 16,
        'max_position_embeddings': 512
    }
    
    print("Initializing MLX Vision-Language Model for M1...")
    model = MLXVisionLanguageModel(model_config)
    
    # Initialize optimizer (AdamW optimized for MLX)
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    # Load the combined datasets (text + images)
    train_dataset, valid_dataset = load_combined_datasets(data_path)

    # Preprocess datasets for MLX
    print("Preprocessing training data for MLX...")
    train_data = preprocess_for_mlx(train_dataset)
    valid_data = preprocess_for_mlx(valid_dataset)
    
    # Training configuration optimized for M1
    num_epochs = 3  # Reduced for M1 efficiency
    batch_size = 2  # Small batch size for M1
    log_interval = 10
    save_interval = 100
    
    print(f"Starting MLX training on M1 with {len(train_data)} samples...")
    
    # Training loop optimized for M1
    step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_start_time = time.time()
        total_loss = 0
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            
            # Prepare batch with proper handling of pixel_values
            batch = {
                'input_ids': [item['input_ids'] for item in batch_data],
                'pixel_values': None,
                'labels': [item['labels'] for item in batch_data]
            }
            
            # Handle pixel_values - collect non-None values
            pixel_values_list = [item['pixel_values'] for item in batch_data if item['pixel_values'] is not None]
            if pixel_values_list:
                # Use first available pixel_values for the batch
                batch['pixel_values'] = pixel_values_list[0]
            
            # Training step
            loss = train_step(model, batch, optimizer)
            total_loss += loss.item()
            step += 1
            
            # Logging
            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"Step {step}, Loss: {avg_loss:.4f}")
                total_loss = 0
            
            # Save checkpoint
            if step % save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model parameters using MLX's native format
                try:
                    model_params = dict(model.parameters())
                    # Convert to numpy arrays for saving
                    numpy_params = {k: np.array(v) for k, v in model_params.items()}
                    np.savez(os.path.join(checkpoint_path, 'model_params.npz'), **numpy_params)
                    print(f"\nCheckpoint saved at step {step}")
                except Exception as e:
                    print(f"\nWarning: Could not save checkpoint at step {step}: {e}")
                
                # Save training state
                training_state = {
                    'step': step,
                    'epoch': epoch,
                    'loss': float(loss),
                    'learning_rate': optimizer.learning_rate
                }
                
                with open(os.path.join(checkpoint_path, 'training_state.json'), 'w') as f:
                    json.dump(training_state, f, indent=2)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        # Validation
        if valid_data:
            print("Running validation...")
            val_loss = 0
            val_steps = 0
            
            for i in range(0, min(len(valid_data), 50), batch_size):  # Limited validation for speed
                batch_data = valid_data[i:i + batch_size]
                batch = {
                    'input_ids': [item['input_ids'] for item in batch_data],
                    'pixel_values': [item['pixel_values'] for item in batch_data],
                    'labels': [item['labels'] for item in batch_data]
                }
                
                # Validation step (no gradient update)
                input_ids = mx.array(batch['input_ids'])
                pixel_values = mx.array(batch['pixel_values']) if batch['pixel_values'][0] is not None else None
                labels = mx.array(batch['labels'])
                
                logits = model(input_ids, pixel_values)
                loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                
                val_loss += loss.item()
                val_steps += 1
            
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # Save final model parameters using numpy format
    try:
        model_params = dict(model.parameters())
        numpy_params = {k: np.array(v) for k, v in model_params.items()}
        np.savez(os.path.join(final_model_path, 'model_params.npz'), **numpy_params)
    except Exception as e:
        print(f"Warning: Could not save final model: {e}")
    
    # Save model config
    with open(os.path.join(final_model_path, "config.json"), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\nTraining completed! Model saved to {final_model_path}")
    print("MLX training optimized for M1 finished successfully.")

if __name__ == '__main__':
    main()
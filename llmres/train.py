import os
import json
from PIL import Image
from datasets import load_dataset, Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, Trainer
import torch
from typing import Dict, List, Any
from qwen_vl_utils import process_vision_info
from download_utils import ResumableDownloader, progress_bar
from huggingface_hub import snapshot_download
import shutil

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

def download_model_with_resume(model_name: str, cache_dir: str = None):
    """Download model with resume capability using HuggingFace Hub."""
    print(f"Downloading model: {model_name}")
    
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

def main():
    # Define paths
    data_path = '/Users/james/www/ailab/llmres/data/'
    output_dir = '/Users/james/www/ailab/llmres/results'
    model_cache_dir = '/Users/james/www/ailab/llmres/models_cache'

    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)

    # Download model with resume capability
    model_name = 'Qwen/Qwen2-VL-7B-Instruct'  # Using Qwen2-VL for vision capabilities
    
    try:
        model_path = download_model_with_resume(model_name, model_cache_dir)
    except Exception as e:
        print(f"Model download failed: {e}")
        print("Attempting to load from cache or download without resume...")
        model_path = model_name  # Fallback to original behavior

    # Load the combined datasets (text + images)
    train_dataset, valid_dataset = load_combined_datasets(data_path)

    # Load vision-language model, processor and tokenizer
    print("Loading model, processor, and tokenizer...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path if os.path.exists(model_path) else model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=model_cache_dir,
        resume_download=True  # Enable resume for transformers
    )
    processor = AutoProcessor.from_pretrained(
        model_path if os.path.exists(model_path) else model_name,
        cache_dir=model_cache_dir,
        resume_download=True
    )
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Preprocess the data for vision-language model
    def process_function(examples):
        prompts = examples['prompt']
        completions = examples['completion']
        image_paths = examples.get('image_path', [None] * len(prompts))
        
        # Process each example individually and collect results
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        labels_list = []
        
        for prompt, completion, img_path in zip(prompts, completions, image_paths):
            # Create conversation format
            if img_path is not None and os.path.exists(img_path):
                # For image-text pairs
                try:
                    image = Image.open(img_path).convert('RGB')
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": prompt}
                            ]
                        },
                        {"role": "assistant", "content": completion}
                    ]
                    
                    # Apply chat template and process
                    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                    inputs = processor(text=text, images=[image], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                    
                    input_ids_list.append(inputs['input_ids'].squeeze(0))
                    attention_mask_list.append(inputs['attention_mask'].squeeze(0))
                    if 'pixel_values' in inputs:
                        pixel_values_list.append(inputs['pixel_values'].squeeze(0))
                    else:
                        pixel_values_list.append(torch.zeros(3, 448, 448))  # Dummy image
                        
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    # Fallback to text-only processing
                    conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"The answer is {completion}."}]
                    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                    inputs = processor(text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                    
                    input_ids_list.append(inputs['input_ids'].squeeze(0))
                    attention_mask_list.append(inputs['attention_mask'].squeeze(0))
                    pixel_values_list.append(torch.zeros(3, 448, 448))  # Dummy image
            else:
                # For text-only pairs
                conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"The answer is {completion}."}]
                text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                inputs = processor(text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                
                input_ids_list.append(inputs['input_ids'].squeeze(0))
                attention_mask_list.append(inputs['attention_mask'].squeeze(0))
                pixel_values_list.append(torch.zeros(3, 448, 448))  # Dummy image for text-only
            
            # Labels are the same as input_ids for causal language modeling
            labels_list.append(inputs['input_ids'].squeeze(0))
        
        # Stack all tensors
        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'pixel_values': torch.stack(pixel_values_list),
            'labels': torch.stack(labels_list)
        }

    processed_train_dataset = train_dataset.map(process_function, batched=True, remove_columns=["prompt", "completion", "image_path", "media_type"])
    processed_valid_dataset = valid_dataset.map(process_function, batched=True, remove_columns=["prompt", "completion", "image_path", "media_type"]) 

    # Set up training arguments for vision model
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Fewer epochs for vision model
        per_device_train_batch_size=1,  # Smaller batch size for vision model
        per_device_eval_batch_size=1,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=3,
        learning_rate=1e-5,  # Learning rate for vision model
        gradient_accumulation_steps=8,  # Higher accumulation for effective larger batch
        fp16=True,  # Use mixed precision for memory efficiency
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        remove_unused_columns=False,  # Keep all columns for multimodal processing
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_valid_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == '__main__':
    main()
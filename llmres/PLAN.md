# Project Plan: QLoRA Fine-Tuning of Qwen-VL on MLX for Outdoor Media Type Identification

## Objective
Set up the repo to fine-tune a Qwen-based multimodal model (using Qwen-VL-7B as proxy for 8B) with QLoRA on Apple M1 using MLX, training on images to classify outdoor media types (e.g., billboards, digital signs, posters).

## Step 1: Research and Planning
- Confirm model: Use Qwen-VL-7B via MLX-LM for vision-language capabilities.
- Identify datasets: Search for image datasets of outdoor advertising/media with type labels.

## Step 2: Dataset Acquisition
- Use web search to find suitable datasets (e.g., Outdoor Advertising Dataset or similar).
- Download and preprocess datasets into the repo.

## Step 3: Environment Setup
- Install MLX and MLX-LM.
- Handle dependencies for QLoRA fine-tuning on M1.

## Step 4: Implementation
- Write `train.py` to load model, prepare data, and run QLoRA training.
- Ensure script handles image-text pairs for classification.

## Step 5: Testing and Completion
- Test the setup.
- Document in README if needed.
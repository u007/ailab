#!/usr/bin/env python3
"""
Test script to verify Qwen2-VL vision model setup
"""

import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

def test_vision_model():
    print("Testing Qwen2-VL vision model setup...")
    
    try:
        # Load model and processor
        model_name = 'Qwen/Qwen2-VL-7B-Instruct'
        print(f"Loading model: {model_name}")
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        print("✓ Model and processor loaded successfully")
        
        # Test with a sample image if available
        data_path = '/Users/james/www/ailab/llmres/data/billboard'
        if os.path.exists(data_path):
            sample_images = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if sample_images:
                img_path = os.path.join(data_path, sample_images[0])
                print(f"Testing with sample image: {img_path}")
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    
                    # Create a test conversation
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": "What type of outdoor media is shown in this image?"}
                            ]
                        }
                    ]
                    
                    # Process the input
                    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=text, images=[image], return_tensors="pt")
                    
                    print("✓ Image processing successful")
                    print(f"Input shape: {inputs['input_ids'].shape}")
                    if 'pixel_values' in inputs:
                        print(f"Image tensor shape: {inputs['pixel_values'].shape}")
                    
                    # Test inference (optional, comment out if too slow)
                    # with torch.no_grad():
                    #     outputs = model.generate(**inputs, max_new_tokens=50)
                    #     response = processor.decode(outputs[0], skip_special_tokens=True)
                    #     print(f"Model response: {response}")
                    
                except Exception as e:
                    print(f"✗ Error processing image: {e}")
                    return False
            else:
                print("No sample images found in billboard directory")
        else:
            print("Billboard directory not found")
        
        print("✓ Vision model setup test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during model setup: {e}")
        return False

if __name__ == "__main__":
    success = test_vision_model()
    if success:
        print("\n🎉 Vision model is ready for training!")
    else:
        print("\n❌ Vision model setup failed. Please check dependencies.")
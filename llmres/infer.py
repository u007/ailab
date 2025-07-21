import mlx.core as mx
from mlx_lm import load, generate

def main():
    # Load the fine-tuned MLX model and tokenizer
    model_dir = '/Users/james/www/ailab/llmres/adapters'  # Path to the LoRA adapters
    base_model = 'mlx-community/Qwen3-8B-8bit'  # Base model used for training
    
    # Load model and tokenizer
    model, tokenizer = load(base_model, adapter_path=model_dir)

    # Get user input from command-line arguments
    import sys
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = "What is the capital of Japan?"

    # Generate a response using MLX
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=50,
        verbose=False
    )
    
    # Clean up the response
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == '__main__':
    main()
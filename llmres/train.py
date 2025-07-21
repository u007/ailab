import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

def main():
    # Define paths
    data_path = '/Users/james/www/ailab/llmres/data/'
    output_dir = '/Users/james/www/ailab/llmres/results'

    # Load the datasets
    train_dataset = load_dataset('json', data_files=os.path.join(data_path, 'train.jsonl'), split='train')
    valid_dataset = load_dataset('json', data_files=os.path.join(data_path, 'valid.jsonl'), split='train')

    # Load tokenizer and model
    model_name = 'Qwen/Qwen3-8B-Instruct'  # Using Qwen3-8B model as requested
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Preprocess the data
    def tokenize_function(examples):
        prompts = examples['prompt']
        completions = examples['completion']
        
        # Format inputs and outputs for better learning
        formatted_inputs = prompts
        formatted_outputs = ["The answer is " + c + "." for c in completions]
        
        # Combine for training (input followed by output)
        combined_texts = [inp + " " + out + tokenizer.eos_token for inp, out in zip(formatted_inputs, formatted_outputs)]
        
        # Tokenize the combined text
        model_inputs = tokenizer(combined_texts, max_length=256, truncation=True, padding="max_length")
        
        # Create labels - mask the prompt tokens with -100 so they don't contribute to loss
        labels = model_inputs["input_ids"].copy()
        for i, prompt in enumerate(formatted_inputs):
            # Tokenize just the prompt to get its length
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_length = len(prompt_ids)
            # Set labels for prompt tokens to -100
            labels[i][:prompt_length] = [-100] * prompt_length
        
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "completion"])
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "completion"]) 

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,  # Increased epochs for better learning
        per_device_train_batch_size=2,  # Increased batch size
        per_device_eval_batch_size=2,
        warmup_steps=100,  # More warmup steps
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        # Removed evaluation_strategy and save_strategy
        save_total_limit=3,  # Only keep the 3 best checkpoints
        # Removed load_best_model_at_end
        learning_rate=5e-6,  # Lower learning rate for more stable training
        gradient_accumulation_steps=4,  # Accumulate gradients for effectively larger batch size
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == '__main__':
    main()
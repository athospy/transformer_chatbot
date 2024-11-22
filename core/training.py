import os
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler,
    DataCollatorWithPadding
)
from torch.optim import AdamW
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd


class DataCollatorWithLabelSmoothing(DataCollatorWithPadding):
    def __init__(self, tokenizer, label_smoothing_factor=0.2):
        super().__init__(tokenizer)
        self.label_smoothing_factor = label_smoothing_factor
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            labels = batch["labels"]
            # Replace padding tokens with -100 for causal modeling
            labels[labels == self.tokenizer.pad_token_id] = -100
        return batch
    

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        """
        Custom loss computation.
        """
        # Extract labels from inputs
        labels = inputs.pop("labels")
        
        # Forward pass through the model
        outputs = model(**inputs)
        
        # Compute loss using the logits and labels
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss



# Convert list of dictionaries to dictionary of lists
def convert_to_dict_of_lists(data):
    inputs = [item['input'] for item in data]
    responses = [item['response'] for item in data]
    return {'input': inputs, 'response': responses}


def main():
    specific_responses_file = os.path.join(os.path.dirname(__file__), 'programs_simpler.tsv')
    specific_responses_data = pd.read_csv(specific_responses_file, sep='\t', header=None, names=['input', 'response']).to_dict(orient='records')

    converted_data = convert_to_dict_of_lists(specific_responses_data)


    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(converted_data)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    # Explicitly set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.add_special_tokens({"additional_special_tokens": ["[QUESTION]", "[ANSWER]"]})

    def tokenize_function(examples):
        max_length = 1024  # Ensuring same max_length for both input and label
        inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=max_length)
        labels = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=max_length)
        
        # The labels will be used as the target for causal language modeling
        inputs["labels"] = labels["input_ids"]
        
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split into train and validation sets
    dataset_split = tokenized_datasets.train_test_split(test_size=0.2)  # 80% train, 20% validation
    train_dataset = dataset_split['train']
    val_dataset = dataset_split['test']

    # Load model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    # Ensure the model's pad_token_id is also set
    model.config.pad_token_id = tokenizer.eos_token_id

    model.resize_token_embeddings(len(tokenizer))

    model.config.hidden_dropout_prob = 0.2  # Set dropout probability
    model.config.attention_probs_dropout_prob = 0.2


    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Adjusted for AdamW


    # Define the scheduler
    num_training_steps = len(train_dataset) // 8 * 10  # Batch size (8) and epochs (10)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)

    data_collator = DataCollatorWithLabelSmoothing(
        tokenizer=tokenizer,
        label_smoothing_factor=0.2,
    )
    
    training_args = TrainingArguments(
        output_dir="./results",                    # Directory to save model checkpoints and outputs
        eval_strategy="steps",                     # Evaluate the model every few steps
        eval_steps=200,                            # Evaluate the model after every 200 steps
        save_strategy="steps",                     # Save checkpoints every few steps
        save_steps=1000,                           # Save the model checkpoint every 1000 steps
        save_total_limit=2,                        # Keep only the last 2 checkpoints to save space
        learning_rate=2e-5,                        # Learning rate for the optimizer
        warmup_steps=100,                          # Number of warm-up steps for learning rate scheduler
        per_device_train_batch_size=2,             # Batch size per device during training
        gradient_accumulation_steps=8,             # Accumulate gradients over 4 steps to simulate a larger batch size
        num_train_epochs=10,                       # Train the model for 10 epochs
        weight_decay=0.01,                         # Apply weight decay to prevent overfitting
        logging_dir="./logs",                      # Directory to save training logs
        logging_steps=100,                         # Log training metrics every 100 steps
        load_best_model_at_end=True,               # Load the best model checkpoint at the end of training
        max_grad_norm=1.0,                         # Clip gradients to this value to prevent exploding gradients
        label_smoothing_factor=0.2,                # Apply label smoothing to handle noisy labels
        use_cpu=True                               # Train using CPU (set to False for GPU if available)
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),  # Custom optimizer and scheduler
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    model.save_pretrained("program_model")
    tokenizer.save_pretrained("program_tokenizer")

    # Generate a sample response after training
    sample_inputs = [
        "What programs are available in engineering?",
        "Tell me about healthcare programs.",
        "Does the university offer online programs?"
    ]

    for sample_input in sample_inputs:
        inputs = tokenizer(sample_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            pad_token_id=tokenizer.pad_token_id,
        )
        print(f"Input: {sample_input}")
        print(f"Generated Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")


if __name__ == '__main__':
    main()

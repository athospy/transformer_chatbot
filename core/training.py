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

'''
data collator is used to pad the input and output sequences to the same length
this is necessary for training the model
as the model expects inputs and outputs to be of the same length
the label smoothing factor is used to prevent the model from overfitting to the training data
20% of the time the model will be penalized for being too confident in its predictions
the -100 value is used to ignore the loss for padding tokens as they are not relevant for training
'''
class DataCollatorWithLabelSmoothing(DataCollatorWithPadding):
    def __init__(self, tokenizer, label_smoothing_factor=0.2):
        super().__init__(tokenizer)
        # Label smoothing factor 20%
        self.label_smoothing_factor = label_smoothing_factor
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = super().__call__(features)
        # Apply label smoothing to the labels
        if "labels" in batch:
            labels = batch["labels"]
            # Compute the number of labels to smooth, -100 are ignored for loss
            labels[labels == self.tokenizer.pad_token_id] = -100
        return batch
    

'''
custom trainer class is used to compute the loss
the loss is computed using the logits and labels that are returned by the model by forward pass
the loss is computed using the cross entropy loss function
the loss is then returned to the trainer
logits are reshaped to 2D tensor and labels are reshaped to 1D tensor
logit means the output of the model before applying the softmax function
'''
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        # Extract labels from inputs
        labels = inputs.pop("labels")
        
        # Forward pass through the model
        outputs = model(**inputs)
        
        # Compute loss using the logits and labels
        logits = outputs.logits
        # Reshape logits and labels to 2D tensor
        loss_fn = torch.nn.CrossEntropyLoss()
        # the -1 means that the size of that dimension is inferred
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss



# Convert list of dictionaries to dictionary of lists
# This is necessary for converting the data to Hugging Face Dataset
def convert_to_dict_of_lists(data):
    inputs = [item['input'] for item in data]
    responses = [item['response'] for item in data]
    return {'input': inputs, 'response': responses}


def main():
    # Load the dataset
    specific_responses_file = os.path.join(os.path.dirname(__file__), 'programs_simpler.tsv')
    specific_responses_data = pd.read_csv(specific_responses_file, sep='\t', header=None, names=['input', 'response']).to_dict(orient='records')

    # Convert to dictionary of lists
    converted_data = convert_to_dict_of_lists(specific_responses_data)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(converted_data)

    # the tokenizer is used to convert the input and output sequences to tokens
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

    # Explicitly set the pad_token to eos_token for GPT models
    tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.add_special_tokens({"additional_special_tokens": ["[QUESTION]", "[ANSWER]"]})

    # Define a function to tokenize the examples
    def tokenize_function(examples):
        # Ensuring same max_length for both input and label
        max_length = 1024  
        # Tokenize the inputs and labels with the same max_length, 1024 is used because it is the maximum length that the model can handle
        inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=max_length)
        labels = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=max_length)
        
        # The labels will be used as the target for causal language modeling
        inputs["labels"] = labels["input_ids"]
        
        return inputs

    # Tokenize the dataset using the tokenize_function for both the input and the target
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split into train and validation sets
    dataset_split = tokenized_datasets.train_test_split(test_size=0.2)  # 80% train, 20% validation
    train_dataset = dataset_split['train']
    val_dataset = dataset_split['test']

    # Load base model for fine-tuning
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    # pad_token_id is used to ignore the loss for padding tokens
    model.config.pad_token_id = tokenizer.eos_token_id

    # Resize the token embeddings to include the newly added tokens
    model.resize_token_embeddings(len(tokenizer))

    # 0.2 is used as the dropout probability of the hidden layers
    model.config.hidden_dropout_prob = 0.2 
    # 0.2 is used as the dropout probability of the attention layers
    model.config.attention_probs_dropout_prob = 0.2

    # Define the optimizer is used to update the weights of the model
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Adjusted for AdamW


    # Define the scheduler to adjust the learning rate during training
    num_training_steps = len(train_dataset) // 8 * 10  # Batch size (8) and epochs (10)
    # Cosine annealing scheduler is used to adjust the learning rate by following the cosine function
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)

    # Define the data collator to pad the input and output sequences to the same length
    data_collator = DataCollatorWithLabelSmoothing(
        tokenizer=tokenizer,
        label_smoothing_factor=0.2,
    )
    
    # Define the training arguments
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

    # Create a custom trainer for fine-tuning
    trainer = CustomTrainer(
        model=model,                                                    # The model to be trained and evaluated
        args=training_args,                                             # Training arguments for customizing the training process
        train_dataset=train_dataset,                                    # Training dataset with input and target sequences
        eval_dataset=val_dataset,                                       # Evaluation dataset that is split from the training dataset
        data_collator=data_collator,                                    # Custom data collator for padding and label smoothing
        optimizers=(optimizer, scheduler),                              # Custom optimizer and scheduler by passing as tuple
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   # Callback for early stopping after 3 evaluations
    )

    # Train the model
    trainer.train()
    # save the model
    model.save_pretrained("program_model")
    # also save the tokenizer to use it later
    tokenizer.save_pretrained("program_tokenizer")

    # Generate a sample response after training
    sample_inputs = [
        "What programs are available in engineering?",
        "Tell me about healthcare programs.",
        "Does the university offer online programs?"
    ]

    # test the model by generating responses to the sample inputs
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

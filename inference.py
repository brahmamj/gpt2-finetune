import os
import re
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from clearml import Task,Dataset
from clearml import PipelineDecorator

checkpoint = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.unk_token
def generate_response(model, tokenizer, prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")      # 'pt' for returning pytorch tensor

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id
    output = model.generate(
        input_ids,                    # Input tokens
        max_length=max_length,        # Maximum length of the generated response
        num_return_sequences=1,       # Generate one sequence
        attention_mask=attention_mask, # Attention mask
        pad_token_id=pad_token_id     # Pad token ID
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

task = Task.get_task(project_name="GPT2-FineTuning", task_name="test-1")
print("Task ID:", task.id)
saved_model_path=task.artifacts["finetuned_gpt2_model"].get()
print("Saved model path:", saved_model_path)

my_model = GPT2LMHeadModel.from_pretrained(saved_model_path)
print("Model loaded successfully from:", saved_model_path)
my_tokenizer = GPT2Tokenizer.from_pretrained(saved_model_path)
print("Tokenizer loaded successfully from:", saved_model_path)

prompt = "How can one live a righteous life?"           # Replace with your desired prompt
response = generate_response(my_model, my_tokenizer, prompt)
print("\nGenerated response:",response)

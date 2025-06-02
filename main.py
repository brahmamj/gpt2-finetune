import os
import re
import PyPDF2
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from clearml import Task,Dataset
from clearml import PipelineDecorator

task = Task.init(project_name="GPT2-Finetuning", task_name="GPT2-Tune", task_type=Task.TaskTypes.optimizer)
checkpoint = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.unk_token

# Function to read text from a PDF document
def read_pdf(pdf_path):
    text = ""  # Initialize an empty string to store the extracted text

    # Open the PDF file in binary read mode
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)  # Create a PdfReader object to read the file

        # Iterate over each page in the PDF
        for page_num in range(len(reader.pages)):
            if page_num > 3:  # Extract text starting from page 5 (0-indexed, so page_num > 3)
                page = reader.pages[page_num]  # Access the specific page
                text += page.extract_text()  # Extract the text from the page and append it

    return text  # Return the concatenated text from the PDF
#@PipelineDecorator.component(name="Preprocess Text",cache=False,return_values=["new_text_file"])
def preprocess_text(text_file):
    text_file = re.sub(r'\n+', '\n', text_file).strip()
    text_file = re.sub(r' +', ' ', text_file).strip()
    text_file = re.sub(r' \d+ International Gita Society', '', text_file)
    text_file = re.sub(r' Bhagavad -Gita \d+', '', text_file)
# Initialize an empty list to temporarily store words and a string to store the reformatted text
    word_list = []  # This will store words in batches of 100
    new_text_file = ''  # This will store the final reformatted text with line breaks

# Iterate through each line of the input text file
    for line in text_file.split('\n'):
        # Split the line into individual words
        words = line.split()

        # Iterate through each word in the current line
        for word in words:
            word_list.append(word)  # Add the word to the temporary list

        # Check if the word list contains 100 words
            if len(word_list) == 100:
            # Join the 100 words into a single line, add a newline character, and append to the new text
                new_text_file += ' '.join(word_list) + '\n'
            # Reset the word list for the next batch
                word_list = []

# If there are remaining words in the word list after processing all lines
    if word_list:
        # Join the remaining words and add them as the final line in the new text
        new_text_file += ' '.join(word_list) + '\n'
    return new_text_file  # Return the reformatted text with line breaks every 100 words
#@PipelineDecorator.component(name="Split Train Val Test",cache=False)
def split_train_val_test(new_text_file, train_ratio=0.8, val_ratio=0.1):
    train_fraction = 0.8
    # Define the fraction of the text to be used for training.
    # Here, 80% of the text will be used for training, and the remaining 20% will be used for validation.

    split_index = int(train_fraction * len(new_text_file))
    # Calculate the index at which to split the text.
    # Multiply the total length of `new_text_file` by the training fraction (0.8 in this case) to determine
    # the number of characters that should be included in the training set.
    # Use `int()` to ensure the result is an integer, as indexing requires integer values.

    train_text = new_text_file[:split_index]
    # Extract the training set from the start of `new_text_file` up to the calculated `split_index`.
    # This contains the first 80% of the characters in the text.

    val_text = new_text_file[split_index:]
    # Extract the validation set from `new_text_file` starting from the `split_index` to the end of the text.
    # This contains the remaining 20% of the characters in the text.
    # Save the training and validation data as text files

    with open("train.txt", "w") as f:
        f.write(train_text)

    with open("val.txt", "w") as f:
        f.write(val_text)

#@PipelineDecorator.component(name="Tokenize Function",cache=False,return_values=["tokenized_datasets"])
def tokenize_function(examples):

    block_size=256
    # Tokenize the text using the GPT-2 tokenizer and return the tokenized input in PyTorch tensor format.
    return tokenizer(examples["text"],
                     padding='max_length',        # Pad sequences to the maximum length (block_size).
                     truncation=True,             # Truncate sequences that exceed the maximum length.
                     max_length=block_size,      # Limit the tokenized sequences to `block_size` tokens.
                     return_tensors='pt')        # Return the tokenized output as PyTorch tensors.
#@PipelineDecorator.component(name="Load and Tokenize Data",cache=False,return_values=["tokenized_datasets"])
def load_and_tokenize_data():
    #checkpoint = "gpt2"
    #tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    #tokenizer.pad_token = tokenizer.unk_token
    train_file_path = 'train.txt'  # Path to the training text file
    val_file_path = 'val.txt'  # Path to the validation text file

    # Load the dataset using the Hugging Face datasets library
    dataset = load_dataset("text", data_files={"train": train_file_path,
                                           "validation": val_file_path})

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets
#@PipelineDecorator.component(name="Train Model",cache=False)
def train_model(tokenized_datasets):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    model_output_path = "content/gpt2_model"

    training_args = TrainingArguments(
    output_dir=model_output_path,              # Directory to save the model checkpoints and logs.
    overwrite_output_dir=True,                 # Allow overwriting of the output directory. Useful when running training again.
    per_device_train_batch_size=4,             # The batch size for training per device (GPU/CPU). It can be adjusted based on available memory.
    per_device_eval_batch_size=4,              # The batch size for evaluation per device (GPU/CPU).
    num_train_epochs=100,                      # The number of epochs to train the model. Here, it's set to 100, which is a large number. You may want to adjust this based on your data.
    save_steps=1_000,                          # Number of steps (iterations) after which the model will be saved.
    save_total_limit=2,                        # Limit the number of saved checkpoints. This helps manage disk space by keeping only the most recent 2 checkpoints.
    logging_dir='./logs',                      # Directory where logs will be saved during training.
    )
    trainer = Trainer(
    model=model,                                 # The model to be trained (GPT-2 in this case).
    args=training_args,                          # The training arguments, which define parameters like batch size, number of epochs, and where to save the model.
    data_collator=data_collator,                  # The data collator that handles padding, batching, and formatting of the data during training.
    train_dataset=tokenized_datasets["train"],    # The training dataset, which is the tokenized version of the training text.
    eval_dataset=tokenized_datasets["validation"],# The validation dataset, used to evaluate the model's performance during training.
    )

    trainer.train()
    saved_model_path = "content/finetuned_gpt2_model"
    trainer.save_model(saved_model_path)
    task = Task.current_task()
    task.upload_artifact(name="finetuned_gpt2_model", artifact_object=saved_model_path)
    # Save the tokenizer
    tokenizer.save_pretrained(saved_model_path)
    

#@PipelineDecorator.pipeline(name="GPT2 Fine-tuning Pipeline",project="GP2-FineTuning",version="1.0")
def main():
    # Initialize ClearML task
    #task = Task.init(project_name="GPT2 Fine-tuning", task_name="Fine-tune GPT2 on Bhagavad Gita")
    #task.set_base_task("gpt2_finetuning")

    # Create a ClearML dataset
    dataset = Dataset.get(dataset_project="GP2-FineTuning", dataset_name="Bhagavatgeeta",dataset_version="1.0.0")
    local_path = dataset.get_local_copy()
    #print(f"Dataset local copy path: {local_path}")
    text_file = read_pdf(local_path+"\document.pdf")       
    new_text_file=preprocess_text(text_file)

    # Split the preprocessed text into training and validation sets
    split_train_val_test(new_text_file)

    # Load and tokenize the dataset
    tokenized_datasets = load_and_tokenize_data()

    # Train the model
    train_model(tokenized_datasets)

if __name__ == "__main__":
    #PipelineDecorator.set_default_execution_queue("default")
    # Run the main function 
    
    # Run the ClearML pipeline
    #PipelineDecorator.run_pipeline("GPT2 Fine-tuning Pipeline")
    main()
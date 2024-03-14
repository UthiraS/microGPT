
# I am going to train Tokenizer from scartch


# 
import os
import numpy as np
import tqdm
import torch
import pdb
from torch.utils.data import Dataset,Subset
from datasets import load_dataset # huggingface datasets



# cuda tests


 
print(f"Is CUDA supported by this system? :{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
       
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


# set device for cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading tokenizer 
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


# PATH-------------------------------------------------------------------------------
# HOME_PATH = "/home/mvelmurugan/"
# MODEL_NAME = "audiodepth"

# OUT_PATH = HOME_PATH + "outputs/"
# DS_PATH = HOME_PATH+"datasets/audiospectrogram_128/data/"

# JOB_FOLDER = OUT_PATH + f"{MODEL_NAME}/{SLURM_JOB_ID}/"
TRAIN_DS_PATH = "dataset/train/"
TEST_DS_PATH = "dataset/test/"

TOKENIZED_PATH = "BPEd"


# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

pdb.set_trace()
# DATASET ---------------------------------------------------------------------------
# LOAD DATA SET
#getting openwebtext dataset from hugging face
dataset = load_dataset("openwebtext",num_proc=num_proc_load_dataset)
# SPLIT DATASET

# Define the dataset size and the split ratio
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)  # 80% for training, adjust as needed
test_size = dataset_size - train_size  # Remaining for testing

# Create indices for training and test sets
indices = list(range(dataset_size))
train_indices, test_indices = indices[:train_size], indices[train_size:]

# Create Subset datasets using SubsetRandomSampler
trainDataset = Subset(dataset, train_indices)
torch.save(trainDataset, TRAIN_DS_PATH)

testDataset = Subset(dataset, test_indices)
# torch.save(testDataset, TEST_DS_PATH)
# TOKENIZER INITIALIZE ----------------------------------------------------------------
# Initialize a tokenizer
tokenizer = Tokenizer(BPE())

# Use the Whitespace pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Initialize a trainer, you can customize its parameters as needed
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# TOKENIZER ----------------------------------------------------------------
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Assuming tran_dataset and test_dataset are lists of dictionaries with 'text' and 'labels' keys
train_dataset = Dataset.from_dict(trainDataset)
test_dataset = Dataset.from_dict(testDataset)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)


# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = trainer
model = model.to(device)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset
)

trainer.train()
trainer.evaluate(tokenized_test_dataset)



tokenizer.save_model(TOKENIZED_PATH)


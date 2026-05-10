import importlib.metadata
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

_original_version = importlib.metadata.version

def _patched_version(pkg):
    res = _original_version(pkg)
    if res is None:
        return "2.0.0"
    return res

importlib.metadata.version = _patched_version

def get_dataloaders(task="base", batch_size=16, num_samples=None):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    if task == "toxic":
        raw_datasets = load_dataset("tweet_eval", "hate")
        def format_toxic(example):
            return {"labels": [float(example["label"]), 0.0]}
        raw_datasets = raw_datasets.map(format_toxic)
        
    elif task == "spam":
        dataset_full = load_dataset("sms_spam")
        raw_datasets = dataset_full["train"].train_test_split(test_size=0.2, seed=42)
        def format_spam(example):
            return {"text": example["sms"], "labels": [0.0, float(example["label"])]}
        raw_datasets = raw_datasets.map(format_spam)
        
    elif task == "base":
        raw_datasets = load_dataset("tweet_eval", "hate")
        def format_neutral(example):
            return {"labels": [0.0, 0.0]}
        raw_datasets = raw_datasets.map(format_neutral)
    else:
        raise ValueError("Task Unknown")

    if num_samples is not None:
        train_ds = raw_datasets["train"].shuffle(seed=42).select(range(min(num_samples, len(raw_datasets["train"]))))
        test_ds = raw_datasets["test"].shuffle(seed=42).select(range(min(int(num_samples*0.2), len(raw_datasets["test"]))))
    else:
        train_ds = raw_datasets["train"].shuffle(seed=42)
        test_ds = raw_datasets["test"].shuffle(seed=42)
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, test_ds
import os
import transformers
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch

transformers.logging.set_verbosity_error()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_zero_shot_baseline(num_samples=None):
    print("\nLoading Zero-Shot LLM Pipeline (BART-Large)...")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    out_dir = "results/06_Baseline_LLM"
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "report.txt")
    
    if os.path.exists(txt_path):
        os.remove(txt_path)
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt", device=device)
    
    print("\nEvaluating Baseline on Toxicity (TweetEval)...")
    toxic_ds = load_dataset("tweet_eval", "hate", split="test").shuffle(seed=42)
    if num_samples: 
        toxic_ds = toxic_ds.select(range(min(num_samples, len(toxic_ds))))
        
    toxic_preds, toxic_labels = [], []
    for item in tqdm(toxic_ds, desc="Predicting Toxicity"):
        result = classifier(item["text"], candidate_labels=["toxic", "safe"])
        predicted_label = 1 if result["labels"][0] == "toxic" else 0
        toxic_preds.append(predicted_label)
        toxic_labels.append(item["label"])

    tox_report = classification_report(toxic_labels, toxic_preds, target_names=["Not Toxic", "Toxic"], zero_division=0)
    cm_tox = confusion_matrix(toxic_labels, toxic_preds)
    
    with open(txt_path, "a") as f:
        f.write("==================================================\n")
        f.write("LLM ZERO-SHOT BASELINE: TOXICITY\n")
        f.write("==================================================\n")
        f.write(tox_report + "\n")
        f.write(f"Confusion Matrix:\n{cm_tox}\n\n")

    print("\nEvaluating Baseline on Spam (SMS Spam Collection)...")
    spam_ds_full = load_dataset("sms_spam", split="train")
    spam_ds = spam_ds_full.train_test_split(test_size=0.2, seed=42)["test"]
    if num_samples: 
        spam_ds = spam_ds.select(range(min(num_samples, len(spam_ds))))
        
    spam_preds, spam_labels = [], []
    for item in tqdm(spam_ds, desc="Predicting Spam"):
        result = classifier(item["sms"], candidate_labels=["spam", "normal message"])
        predicted_label = 1 if result["labels"][0] == "spam" else 0
        spam_preds.append(predicted_label)
        spam_labels.append(item["label"])

    spam_report = classification_report(spam_labels, spam_preds, target_names=["Not Spam", "Spam"], zero_division=0)
    cm_spam = confusion_matrix(spam_labels, spam_preds)
    
    with open(txt_path, "a") as f:
        f.write("==================================================\n")
        f.write("LLM ZERO-SHOT BASELINE: SPAM\n")
        f.write("==================================================\n")
        f.write(spam_report + "\n")
        f.write(f"Confusion Matrix:\n{cm_spam}\n\n")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import numpy as np

def train_model(model, train_loader, epochs=3, lr=5e-5, device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
    return model

def extract_embeddings(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            all_labels.append(labels.cpu().numpy())
    return np.vstack(embeddings), np.vstack(all_labels)

def evaluate_model(model, test_loader, test_dataset, task_idx, task_name, device="cpu", output_dir=None):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs[:, task_idx].cpu().numpy())
            all_preds.extend(preds[:, task_idx].cpu().numpy())
            all_labels.extend(labels[:, task_idx].cpu().numpy())
            
    classes = ["Negative", "Positive"]
    
    print(f"\n--- {task_name} Classification Report ---")
    report_dict = classification_report(all_labels, all_preds, target_names=classes, zero_division=0, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report_str)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, "report.txt")
        
        cm_text = (
            f"Confusion Matrix (Textual):\n"
            f"               | Predicted Negative | Predicted Positive |\n"
            f"----------------------------------------------------------\n"
            f"True Negative  | {cm[0,0]:18d} | {cm[0,1]:18d} |\n"
            f"True Positive  | {cm[1,0]:18d} | {cm[1,1]:18d} |\n"
        )
        
        with open(txt_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"RESULTS FOR: {task_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Classification Report:\n{report_str}\n")
            f.write(f"{cm_text}\n")
            f.write(f"--> Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}\n")
            f.write(f"--> AUC-ROC Score:  {auc_score:.4f}\n\n")
            
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix:\n{task_name}", fontsize=12)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        safe_name = task_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace(".", "")
        cm_path = os.path.join(output_dir, f"CM_{safe_name}.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()

        if len(np.unique(all_labels)) > 1:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {task_name}')
            plt.legend(loc="lower right")
            roc_path = os.path.join(output_dir, f"ROC_{safe_name}.png")
            plt.savefig(roc_path, dpi=200)
            plt.close()

        error_csv_path = os.path.join(output_dir, f"Errors_{safe_name}.csv")
        with open(error_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "True_Label", "Predicted_Label", "Probability", "Text"])
            for i, (true_lbl, pred_lbl, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
                if true_lbl != pred_lbl:
                    writer.writerow([i, true_lbl, pred_lbl, prob, test_dataset[i]["text"]])

    return report_dict['macro avg']['f1-score'], auc_score, all_probs
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def get_base_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        problem_type="multi_label_classification",
        output_hidden_states=True
    )
    return model
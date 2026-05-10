import sys
import importlib.metadata
import os
import json
import random
import numpy as np
import transformers
transformers.logging.set_verbosity_error()

_original_version = importlib.metadata.version
def _patched_version(pkg):
    res = _original_version(pkg)
    if res is None: return "2.0.0"
    return res
importlib.metadata.version = _patched_version

import torch
import copy
from model import get_base_model
from dataset import get_dataloaders
from train import train_model, evaluate_model, extract_embeddings
from merging import merge_models, calculate_layerwise_cosine_similarity
from baseline_llm import run_zero_shot_baseline
from plot_results import generate_plots

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment():
    set_global_seed(42)
    print("Global seed initialized to 42 for strict reproducibility.\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware Allocation: {device}")
    
    dirs = {
        "base": "results/00_Base_Model",
        "mod_a": "results/01_Task_A_Toxicity",
        "mod_b": "results/02_Task_B_Spam",
        "cross": "results/03_Cross_Task_Evaluation",
        "merged_std": "results/04_Merged_Standard",
        "merged_ties": "results/05_Merged_TIES",
        "plots": "results/Plots",
        "checkpoints": "results/Checkpoints"
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        txt_path = os.path.join(d, "report.txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)

    print("\nLoading and Preparing Datasets...")
    toxic_train, toxic_test, toxic_ds = get_dataloaders("toxic")
    spam_train, spam_test, spam_ds = get_dataloaders("spam")

    print("\nInitializing Base Architecture...")
    base_model = get_base_model()
    evaluate_model(base_model, toxic_test, toxic_ds, task_idx=0, task_name="Base Toxicity", device=device, output_dir=dirs["base"])
    evaluate_model(base_model, spam_test, spam_ds, task_idx=1, task_name="Base Spam", device=device, output_dir=dirs["base"])
    
    print("\nFine-tuning Task-Specific Encoder A (Toxicity)...")
    model_a = copy.deepcopy(base_model)
    model_a = train_model(model_a, toxic_train, epochs=3, device=device)
    evaluate_model(model_a, toxic_test, toxic_ds, task_idx=0, task_name="Toxicity (Model A)", device=device, output_dir=dirs["mod_a"])

    print("\nFine-tuning Task-Specific Encoder B (Spam)...")
    model_b = copy.deepcopy(base_model)
    model_b = train_model(model_b, spam_train, epochs=3, device=device)
    evaluate_model(model_b, spam_test, spam_ds, task_idx=1, task_name="Spam (Model B)", device=device, output_dir=dirs["mod_b"])

    print("\nExecuting Cross-Task Zero-Shot Evaluation...")
    evaluate_model(model_a, spam_test, spam_ds, task_idx=1, task_name="Model A on Spam", device=device, output_dir=dirs["cross"])
    evaluate_model(model_b, toxic_test, toxic_ds, task_idx=0, task_name="Model B on Toxicity", device=device, output_dir=dirs["cross"])

    print("\nCalculating Layer-wise Cosine Similarities...")
    cos_sims = calculate_layerwise_cosine_similarity(model_a, model_b, base_model)
    with open("results/cosine_similarities.json", "w") as f:
        json.dump(cos_sims, f, indent=4)

    print("\nExtracting Latent Space Embeddings for T-SNE...")
    emb_tox, lbl_tox = extract_embeddings(model_a, toxic_test, device=device)
    emb_spam, lbl_spam = extract_embeddings(model_b, spam_test, device=device)
    np.savez("results/tsne_data.npz", 
             embeddings=np.vstack([emb_tox[:200], emb_spam[:200]]), 
             labels=np.vstack([lbl_tox[:200], lbl_spam[:200]]), 
             tasks=np.concatenate([np.zeros(200), np.ones(200)]))

    print("\nInitiating Parameter-Efficient Merging Protocol...")
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    alpha_results = []
    
    best_ties_score = 0
    best_ties_model = None

    for alpha in alphas:
        print(f"\nMerging Vector Interpolation Alpha = {alpha}")
        
        merged_std = merge_models(base_model, model_a, model_b, alpha=alpha, method="standard")
        f1_tox_std, _, _ = evaluate_model(merged_std, toxic_test, toxic_ds, task_idx=0, task_name=f"Standard Tox (a={alpha})", device=device, output_dir=dirs["merged_std"])
        f1_spam_std, _, _ = evaluate_model(merged_std, spam_test, spam_ds, task_idx=1, task_name=f"Standard Spam (a={alpha})", device=device, output_dir=dirs["merged_std"])
        
        merged_ties = merge_models(base_model, model_a, model_b, alpha=alpha, method="ties", density=0.8)
        f1_tox_ties, _, _ = evaluate_model(merged_ties, toxic_test, toxic_ds, task_idx=0, task_name=f"TIES Tox (a={alpha})", device=device, output_dir=dirs["merged_ties"])
        f1_spam_ties, _, _ = evaluate_model(merged_ties, spam_test, spam_ds, task_idx=1, task_name=f"TIES Spam (a={alpha})", device=device, output_dir=dirs["merged_ties"])
        
        current_score = f1_tox_ties + f1_spam_ties
        if current_score > best_ties_score:
            best_ties_score = current_score
            best_ties_model = copy.deepcopy(merged_ties)

        alpha_results.append({
            "alpha": alpha,
            "std_toxicity_f1": f1_tox_std,
            "std_spam_f1": f1_spam_std,
            "ties_toxicity_f1": f1_tox_ties,
            "ties_spam_f1": f1_spam_ties
        })

    with open("results/alpha_experiment_data.json", "w") as f:
        json.dump(alpha_results, f, indent=4)

    print("\nGenerating Parameter-Efficiency Report...")
    base_params = sum(p.numel() for p in base_model.parameters())
    base_mb = base_params * 4 / (1024 ** 2)
    with open("results/efficiency_report.txt", "w") as f:
        f.write("=== Parameter Efficiency Analysis ===\n")
        f.write(f"Base Architecture Parameters: {base_params:,} ({base_mb:.2f} MB)\n")
        f.write(f"Joint Training Cost (2 Models): {base_params*2:,} ({base_mb*2:.2f} MB)\n")
        f.write(f"Merged Deployment Cost: {base_params:,} ({base_mb:.2f} MB)\n")
        f.write(f"Total Resources Saved: {base_params:,} parameters ({base_mb:.2f} MB)\n")

    print("\nCheckpointing Optimal Merged Architecture...")
    torch.save(best_ties_model.state_dict(), os.path.join(dirs["checkpoints"], "optimal_merged_ties.pt"))

    print("\nFlushing Volatile Memory for Baseline Generation...")
    del model_a, model_b, merged_std, merged_ties, base_model, best_ties_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("Initiating Generative Zero-Shot Baseline...")
    run_zero_shot_baseline(num_samples=None) 

    print("\nCompiling Final Graphical Analytics...")
    generate_plots()

    print("\nEXPERIMENTAL PROTOCOL CONCLUDED SUCCESSFULLY.")

if __name__ == "__main__":
    run_experiment()
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg') 

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def generate_plots():
    print("Generating extensive analytic plot suite...")
    json_path = "results/alpha_experiment_data.json"
    
    if not os.path.exists(json_path):
        print("Data source not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    alphas = [item["alpha"] for item in data]
    std_tox = [item["std_toxicity_f1"] for item in data]
    std_spam = [item["std_spam_f1"] for item in data]
    ties_tox = [item["ties_toxicity_f1"] for item in data]
    ties_spam = [item["ties_spam_f1"] for item in data]

    std_total = [t + s for t, s in zip(std_tox, std_spam)]
    ties_total = [t + s for t, s in zip(ties_tox, ties_spam)]
    
    delta_tox = [t - s for t, s in zip(ties_tox, std_tox)]
    delta_spam = [t - s for t, s in zip(ties_spam, std_spam)]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    os.makedirs("results/Plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, std_tox, marker='o', linestyle='-', color='#d62728', linewidth=2.5, label="Toxicity (Standard)")
    plt.plot(alphas, std_spam, marker='s', linestyle='-', color='#1f77b4', linewidth=2.5, label="Spam (Standard)")
    plt.title("Method 1: Standard Task Arithmetic (Trade-off)", fontweight='bold', pad=15)
    plt.xlabel("Alpha (Interpolation Parameter)")
    plt.ylabel("Macro F1-Score")
    plt.xticks(alphas)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig("results/Plots/plot_1_standard_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, ties_tox, marker='o', linestyle='-', color='#d62728', linewidth=2.5, label="Toxicity (TIES)")
    plt.plot(alphas, ties_spam, marker='s', linestyle='-', color='#1f77b4', linewidth=2.5, label="Spam (TIES)")
    plt.title("Method 2: True TIES-Merging (Trade-off)", fontweight='bold', pad=15)
    plt.xlabel("Alpha (Interpolation Parameter)")
    plt.ylabel("Macro F1-Score")
    plt.xticks(alphas)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig("results/Plots/plot_2_ties_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(alphas, std_tox, marker='o', linestyle='--', color='#ff7f0e', linewidth=2, label="Standard")
    axes[0].plot(alphas, ties_tox, marker='D', linestyle='-', color='#2ca02c', linewidth=2.5, label="TIES")
    axes[0].set_title("Preservation: Toxicity Detection", fontweight='bold')
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_xticks(alphas)
    axes[0].legend()
    
    axes[1].plot(alphas, std_spam, marker='o', linestyle='--', color='#9467bd', linewidth=2, label="Standard")
    axes[1].plot(alphas, ties_spam, marker='D', linestyle='-', color='#8c564b', linewidth=2.5, label="TIES")
    axes[1].set_title("Preservation: Spam Detection", fontweight='bold')
    axes[1].set_xlabel("Alpha")
    axes[1].set_xticks(alphas)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/Plots/plot_3_4_direct_comparisons.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.fill_between(alphas, std_total, color="skyblue", alpha=0.4, label="Total F1 (Standard)")
    plt.plot(alphas, std_total, color="dodgerblue", marker='o', linewidth=2)
    plt.fill_between(alphas, ties_total, color="lightgreen", alpha=0.4, label="Total F1 (TIES)")
    plt.plot(alphas, ties_total, color="forestgreen", marker='s', linewidth=2)
    plt.title("Total Preserved Knowledge (Toxicity F1 + Spam F1)", fontweight='bold', pad=15)
    plt.xlabel("Alpha (Interpolation Parameter)")
    plt.ylabel("Sum of F1-Scores (Max 2.0)")
    plt.xticks(alphas)
    plt.legend(loc="lower center")
    plt.savefig("results/Plots/plot_5_total_knowledge_area.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(9, 5))
    x = np.arange(len(alphas))
    width = 0.35
    plt.bar(x - width/2, delta_tox, width, label='Delta Toxicity (TIES - Std)', color=np.where(np.array(delta_tox)>0, '#2ca02c', '#d62728'))
    plt.bar(x + width/2, delta_spam, width, label='Delta Spam (TIES - Std)', color=np.where(np.array(delta_spam)>0, '#98df8a', '#ff9896'))
    plt.axhline(0, color='black', linewidth=1)
    plt.title("TIES Improvement vs Standard Merging (+/- F1 Score)", fontweight='bold', pad=15)
    plt.xlabel("Alpha")
    plt.ylabel("F1 Score Difference")
    plt.xticks(x, alphas)
    plt.legend()
    plt.savefig("results/Plots/plot_6_delta_improvement.png", dpi=300, bbox_inches='tight')
    plt.close()

    cos_path = "results/cosine_similarities.json"
    if os.path.exists(cos_path):
        with open(cos_path, "r") as f:
            cos_data = json.load(f)
        
        raw_layers = list(cos_data.keys())
        sims = list(cos_data.values())
        
        clean_layers = []
        for l in raw_layers:
            cl = l.replace("distilbert.transformer.layer.", "Layer ")
            cl = cl.replace(".attention", " Attn")
            cl = cl.replace(".weight", "")
            cl = cl.replace("q_lin", "Q")
            cl = cl.replace("k_lin", "K")
            cl = cl.replace("v_lin", "V")
            cl = cl.replace("out_lin", "Out")
            cl = cl.replace("sa_layer_norm", "SA Norm")
            cl = cl.replace("output_layer_norm", "Out Norm")
            cl = cl.replace("ffn.lin1", "FFN 1")
            cl = cl.replace("ffn.lin2", "FFN 2")
            clean_layers.append(cl)
        
        plt.figure(figsize=(8, 14)) 
        sns.heatmap(np.array(sims).reshape(len(sims), 1), annot=True, cmap="coolwarm", 
                         yticklabels=clean_layers, xticklabels=["Cosine Similarity"],
                         cbar_kws={'label': 'Similarity Score (0 = orthogonal, 1 = identical)'})
        
        plt.title("Layer-wise Cosine Similarity: Toxicity vs Spam Vectors", fontweight='bold', pad=20)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig("results/Plots/plot_7_cosine_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    tsne_path = "results/tsne_data.npz"
    if os.path.exists(tsne_path):
        tsne_data = np.load(tsne_path)
        embeddings = tsne_data["embeddings"]
        labels = tsne_data["labels"]
        tasks = tsne_data["tasks"]

        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 7))
        
        task_names = {0.0: "Toxicity Domain (Model A)", 1.0: "Spam Domain (Model B)"}
        colors = ['#d62728', '#1f77b4'] 
        
        for i, task_id in enumerate(np.unique(tasks)):
            mask = tasks == task_id
            plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                        c=colors[i], label=task_names[task_id], 
                        alpha=0.7, edgecolors='white', s=60)
        
        plt.title("Latent Space Embeddings (T-SNE Visualization)", fontweight='bold', pad=15)
        plt.xlabel("T-SNE Dimension 1")
        plt.ylabel("T-SNE Dimension 2")
        plt.legend(fontsize=12, loc="best")
        plt.tight_layout()
        plt.savefig("results/Plots/plot_8_tsne_latent_space.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("All High-Resolution Plots Generated Successfully.")

if __name__ == "__main__":
    generate_plots()
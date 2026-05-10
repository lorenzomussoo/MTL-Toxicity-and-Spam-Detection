# Multi-Task Learning via Model Merging for Toxicity and Spam Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Sapienza University of Rome - Natural Language Processing Course (HWp)** > 📄 **Final Report:** The comprehensive scientific discussion and analysis can be found in the [`NLP_Report.pdf`](./NLP_Report.pdf) file located in the root directory. DA SISTEMAREEEEEEEE!!

## 📖 Abstract
This project addresses a core challenge in Natural Language Processing: mitigating catastrophic forgetting without the computational cost of joint-training. We explore **Task Arithmetic** (Model Merging) as a parameter-efficient solution for Multi-Task Text Classification. 

The goal is to merge the capabilities of a Transformer-based encoder (`distilbert-base-uncased`) to simultaneously resolve two semantically distant NLP tasks: Toxicity Detection and SMS Spam Detection. By fine-tuning separate models and fusing their extracted "Task Vectors" via linear interpolation, the project investigates how linguistic representations interact within the model's latent space. The merged architecture is evaluated against both the single-task encoders and a Zero-Shot Generative LLM baseline (`facebook/bart-large-mnli`). By tuning the interpolation hyperparameter ($\alpha$), the study quantifies parameter interference and finds the optimal representation trade-off when merging orthogonal linguistic domains.

## 📊 Datasets
We selected two well-established benchmarks representing different classification complexities:
1. **TweetEval (Hate Subset):** Used to evaluate Toxicity Detection, representing a complex, noisy, and highly subjective NLP task.
2. **SMS Spam Collection:** Used to evaluate Spam Detection, representing a more rigid, pattern-based classification task.

## 🧠 Methodology & Models
- **Base Architecture:** `DistilBERT` (Sanh et al., 2019), a distilled version of BERT.
- **System A & B (Task-Specific Encoders):** Independent models obtained by fine-tuning the base architecture separately on the two datasets. These serve as the specialized upper-bound performance benchmarks.
- **System B (Merged Model via Task Arithmetic):** The single parameter-efficient model resulting from the linear combination of the two task vectors ($\tau_{Toxicity} + \tau_{Spam}$). To address parameter interference, we evaluate both Standard Model Merging (Ilharco et al., 2022) and the **TIES-Merging** technique (Yadav et al., 2023).
- **Baseline System (Zero-Shot Generative LLM):** An out-of-the-box `bart-large-mnli` model used in Zero-Shot mode (Yin et al., 2019) to compare specialized representations against broad parametric knowledge.

## 📂 Repository Structure
The codebase is modular and automatically generates a structured analytics directory upon execution:

```text
.
├── NLP_Report.pdf                  # Final academic report
├── README.md                       
├── src/                            # Source Code
│   ├── dataset.py                  # Dataloaders and HuggingFace dataset processing
│   ├── model.py                    # Base architecture initialization
│   ├── train.py                    # Fine-tuning, evaluation, and latent space extraction
│   ├── merging.py                  # Task Arithmetic, TIES-Merging, and Cosine Similarities
│   ├── baseline_llm.py             # Zero-shot generative LLM evaluation
│   ├── plot_results.py             # High-resolution academic plotting suite
│   └── main.py                     # Orchestrator script
│
└── results/                        # Automatically generated output directory
    ├── 00_Base_Model/              # Untrained baseline metrics
    ├── 01_Task_A_Toxicity/         # Fine-tuned System A
    ├── 02_Task_B_Spam/             # Fine-tuned System B
    ├── 03_Cross_Task_Evaluation/   # Zero-shot cross-domain testing
    ├── 04_Merged_Standard/         # Results for Standard Task Arithmetic (alphas 0.0 to 1.0)
    ├── 05_Merged_TIES/             # Results for TIES-Merging (alphas 0.0 to 1.0)
    ├── 06_Baseline_LLM/            # Generative BART-Large evaluation
    ├── Plots/                      # 8 high-res charts (Trade-offs, ROCs, Heatmaps, T-SNE)
    └── [Raw Data Files]            # efficiency_report.txt, tsne_data.npz, .json analytics

(Note: Each subfolder in results/ contains detailed report.txt files, Confusion Matrices, ROC-AUC curves, and Error Analysis CSVs tracking specific misclassifications).
```

🚀 Execution & Reproducibility
Global random seeds (seed=42) are strictly enforced for 100% reproducibility.

1. Install dependencies:
```text
pip install torch transformers datasets scikit-learn matplotlib seaborn
```

2. Run the full experimental pipeline:
```text
python src/main.py
```

3. Regenerate plots (Optional):
If raw data files exist in results/, you can regenerate the plot suite without re-training:
```text
python src/plot_results.py
```

👥 Authors
•	[Lorenzo Musso]
•	[Giulia Pietrangeli]

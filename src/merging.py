import torch
import copy
from model import get_base_model

def state_dict_sub(model_a, model_b):
    task_vector = {}
    for key in model_a.keys():
        task_vector[key] = model_a[key].cpu() - model_b[key].cpu()
    return task_vector

def trim_task_vector(vector, density=0.8):
    trimmed_vector = {}
    for k, v in vector.items():
        if v.numel() == 0:
            trimmed_vector[k] = v
            continue
        k_val = int(v.numel() * density)
        if k_val == 0:
            trimmed_vector[k] = torch.zeros_like(v)
            continue
        abs_v = torch.abs(v)
        threshold = torch.kthvalue(abs_v.flatten(), v.numel() - k_val + 1).values
        mask = abs_v >= threshold
        trimmed_vector[k] = v * mask
    return trimmed_vector

def apply_sign_consensus(tau_a, tau_b):
    consensus_a = {}
    consensus_b = {}
    for key in tau_a.keys():
        sum_vec = tau_a[key] + tau_b[key]
        dominant_sign = torch.sign(sum_vec)
        mask_a = (torch.sign(tau_a[key]) == dominant_sign)
        mask_b = (torch.sign(tau_b[key]) == dominant_sign)
        consensus_a[key] = tau_a[key] * mask_a
        consensus_b[key] = tau_b[key] * mask_b
    return consensus_a, consensus_b

def state_dict_add(base, vector_list, alpha=1.0):
    merged = {}
    for key in base.keys():
        merged[key] = base[key].cpu() + alpha * sum(vec[key].cpu() for vec in vector_list)
    return merged

def merge_models(base_model, model_a, model_b, alpha=0.5, method="standard", density=0.8):
    base_sd = base_model.state_dict()
    a_sd = model_a.state_dict()
    b_sd = model_b.state_dict()
    
    tau_a = state_dict_sub(a_sd, base_sd)
    tau_b = state_dict_sub(b_sd, base_sd)
    
    if method == "ties":
        tau_a = trim_task_vector(tau_a, density=density)
        tau_b = trim_task_vector(tau_b, density=density)
        tau_a, tau_b = apply_sign_consensus(tau_a, tau_b)
    
    merged_sd = state_dict_add(base_sd, [tau_a, tau_b], alpha=alpha)
    
    merged_model = get_base_model()
    merged_model.load_state_dict(merged_sd)
    return merged_model

def calculate_layerwise_cosine_similarity(model_a, model_b, base_model):
    base_sd = base_model.state_dict()
    tau_a = state_dict_sub(model_a.state_dict(), base_sd)
    tau_b = state_dict_sub(model_b.state_dict(), base_sd)
    
    similarities = {}
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for key in tau_a.keys():
        if "layer" in key and "weight" in key:
            vec_a = tau_a[key].flatten()
            vec_b = tau_b[key].flatten()
            if vec_a.numel() > 0:
                sim = cos(vec_a, vec_b).item()
                similarities[key] = sim
    return similarities
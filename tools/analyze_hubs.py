import os
import json
import torch
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import RAW_CO_MATRIX_PATH, HUB_JSON_PATH

NUM_LAYERS = 24

def compute_hub_scores(co_matrix_layer):
    freq = torch.diag(co_matrix_layer).float()
    max_freq = freq.max() + 1e-9
    norm_freq = freq / max_freq  
    off_diag = co_matrix_layer.clone()
    off_diag.fill_diagonal_(0)
    row_sums = off_diag.sum(dim=1, keepdim=True) + 1e-9
    p_j_given_i = off_diag / row_sums
    entropy = -torch.sum(p_j_given_i * torch.log(p_j_given_i + 1e-9), dim=1)
    return norm_freq * torch.exp(entropy)

def main():
    print("🚀 开始计算全量专家的数学放大分数 (Amplified Hub Score)...")
    co_matrices = torch.load(RAW_CO_MATRIX_PATH, weights_only=True) 
    all_layers_hubs = {}  

    for l in range(NUM_LAYERS):
        scores = compute_hub_scores(co_matrices[l]).numpy()
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        
        all_layers_hubs[f"layer_{l}"] = {
            "hub_experts": sorted_indices.tolist(),
            "hub_scores": sorted_scores.tolist()
        }
        
    HUB_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HUB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_layers_hubs, f, indent=4)

    print(f"🎉 提取完成！全量 60 个专家的排行榜已安全保存至: {HUB_JSON_PATH}")

if __name__ == "__main__":
    main()
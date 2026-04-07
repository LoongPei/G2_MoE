import os
import json
import torch
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 🌟 引入我们新写的 Solver 类
from g2moe.solver.gurobi_solver import G2MoEPlacementSolver
from g2moe.config import (
    RAW_CO_MATRIX_PATH, PMI_MATRIX_PATH, MARKOV_MATRIX_PATH, 
    HUB_JSON_PATH, PLACEMENT_MAP_PATH
)

NUM_GPUS = 4
NUM_LAYERS = 24
NUM_EXPERTS = 60

def profile_p2p_latency(num_gpus, tensor_size_mb=64, num_warmup=3, num_iters=10):
    print(f"🔍 正在启动物理级 PCIe/NVLink 探针 (测试负载: {tensor_size_mb}MB)...")
    num_elements = (tensor_size_mb * 1024 * 1024) // 2 
    latency_matrix = np.zeros((num_gpus, num_gpus))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src == dst: continue
            try:
                data = torch.randn(num_elements, dtype=torch.bfloat16, device=f"cuda:{src}")
                for _ in range(num_warmup): _ = data.to(f"cuda:{dst}")
                torch.cuda.synchronize()
                
                start_event.record()
                for _ in range(num_iters): _ = data.to(f"cuda:{dst}")
                end_event.record()
                torch.cuda.synchronize()
                
                latency_matrix[src, dst] = start_event.elapsed_time(end_event) / num_iters
            except Exception:
                latency_matrix[src, dst] = 999.9 

    mean_lat = np.mean(latency_matrix[latency_matrix > 0])
    return latency_matrix / mean_lat

def main():
    # 1. 探针测速
    D = profile_p2p_latency(NUM_GPUS)
    
    # 2. 加载底层矩阵数据
    print("\n🚀 正在加载底层特征矩阵并计算信号级动态校准...")
    co_matrices = torch.load(RAW_CO_MATRIX_PATH, weights_only=True).numpy()
    pmi_matrices = torch.load(PMI_MATRIX_PATH, weights_only=True).numpy()
    markov_matrices = torch.load(MARKOV_MATRIX_PATH, weights_only=True).numpy()

    DYNAMIC_LAMBDA_INTER = np.mean(np.abs(pmi_matrices)) / (np.mean(np.abs(markov_matrices)) + 1e-9)

    with open(HUB_JSON_PATH, "r", encoding="utf-8") as f:
        hub_data = json.load(f)

    # 3. 预处理出专门给 Solver 使用的结构
    specialized_experts, expert_loads = {}, {}
    for l in range(NUM_LAYERS):
        hubs = hub_data[f"layer_{l}"]["hub_experts"][:4]
        specs = [i for i in range(NUM_EXPERTS) if i not in hubs]
        specialized_experts[l] = specs
        expert_loads[l] = {i: co_matrices[l][i, i] for i in specs}

    # 🌟 4. 优雅地实例化 Solver 并求解
    solver = G2MoEPlacementSolver(
        num_gpus=NUM_GPUS, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS
    )
    
    placement_map = solver.solve(
        D=D, 
        pmi_matrices=pmi_matrices, 
        markov_matrices=markov_matrices, 
        hub_data=hub_data, 
        specialized_experts=specialized_experts, 
        expert_loads=expert_loads, 
        dynamic_lambda=DYNAMIC_LAMBDA_INTER
    )

    # 5. 落盘
    if placement_map:
        PLACEMENT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PLACEMENT_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(placement_map, f, indent=4)
        print(f"💾 最优调度表已覆盖至: {PLACEMENT_MAP_PATH}")

if __name__ == "__main__":
    main()
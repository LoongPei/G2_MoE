import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import DEFAULT_MODEL_PATH, MATRIX_DIR, CACHE_DIR

# 自然分布数据集路径
DATA_JSON_PATH = Path(CACHE_DIR) / "The_Pile_g2moe" / "moe_natural_subset.json"

SEQ_MAX_LEN = 1024
BATCH_SIZE = 4
NUM_LAYERS = 24
NUM_EXPERTS = 60
TOP_K = 4

def main():
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 [1/4] 正在加载 Tokenizer 并进行自然分布序列分块...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        natural_data = json.load(f)

    all_input_ids = []
    for text in tqdm(natural_data, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_input_ids.extend(tokens)

    num_chunks = len(all_input_ids) // SEQ_MAX_LEN
    input_chunks = [all_input_ids[i * SEQ_MAX_LEN : (i + 1) * SEQ_MAX_LEN] for i in range(num_chunks)]
    actual_total_tokens = num_chunks * SEQ_MAX_LEN

    print(f"\n🚀 [2/4] 正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    intra_co_occurrence = torch.zeros((NUM_LAYERS, NUM_EXPERTS, NUM_EXPERTS), dtype=torch.float32)
    inter_transition = torch.zeros((NUM_LAYERS - 1, NUM_EXPERTS, NUM_EXPERTS), dtype=torch.float32)

    print("\n🚀 [3/4] 开始前向传播与路由统计...")
    with torch.no_grad():
        for i in tqdm(range(0, len(input_chunks), BATCH_SIZE), desc="Inference"):
            batch_ids = input_chunks[i : i + BATCH_SIZE]
            inputs = torch.tensor(batch_ids).to(model.device)
            outputs = model(inputs, output_router_logits=True)
            router_logits = outputs.router_logits
            
            layer_multi_hots = []
            for l in range(NUM_LAYERS):
                logits = router_logits[l].float().cpu()
                _, topk_indices = torch.topk(logits, TOP_K, dim=-1)
                multi_hot = torch.zeros((topk_indices.shape[0], NUM_EXPERTS), dtype=torch.float32)
                multi_hot.scatter_(1, topk_indices, 1.0)
                layer_multi_hots.append(multi_hot)
                
                intra_co_occurrence[l] += torch.matmul(multi_hot.T, multi_hot)
                
            for l in range(NUM_LAYERS - 1):
                inter_transition[l] += torch.matmul(layer_multi_hots[l].T, layer_multi_hots[l+1])

    print("\n🚀 [4/4] 正在计算归一化矩阵 (PMI & Markov)...")
    intra_pmi = torch.zeros_like(intra_co_occurrence)
    for l in range(NUM_LAYERS):
        co_matrix = intra_co_occurrence[l]
        expert_counts = torch.diag(co_matrix)
        P_i = (expert_counts / actual_total_tokens).unsqueeze(1) + 1e-10
        P_j = (expert_counts / actual_total_tokens).unsqueeze(0) + 1e-10
        P_ij = co_matrix / actual_total_tokens
        pmi = torch.log2(P_ij / (P_i * P_j) + 1e-10)
        pmi.fill_diagonal_(0)
        intra_pmi[l] = torch.clamp(pmi, min=0.0)

    inter_markov = torch.zeros_like(inter_transition)
    for l in range(NUM_LAYERS - 1):
        trans_matrix = inter_transition[l]
        row_sums = trans_matrix.sum(dim=1, keepdim=True) + 1e-10
        inter_markov[l] = trans_matrix / row_sums

    torch.save(intra_pmi, MATRIX_DIR / "intra_pmi_matrix.pt")
    torch.save(inter_markov, MATRIX_DIR / "inter_markov_matrix.pt")
    torch.save(intra_co_occurrence, MATRIX_DIR / "raw_co_occurrence.pt") 
    print(f"\n🎉 矩阵已保存至 {MATRIX_DIR}")

if __name__ == "__main__":
    main()
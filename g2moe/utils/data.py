import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from g2moe.config import CACHE_DIR

def get_domain_tokens(tokenizer, domain, total_required_tokens):
    """
    统一的离线本地数据加载器
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"📚 正在从本地极速加载 {total_required_tokens} 个 [{domain.upper()}] Tokens...")
        
    tokens_list = []
    
    if domain == "wiki":
        local_data_paths = [os.path.join(CACHE_DIR, "wikitext", "wikitext-103-raw-v1", f"train-0000{i}-of-00002.parquet") for i in range(2)]
        dataset = load_dataset("parquet", data_files=local_data_paths, split="train")
        for row in dataset:
            if not row["text"].strip(): continue
            toks = tokenizer(row["text"], return_tensors="pt").input_ids.squeeze(0)
            tokens_list.append(toks)
            if sum(len(t) for t in tokens_list) >= total_required_tokens: break
            
    elif domain == "code":
        local_path = os.path.join(CACHE_DIR, "theblackcat102", "evol-codealpaca-v1", "train.jsonl")
        dataset = load_dataset("json", data_files=local_path, split="train")
        for row in dataset:
            text = row["instruction"] + "\n" + row["output"]
            toks = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
            tokens_list.append(toks)
            if sum(len(t) for t in tokens_list) >= total_required_tokens: break
            
    elif domain == "math":
        local_path = os.path.join(CACHE_DIR, "meta-math", "MetaMathQA", "MetaMathQA-395K.json")
        dataset = load_dataset("json", data_files=local_path, split="train")
        for row in dataset:
            text = row["query"] + "\n" + row["response"]
            toks = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
            tokens_list.append(toks)
            if sum(len(t) for t in tokens_list) >= total_required_tokens: break

    return torch.cat(tokens_list)[:total_required_tokens]
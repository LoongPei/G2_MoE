import os
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

import sys
from pathlib import Path
# ==========================================
# 动态解析项目根目录并加入环境变量
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import DEFAULT_MODEL_PATH, EVAL_RESULTS_DIR
from g2moe.core.factory import build_g2moe_model
from g2moe.utils.data import get_domain_tokens

def run_micro_traffic_test(model, tokenizer, args, rank, world_size):
    device = f"cuda:{rank}"
    total_tokens_per_step = args.batch_size * args.seq_len
    total_required_tokens = args.steps * total_tokens_per_step
    chunk_size = (args.seq_len * args.batch_size) // world_size  
    
    if rank == 0:
        print(f"\n🚀 [Mode={args.mode} | Domain={args.domain} | Hubs={args.hub_size} | TopK={args.top_k} | BS={args.batch_size} | Seq={args.seq_len}]")

    tokens = get_domain_tokens(tokenizer, args.domain, total_required_tokens)
    all_steps_traffic = []
    model.eval()
    
    oom_flag = False
    
    for step in range(args.steps):
        step_start = step * total_tokens_per_step
        local_start = step_start + rank * chunk_size
        local_input_ids = tokens[local_start : local_start + chunk_size].view(args.batch_size, -1).to(device)
        
        # 🌟 OOM 全局熔断信号灯
        oom_detected = torch.tensor([0], dtype=torch.int, device=device)
        
        try:
            with torch.no_grad(): 
                _ = model(local_input_ids)
                
            layer_traffic = [getattr(layer.mlp, 'current_comm_bytes', 0) for layer in model.model.layers]
            layer_traffic_tensor = torch.tensor(layer_traffic, dtype=torch.long, device=device)
            
            gather_list = [torch.zeros_like(layer_traffic_tensor) for _ in range(world_size)] if rank == 0 else None
            dist.gather(layer_traffic_tensor, gather_list, dst=0)
            
            if rank == 0:
                all_steps_traffic.append(torch.stack(gather_list, dim=1).tolist())
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_detected[0] = 1
                torch.cuda.empty_cache()
            else:
                raise e
                
        # 🌟 全局同步 OOM 状态
        dist.all_reduce(oom_detected, op=dist.ReduceOp.MAX)
        if oom_detected.item() > 0:
            oom_flag = True
            if rank == 0:
                print(f"⚠️ 触发显存极限 (OOM)! 自动熔断跳过当前参数组合...")
            torch.cuda.empty_cache()
            break

    # 落盘保存 (依赖 config.py 中的全局路径)
    if rank == 0:
        output_dir = EVAL_RESULTS_DIR / "comprehensive_ablation_1"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        out_name = f"{args.mode}_{args.domain}_H{args.hub_size}_K{args.top_k}_B{args.batch_size}_S{args.seq_len}.json"
        
        if oom_flag:
            output_data = {"status": "OOM", "mode": args.mode, "config": vars(args)}
        else:
            arr_mb = np.array(all_steps_traffic) / (1024 * 1024)
            step_total_traffic_mb = arr_mb.sum(axis=(1, 2)).tolist()
            print(f"📉 平均跨卡流量: {np.mean(step_total_traffic_mb):.2f} MB")
            output_data = {
                "status": "SUCCESS",
                "mode": args.mode,
                "domain": args.domain,
                "config": vars(args),
                "metrics": {"step_total_traffic_mb": step_total_traffic_mb}
            }
            
        with open(output_dir / out_name, "w") as f:
            json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["standard_ep", "ep_hub", "ep_topo", "ours"])
    parser.add_argument("--domain", type=str, default="wiki", choices=["wiki", "code", "math"])
    parser.add_argument("--hub_size", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    
    # 🚀 直接调用统一工厂构建带有探针的模型
    model = build_g2moe_model(
        model_path=DEFAULT_MODEL_PATH, 
        mode=args.mode, 
        rank=rank, 
        world_size=world_size, 
        target_hub_size=args.hub_size, 
        target_top_k=args.top_k
    )
    
    run_micro_traffic_test(model, tokenizer, args, rank, world_size)
    dist.destroy_process_group()
import os
import time
import json
import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 动态解析项目根目录并加入环境变量
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import DEFAULT_MODEL_PATH, OUTPUT_DIR
from g2moe.core.factory import build_g2moe_model

def run_benchmark(model, input_ids, mode, steps=10, warmup=3, exp_name="batchsize"):
    is_main_process = (mode == "baseline") or (dist.get_rank() == 0)
    
    if is_main_process:
        print(f"\n🔥 开始 {mode.upper()} 吞吐量压测 (Warmup: {warmup}, Steps: {steps})...")
        
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
            torch.cuda.synchronize()
            if mode != "baseline": dist.barrier()

    if mode != "baseline": dist.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(steps):
            _ = model(input_ids)
            
    torch.cuda.synchronize()
    if mode != "baseline": dist.barrier()
    total_time = time.time() - start_time
    
    global_batch = input_ids.shape[0] if mode == "baseline" else input_ids.shape[0] * dist.get_world_size()
    seq_len = input_ids.shape[1]
    throughput = (global_batch * seq_len * steps) / total_time
    
    if is_main_process:
        comm_info = ""
        # 从第一层的 Wrapper 中提取通信探针数据
        first_layer_mlp = getattr(model.model.layers[0], 'mlp', None)
        if mode != "baseline" and hasattr(first_layer_mlp, 'benchmark_comm_bytes'):
            total_mb = sum([layer.mlp.benchmark_comm_bytes for layer in model.model.layers]) / (1024*1024)
            bypassed = sum([layer.mlp.benchmark_tokens_bypassed for layer in model.model.layers])
            sent = sum([layer.mlp.benchmark_tokens_sent for layer in model.model.layers])
            reduction_rate = (bypassed / (bypassed + sent) * 100) if (bypassed + sent) > 0 else 0
            
            comm_info = (
                f"  ➤ 实际网络传输流量: {total_mb:.2f} MB\n"
                f"  ➤ Hub Bypass 拦截 Token 数: {bypassed}\n"
                f"  ➤ 跨卡发送 Token 数: {sent}\n"
                f"  ➤ 物理通信削减率: {reduction_rate:.2f} %\n"
            )

        report = (
            f"=====================================\n"
            f"🏆 {mode.upper()} 性能战报\n"
            f"  ➤ 全局 Batch Size: {global_batch}\n"
            f"  ➤ Sequence Length: {seq_len}\n"
            f"  ➤ 吞吐量: {throughput:.2f} Tokens/sec\n"
            f"{comm_info}"
            f"=====================================\n"
        )
        print(report)
        
        # 🚀 亮点 3：直接存入 config.py 定义好的全局 Outputs 目录下
        out_dir = OUTPUT_DIR / "throughput" / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = f"{mode}_batch{global_batch}_seq{seq_len}"
        
        with open(out_dir / f"{file_prefix}_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            
        json_data = {
            "mode": mode, "global_batch": global_batch, "seq_len": seq_len,
            "steps": steps, "total_time_sec": round(total_time, 4), "throughput_tokens_per_sec": round(throughput, 2)
        }
        if comm_info:
            json_data.update({
                "total_traffic_mb": round(total_mb, 2),
                "tokens_bypassed": bypassed,
                "tokens_sent": sent,
                "traffic_reduction_rate": round(reduction_rate, 2)
            })
            
        with open(out_dir / f"{file_prefix}_data.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "standard_ep", "ours"], required=True)
    parser.add_argument("--global_batch", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--exp_name", type=str, default="batchsize")
    args = parser.parse_args()

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    if args.mode == "baseline":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.eval()
        input_ids = torch.randint(0, model.config.vocab_size, (args.global_batch, args.seq_len), device="cuda:0")
        run_benchmark(model, input_ids, "baseline", exp_name=args.exp_name)
    else:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 🚀 亮点 4：一行代码完成模型切片和组装！
        model = build_g2moe_model(DEFAULT_MODEL_PATH, mode=args.mode, rank=rank, world_size=world_size)
        model.eval()
        
        local_batch = args.global_batch // world_size
        input_ids = torch.randint(0, model.config.vocab_size, (local_batch, args.seq_len), device=f"cuda:{rank}")
        run_benchmark(model, input_ids, args.mode, exp_name=args.exp_name)
        dist.destroy_process_group()
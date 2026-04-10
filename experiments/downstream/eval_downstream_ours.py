import os
import json
import sys
from pathlib import Path

# 动态解析项目根目录并加入环境变量
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table 

from g2moe.config import DEFAULT_MODEL_PATH, EVAL_RESULTS_DIR, HF_DATASETS_CACHE
from g2moe.core.factory import build_g2moe_model

# 可以选择强制离线评测与指定本地离线缓存库，也可在线下载
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 默认配置
TASKS = ["race", "sciq", "rte", "boolq", "copa", "wsc", "piqa"]
# TASKS = ["rte", "sciq"]  # 按需修改
# TASKS = ["piqa"] 
BATCH_SIZE = 2
TEST_LIMIT = None  

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 1. 🚀 直接调用统一工厂，一键完成 DDP+EP 切片和模型构建
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    model = build_g2moe_model(
        model_path=DEFAULT_MODEL_PATH,
        mode="ours",
        rank=rank,
        world_size=world_size
    )
    model.eval()
    
    if rank == 0:
        print("\n🔬 正在挂载至 LM-Evaluation-Harness 测试台...")
        print(f"\n🚀 开始跑分: 任务 {TASKS}")
        
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)
    
    # 2. 开始评测
    results = lm_eval.simple_evaluate(
        model=lm_eval_model, 
        tasks=TASKS, 
        limit=TEST_LIMIT, 
        num_fewshot=0,
        log_samples=True
    )
    
    # 3. 战报保存
    if rank == 0 and results is not None:
        output_dir = EVAL_RESULTS_DIR / "downstream" / "ours"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        table_str = make_table(results)
        task_suffix = "_".join(TASKS)
        
        info_str = (
            f"\n==================================================\n"
            f"🏆 Ours (DDP+EP) 下游任务准确率报告\n"
            f"  ➤ 测试任务: {TASKS}\n"
            f"==================================================\n"
        )
        print(info_str + table_str)
        
        with open(output_dir / f"metrics_{task_suffix}.txt", "w", encoding="utf-8") as f:
            f.write(table_str + info_str)
        
        with open(output_dir / f"results_{task_suffix}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"💾 结果已保存至: {output_dir}/")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
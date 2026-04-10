import os
import json
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table 
from g2moe.config import DEFAULT_MODEL_PATH, EVAL_RESULTS_DIR, HF_DATASETS_CACHE

# 严格对齐离线环境变量
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

TASKS = ["race", "sciq", "rte", "boolq", "copa", "wsc", "piqa"]
# TASKS = ["sciq", "rte"] # 按需修改
# TASKS = ["race", "boolq"]
# TASKS = ["piqa"]
BATCH_SIZE = 2
TEST_LIMIT = None 

def main():
    print("🛑 启动 Baseline 下游任务评估 (HuggingFace 原生流水线)")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True
    )
    model.eval()
    
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=BATCH_SIZE)
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=TASKS, limit=TEST_LIMIT, num_fewshot=0)
    
    if results is not None:
        output_dir = EVAL_RESULTS_DIR / "downstream" / "baseline"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        table_str = make_table(results) 
        task_suffix = "_".join(TASKS)
        info_str = f"\n🏆 Baseline 下游任务准确率报告\n  ➤ ��试任务: {TASKS}\n"
        print(info_str + table_str)
        
        with open(output_dir / f"metrics_{task_suffix}.txt", "w", encoding="utf-8") as f:
            f.write(table_str + info_str)
        with open(output_dir / f"results_{task_suffix}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 结果已保存至: {output_dir}/")

if __name__ == "__main__":
    main()
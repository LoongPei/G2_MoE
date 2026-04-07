import os
import json
import sys
from pathlib import Path
from datasets import load_dataset

# 动态解析项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import CACHE_DIR

def main():
    CUSTOM_CACHE_DIR = Path(CACHE_DIR) / "The_Pile_g2moe"
    CUSTOM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CUSTOM_CACHE_DIR / "moe_natural_subset.json"

    TOTAL_SAMPLES = 5000  # 直接取 5000 篇

    print("🚀 开始流式抽取 The Pile 真实自然分布的切片...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    natural_data = []
    for idx, item in enumerate(dataset):
        natural_data.append(item["text"])
        if (idx + 1) % 500 == 0:
            print(f"✅ 已收集 {idx + 1}/{TOTAL_SAMPLES} 篇")
        if len(natural_data) >= TOTAL_SAMPLES:
            break

    print(f"正在写入本地文件: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(natural_data, f, ensure_ascii=False, indent=2)

    print("🎉 真实通用负载数据集已生成！去跑探针吧！")

if __name__ == "__main__":
    main()
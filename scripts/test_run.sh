#!/bin/bash

export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH

echo "🛠️ 开始轻量级快速冒烟测试 (Smoke Test)..."

# 🌟 极小参数：仅跑 3 步，序列长度缩减到 256，瞬间就能跑完
STEPS=3
SEQ=256
BS=1

echo "▶️ 测试 1: 基线模式 (Standard EP)"
torchrun --nproc_per_node=4 experiments/ablation/comprehensive_ablation.py \
    --mode standard_ep --domain wiki \
    --hub_size 4 --top_k 4 --batch_size $BS --seq_len $SEQ --steps $STEPS

echo "▶️ 测试 2: 完全体模式 (Ours) + 动态 Hub 测试 (Hub=8)"
torchrun --nproc_per_node=4 experiments/ablation/comprehensive_ablation.py \
    --mode ours --domain wiki \
    --hub_size 8 --top_k 2 --batch_size $BS --seq_len $SEQ --steps $STEPS

echo "✅ 冒烟测试执行完毕！"
echo "🔍 请检查 outputs/eval_results/comprehensive_ablation/ 目录下是否成功生成了对应的 json 文件。"
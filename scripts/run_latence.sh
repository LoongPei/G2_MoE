#!/bin/bash

# 发生错误时停止脚本
set -e

# 设置环境变量，与 Python 脚本内一致（如果脚本里已经写了也可以在这里再次确认）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 定义使用的 GPU 数量 (请根据你的实际情况修改)
NUM_GPUS=4
SCRIPT_NAME="experiments/throughput/latency_breakdown.py"

echo "=========================================================="
echo "🚀 开始实验一：测试不同 SeqLen (exp_name=seqlen)"
echo "固定 Global Batch = 4 | SeqLen = [1024, 2048, 4096, 8192]"
echo "=========================================================="

echo ">>> [SeqLen 测试] 运行 ours 模式..."
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_NAME \
    --mode ours \
    --seq_lens 1024 2048 4096 8192 \
    --global_batches 4 \
    --exp_name seqlen

echo ">>> [SeqLen 测试] 运行 standard_ep 模式..."
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_NAME \
    --mode standard_ep \
    --seq_lens 1024 2048 4096 8192 \
    --global_batches 4 \
    --exp_name seqlen


echo "=========================================================="
echo "🚀 开始实验二：测试不同 BatchSize (exp_name=batchsize)"
echo "固定 SeqLen = 1024 | Global Batch = [4, 8, 16, 32]"
echo "=========================================================="

echo ">>> [BatchSize 测试] 运行 ours 模式..."
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_NAME \
    --mode ours \
    --seq_lens 1024 \
    --global_batches 4 8 16 32 \
    --exp_name batchsize

echo ">>> [BatchSize 测试] 运行 standard_ep 模式..."
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_NAME \
    --mode standard_ep \
    --seq_lens 1024 \
    --global_batches 4 8 16 32 \
    --exp_name batchsize

echo "🎉 所有测试运行完毕！"
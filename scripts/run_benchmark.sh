#!/bin/bash

# 设置遇到错误时继续执行，避免因为某一组参数 OOM 导致整个测试停止
set +e 

SCRIPT="experiments/throughput/throughput_benchmark.py"
MODES=("ours" "standard_ep" "baseline")

echo "================================================="
echo "          开始吞吐量压测自动化脚本"
echo "================================================="

# ---------------------------------------------------------
# 测试一：seq_len 测试
# 固定 global_batch=4，遍历 seq_len 和 mode
# ---------------------------------------------------------
# EXP_NAME_SEQLEN="seqlen"
# GLOBAL_BATCH_FIXED=4
# SEQ_LENS=(1024)

# echo -e "\n>>> [测试一] 开始 seq_len 测试 (exp_name=${EXP_NAME_SEQLEN}, global_batch=${GLOBAL_BATCH_FIXED})"

# for mode in "${MODES[@]}"; do
#     for seq_len in "${SEQ_LENS[@]}"; do
#         echo "-------------------------------------------------"
#         echo "🏃 Running -> Mode: ${mode} | Batch: ${GLOBAL_BATCH_FIXED} | SeqLen: ${seq_len}"
        
#         if [ "$mode" == "baseline" ]; then
#             python ${SCRIPT} \
#                 --mode ${mode} \
#                 --global_batch ${GLOBAL_BATCH_FIXED} \
#                 --seq_len ${seq_len} \
#                 --exp_name ${EXP_NAME_SEQLEN}
#         else
#             torchrun --nproc_per_node=4 ${SCRIPT} \
#                 --mode ${mode} \
#                 --global_batch ${GLOBAL_BATCH_FIXED} \
#                 --seq_len ${seq_len} \
#                 --exp_name ${EXP_NAME_SEQLEN}
#         fi
        
#         # 每次运行结束后稍微休眠一下，给 GPU 显��释放留出时间
#         sleep 2 
#     done
# done


# ---------------------------------------------------------
# 测试二：batchsize 测试
# 固定 seq_len=2048，遍历 global_batch 和 mode
# ---------------------------------------------------------
EXP_NAME_BATCH="batchsize"
SEQ_LEN_FIXED=1024
GLOBAL_BATCHES=(4 8 16 32)

echo -e "\n>>> [测试二] 开始 batchsize 测试 (exp_name=${EXP_NAME_BATCH}, seq_len=${SEQ_LEN_FIXED})"

for mode in "${MODES[@]}"; do
    for batch in "${GLOBAL_BATCHES[@]}"; do
        echo "-------------------------------------------------"
        echo "🏃 Running -> Mode: ${mode} | Batch: ${batch} | SeqLen: ${SEQ_LEN_FIXED}"
        
        if [ "$mode" == "baseline" ]; then
            python ${SCRIPT} \
                --mode ${mode} \
                --global_batch ${batch} \
                --seq_len ${SEQ_LEN_FIXED} \
                --exp_name ${EXP_NAME_BATCH}
        else
            torchrun --nproc_per_node=4 ${SCRIPT} \
                --mode ${mode} \
                --global_batch ${batch} \
                --seq_len ${SEQ_LEN_FIXED} \
                --exp_name ${EXP_NAME_BATCH}
        fi
        
        # 每次运行结束后稍微休眠一下，给 GPU 显存释放留出时间
        sleep 2
    done
done

echo "================================================="
echo "          🎉 所有压测任务执行完毕！"
echo "================================================="
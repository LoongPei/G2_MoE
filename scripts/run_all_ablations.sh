#!/bin/bash

# ==========================================
# OSDI/MLSys 级全自动消融实验控制中枢
# ==========================================
# 你可以在这里随意调整循环的粒度！
export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH

# 基础参数
STEPS=15
BASE_SEQ=8192
BASE_BS=1
BASE_TOPK=4
BASE_HUB=4

# 实验控制变量列表
DOMAINS=("wiki" "code" "math")
HUB_SIZES=(0 4 8 12 16)      # 要求: 60 - size 必须能被 4 整除
BATCH_SIZES=(1 2 4 8)        # 收窄了跨度，增加粒度
SEQ_LENS=(2048 4096 8192)    # 长度梯度
TOP_KS=(2 4 8)

echo "🚀 [系统启动] 开始全自动 OSDI/MLSys 消融测试群..."

# ------------------------------------------
# 战役 1：核心解构 (4 Mode × 3 Domain)
# ------------------------------------------
echo "========== 战役 1: 核心解构 =========="
for DOMAIN in "${DOMAINS[@]}"; do
    for MODE in standard_ep ep_hub ep_topo ours; do
        torchrun --nproc_per_node=4 experiments/ablation/comprehensive_ablation.py \
            --mode $MODE --domain $DOMAIN \
            --hub_size $BASE_HUB --top_k $BASE_TOPK --batch_size $BASE_BS --seq_len $BASE_SEQ --steps $STEPS
    done
done

# ------------------------------------------
# 战役 2：帕累托前沿探测 (Hub Size)
# ------------------------------------------
echo "========== 战役 2: Hub Size Pareto =========="
# 我们只需要用 ep_hub 模式来探测 Hub Size 的单纯物理收益
for HUB in "${HUB_SIZES[@]}"; do
    torchrun --nproc_per_node=4 experiments/ablation/comprehensive_ablation.py \
        --mode ep_hub --domain wiki \
        --hub_size $HUB --top_k $BASE_TOPK --batch_size $BASE_BS --seq_len $BASE_SEQ --steps $STEPS
done

# ------------------------------------------
# 战役 3：并发缩放墙 (Batch Size & SeqLen 矩阵)
# ------------------------------------------
echo "========== 战役 3: 并发缩放墙 (BS x Seq) =========="
# 选取最极端的对比：standard_ep vs ours
for BS in "${BATCH_SIZES[@]}"; do
    for SEQ in "${SEQ_LENS[@]}"; do
        for MODE in standard_ep ours; do
            torchrun --nproc_per_node=4 experiments/ablation/comprehensive_ablation.py \
                --mode $MODE --domain wiki \
                --hub_size $BASE_HUB --top_k $BASE_TOPK --batch_size $BS --seq_len $SEQ --steps $STEPS
        done
    done
done

# ------------------------------------------
# 战役 4：路由爆炸抗压 (Top-K)
# ------------------------------------------
echo "========== 战役 4: 路由爆炸 (Top-K) =========="
for K in "${TOP_KS[@]}"; do
    for MODE in standard_ep ours; do
        torchrun --nproc_per_node=4 experiments/ablation/comprehensive_ablation.py \
            --mode $MODE --domain wiki \
            --hub_size $BASE_HUB --top_k $K --batch_size $BASE_BS --seq_len $BASE_SEQ --steps $STEPS
    done
done

echo "🎉 [完美收官] 所有消融实验已全部安全结束，并受到 OOM 熔断保护！"
echo "请查看 outputs/eval_results/comprehensive_ablation/ 目录获取所有数据！"
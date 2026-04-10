import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# 动态定位项目根目录并导入 g2moe
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import EVAL_RESULTS_DIR, FIGURES_DIR

RESULTS_DIR = EVAL_RESULTS_DIR / "comprehensive_ablation"
OUTPUT_DIR = FIGURES_DIR / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})

# ==========================================
# 数据读取辅助函数
# ==========================================
def get_metric(mode, metric="step_total_traffic_mb", domain="wiki", h=4, k=4, b=1, s=8192):
    """从 JSON 日志中读取指定的 metric 平均值"""
    # 如果是计算 h=0 的情况，等同于 standard_ep (Baseline)
    if mode == "ep_hub" and h == 0:
        mode = "standard_ep"
        h = 4 # 恢复默认文件名参数
        
    path = RESULTS_DIR / f"{mode}_{domain}_H{h}_K{k}_B{b}_S{s}.json"
    if not path.exists():
        print(f"[警告] 文件不存在，返回 0.0: {path}")
        return 0.0
        
    with open(path, "r") as f:
        data = json.load(f)
        if data.get("status") == "OOM": 
            return 0.0
        return round(np.mean(data["metrics"].get(metric, [0])), 1)

# ==========================================
# 图 1：正交消融柱状图（仅流量）
# ==========================================
def plot_fig1_waterfall():
    # 动态读取流量数据
    traffic_vols = [
        get_metric('standard_ep'), 
        get_metric('ep_topo'), 
        get_metric('ep_hub'), 
        get_metric('ours')
    ]
    
    labels = ["Baseline\n(Standard EP)", "+ Gurobi Topo\n(Reduce Entropy)", "+ Hub Bypass\n(Reduce Volume)", "G2MoE\n(Complete)"]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    colors = ['#D5D8DC', '#F5B041', '#85C1E9', '#27AE60']
    x = np.arange(len(labels))
    width = 0.5

    bars = ax1.bar(x, traffic_vols, width, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Total Traffic Volume (MB)', color='black', fontweight='bold')
    
    # 根据动态数据自适应 Y 轴范围
    min_vol, max_vol = min(traffic_vols), max(traffic_vols)
    ax1.set_ylim(max(0, min_vol - 200), max_vol + 100) 
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max_vol-min_vol)*0.02, f'{yval:.1f} MB', ha='center', va='bottom', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.title("Orthogonal Ablation: Traffic Volume", pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_orthogonal_ablation.pdf")
    plt.close()


# ==========================================
# 图 2：Hub 增加带来的边际流量下降 (Marginal Reduction)
# 展现边际收益递减：前期降得快，后期降得慢
# ==========================================
def plot_fig2_elasticity():
    hubs = [0, 4, 8, 12, 16]
    # 动态读取不同 Hub 数量的流量
    raw_traffic = [get_metric('ep_hub', h=h) for h in hubs]
    
    # 计算边际下降值 (Marginal Reduction): 相较于上一个 Hub 数量的流量减少量
    marginal_reduction = []
    x_labels = []
    for i in range(1, len(hubs)):
        # 当前区间省下的流量 = 上一个阶段流量 - 当前阶段流量
        reduction = raw_traffic[i-1] - raw_traffic[i]
        # 防止出现负数（万一流量变大了），最小设为0
        marginal_reduction.append(max(0, reduction)) 
        x_labels.append(f"{hubs[i-1]} → {hubs[i]}")
        
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(x_labels))
    
    # 绘制辅助柱状图：展示每个区间的绝对收益块
    ax.bar(x_pos, marginal_reduction, width=0.45, color='#AED6F1', edgecolor='#2874A6', alpha=0.7)
    
    # 绘制折线图：强调“下降收益正在衰减”的趋势
    ax.plot(x_pos, marginal_reduction, marker='D', markersize=8, linewidth=2.5, color='#2874A6', mfc='#E74C3C')
    
    # 添加数据标签
    for i, txt in enumerate(marginal_reduction):
        ax.annotate(f"{txt:.1f}", (x_pos[i], marginal_reduction[i]), 
                    textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#C0392B')

    # 设置轴标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontweight='bold')
    ax.set_xlabel("Hub Addition Interval", fontweight='bold')
    ax.set_ylabel("Marginal Traffic Reduction (MB)", fontweight='bold')
    
    # Y轴自适应：确保底部从 0 开始或者留出合理空间，顶部留出空间给标签
    min_red, max_red = min(marginal_reduction), max(marginal_reduction)
    margin = (max_red - min_red) * 0.2 if max_red > min_red else 5
    ax.set_ylim(max(0, min_red - margin), max_red + margin * 1.5)
    
    plt.title("Marginal Diminishing Returns of Adding Hubs", pad=15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_marginal_reduction.pdf")
    plt.close()

# ==========================================
# 图 4：路由爆炸抗性 (Top-K)
# ==========================================
def plot_fig3_routing():
    ks = [2, 4, 8]
    # 动态读取不同 K 值的流量
    base = [get_metric('standard_ep', k=k) for k in ks]
    ours = [get_metric('ours', k=k) for k in ks]
    
    ratios = [(b-o)/b*100 if b > 0 else 0 for b, o in zip(base, ours)]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ks))
    width = 0.35
    
    ax.bar(x - width/2, base, width, label='Standard EP', color='#BDC3C7', edgecolor='black')
    ax.bar(x + width/2, ours, width, label='G2MoE', color='#27AE60', edgecolor='black')
    
    for i in range(len(ks)):
        ax.text(x[i] + width/2, ours[i] + (max(base)*0.02), f"-{ratios[i]:.1f}%", ha='center', va='bottom', color='#C0392B', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Traffic Volume (MB)")
    plt.title("Routing Resilience under Extreme Top-K")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_routing_resilience.pdf")
    plt.close()

if __name__ == "__main__":
    plot_fig1_waterfall()
    plot_fig2_elasticity()
    plot_fig3_routing()
    print(f"\n✅ 所有的图表已基于 {RESULTS_DIR.name} 中的 JSON 数据自动生成！请查看 {OUTPUT_DIR} 目录。")
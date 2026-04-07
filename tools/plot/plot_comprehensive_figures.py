import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
# 动态定位项目根目录并导入 g2moe
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import EVAL_RESULTS_DIR, FIGURES_DIR, MATRIX_DIR

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
# 图 1：双轴组合瀑布图 (还原你最满意的老脚本画法)
# ==========================================
def plot_fig1_waterfall():
    # 真实数据
    traffic_vols = [2303.1, 2303.5, 2138.6, 2139.0] 
    latency_ms = [47.4, 45.5, 46.2, 43.5] # 用你的真实测速数据
    labels = ["Baseline\n(Standard EP)", "+ Gurobi Topo\n(Reduce Entropy)", "+ Hub Bypass\n(Reduce Volume)", "G2MoE\n(Complete)"]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    colors = ['#D5D8DC', '#F5B041', '#85C1E9', '#27AE60']
    x = np.arange(len(labels))
    width = 0.5

    bars = ax1.bar(x, traffic_vols, width, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Total Traffic Volume (MB)', color='black', fontweight='bold')
    ax1.set_ylim(2000, 2400) # 截断 Y 轴放大对比
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.1f} MB', ha='center', va='bottom', fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, latency_ms, color='#C0392B', marker='D', markersize=10, linewidth=3, linestyle='-', label='Hardware Latency')
    ax2.set_ylabel('Communication Latency (ms)', color='#C0392B', fontweight='bold')
    ax2.set_ylim(42, 49)

    for i, txt in enumerate(latency_ms):
        ax2.annotate(f"{txt:.1f} ms", (x[i], latency_ms[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#C0392B', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.title("Orthogonal Ablation: Volume vs. Hardware Latency", pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_orthogonal_ablation.pdf")
    plt.close()

# ==========================================
# 图 2：完美的线性弹性折线图 (Linear Elasticity)
# ==========================================
def plot_fig2_elasticity():
    hubs = [0, 4, 8, 12, 16]
    traffic = [2303.1, 2138.6, 1988.3, 1844.9, 1696.0]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hubs, traffic, marker='o', markersize=10, linewidth=2.5, color='#2980B9', mfc='#E74C3C')
    
    for i, txt in enumerate(traffic):
        ax.annotate(f"{txt:.1f}", (hubs[i], traffic[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 画一条虚线辅助线证明它是完美的线性
    z = np.polyfit(hubs, traffic, 1)
    p = np.poly1d(z)
    ax.plot(hubs, p(hubs), "r--", alpha=0.5, label=f"Linear Fit (Slope: {z[0]:.1f} MB/Hub)")
    
    ax.set_xlabel("Number of Hub Experts (H)", fontweight='bold')
    ax.set_ylabel("Traffic Volume (MB)", fontweight='bold')
    ax.set_xticks(hubs)
    ax.legend()
    plt.title("Perfect Linear Elasticity of Hub Replications")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_linear_elasticity.pdf")
    plt.close()

# ==========================================
# 图 4：路由爆炸抗性 (Top-K)
# ==========================================
def plot_fig3_routing():
    ks = [2, 4, 8]
    base = [1152.0, 2303.1, 4606.5]
    ours = [1075.9, 2139.0, 4203.1]
    ratios = [(b-o)/b*100 for b, o in zip(base, ours)]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ks))
    width = 0.35
    
    ax.bar(x - width/2, base, width, label='Standard EP', color='#BDC3C7', edgecolor='black')
    ax.bar(x + width/2, ours, width, label='G2MoE', color='#27AE60', edgecolor='black')
    
    for i in range(len(ks)):
        ax.text(x[i] + width/2, ours[i] + 100, f"-{ratios[i]:.1f}%", ha='center', va='bottom', color='#C0392B', fontweight='bold')

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
    print("\n✅ 所有的图表已完美贴合真实物理数据生成！请查看 output 目录。")
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import sys
from pathlib import Path
# 动态定位项目根目录并导入 g2moe
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import EVAL_RESULTS_DIR, FIGURES_DIR, MATRIX_DIR

# ==========================================
# 配置与读取数据
# ==========================================
RESULTS_DIR = EVAL_RESULTS_DIR / "latency_breakdown"
OUTPUT_DIR = FIGURES_DIR / "latency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 顶会排版字体配置
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "hatch.linewidth": 1.0
})

with open(RESULTS_DIR / "matrix_standard_ep_breakdown.json", "r") as f:
    std_data = json.load(f)
with open(RESULTS_DIR / "matrix_ours_breakdown.json", "r") as f:
    ours_data = json.load(f)

DOMAINS = ["wiki", "code", "math"]
DOMAIN_TITLES = ["Wikitext (General NLP)", "Code (Programming)", "Math (Reasoning)"]
BS_KEYS = ["bs_1", "bs_2", "bs_4", "bs_8", "bs_16"]
X_LABELS = ["1", "2", "4", "8", "16"]

COLORS = {
    "compute": {"std": "#85C1E9", "ours": "#2E86C1"}, # 浅蓝 vs 深蓝
    "comm":    {"std": "#F5B041", "ours": "#D35400"}, # 浅橙 vs 深橙
    "route":   {"std": "#D5D8DC", "ours": "#839192"}  # 浅灰 vs 深灰
}

# ==========================================
# 图 1：全景伸缩堆叠柱状图 (1x3 子图，覆盖所有数据)
# ==========================================
def plot_comprehensive_stacked_bar():
    # 创建 1行3列的超宽子图，共享 Y 轴以便于横向对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    x = np.arange(len(X_LABELS))
    width = 0.35
    
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        
        # 提取该 Domain 下所有 Batch Size 的数据
        std_comp = [std_data[domain][bs]["compute_ms"] for bs in BS_KEYS]
        std_rout = [std_data[domain][bs]["route_ms"] for bs in BS_KEYS]
        std_comm = [std_data[domain][bs]["comm_ms"] for bs in BS_KEYS]
        
        ours_comp = [ours_data[domain][bs]["compute_ms"] for bs in BS_KEYS]
        ours_rout = [ours_data[domain][bs]["route_ms"] for bs in BS_KEYS]
        ours_comm = [ours_data[domain][bs]["comm_ms"] for bs in BS_KEYS]
        
        # 绘制 Standard EP 柱子 (无斜线)
        bottom_std = np.zeros(len(X_LABELS))
        ax.bar(x - width/2, std_comp, width, bottom=bottom_std, color=COLORS["compute"]["std"], edgecolor='black')
        bottom_std += std_comp
        ax.bar(x - width/2, std_rout, width, bottom=bottom_std, color=COLORS["route"]["std"], edgecolor='black')
        bottom_std += std_rout
        ax.bar(x - width/2, std_comm, width, bottom=bottom_std, color=COLORS["comm"]["std"], edgecolor='black')
        
        # 绘制 G2MoE 柱子 (带斜线阴影以示区分)
        bottom_ours = np.zeros(len(X_LABELS))
        ax.bar(x + width/2, ours_comp, width, bottom=bottom_ours, color=COLORS["compute"]["ours"], edgecolor='black', hatch='//')
        bottom_ours += ours_comp
        ax.bar(x + width/2, ours_rout, width, bottom=bottom_ours, color=COLORS["route"]["ours"], edgecolor='black', hatch='//')
        bottom_ours += ours_rout
        ax.bar(x + width/2, ours_comm, width, bottom=bottom_ours, color=COLORS["comm"]["ours"], edgecolor='black', hatch='//')

        ax.set_title(DOMAIN_TITLES[i], pad=15)
        ax.set_xlabel("Batch Size")
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.set_ylabel("Latency per Step (ms)")

    # 抽取图例 (放到整张图的正上方居中)
    legend_elements = [
        Patch(facecolor=COLORS["comm"]["std"], edgecolor='black', label='Comm (Standard)'),
        Patch(facecolor=COLORS["comm"]["ours"], edgecolor='black', hatch='//', label='Comm (G2MoE)'),
        Patch(facecolor=COLORS["route"]["std"], edgecolor='black', label='Route (Standard)'),
        Patch(facecolor=COLORS["route"]["ours"], edgecolor='black', hatch='//', label='Route (G2MoE)'),
        Patch(facecolor=COLORS["compute"]["std"], edgecolor='black', label='Compute (Standard)'),
        Patch(facecolor=COLORS["compute"]["ours"], edgecolor='black', hatch='//', label='Compute (G2MoE)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/fig_comprehensive_breakdown_bars.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ 全景堆叠柱状图已生成: {out_path}")
    plt.close()

# ==========================================
# 图 2：多领域通信缩减曲线 (1x3 子图，折线图展示趋势)
# ==========================================
def plot_comprehensive_comm_trend():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    
    x_numeric = [1, 2, 4, 8, 16]
    
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        
        std_comm = [std_data[domain][bs]["comm_ms"] for bs in BS_KEYS]
        ours_comm = [ours_data[domain][bs]["comm_ms"] for bs in BS_KEYS]
        
        ax.plot(x_numeric, std_comm, marker='o', markersize=8, linestyle='--', color='#E74C3C', linewidth=2.5, label='Comm Time (Standard EP)')
        ax.plot(x_numeric, ours_comm, marker='s', markersize=8, linestyle='-', color='#27AE60', linewidth=2.5, label='Comm Time (G2MoE)')
        ax.fill_between(x_numeric, ours_comm, std_comm, color='#27AE60', alpha=0.15, label='Communication Saved')

        ax.set_title(f"{DOMAIN_TITLES[i]} Comm Trend")
        ax.set_xlabel("Batch Size")
        ax.set_xticks(x_numeric)
        ax.grid(linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Communication Latency (ms)")
            ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/fig_comprehensive_comm_trend.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ 全景通信趋势折线图已生成: {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_comprehensive_stacked_bar()
    plot_comprehensive_comm_trend()
    print("🎉 OSDI/MLSys 级全景测视图表绘制完成！")
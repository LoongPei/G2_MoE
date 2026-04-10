import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as patches
from tqdm import tqdm
import sys
from scipy.interpolate import make_interp_spline
import matplotlib.patches as patches

# 动态添加项目根目录到 sys.path
# ==========================================
# 获取当前脚本所在目录 (tools/plot)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (向上退两级：tools/plot -> tools -> root)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 将项目根目录加入到 Python 的包搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# 1. 全局配置与数据加载
# ==========================================
from g2moe.config import RAW_CO_MATRIX_PATH, PMI_MATRIX_PATH, MARKOV_MATRIX_PATH, PLACEMENT_MAP_PATH, FIGURES_DIR, HUB_JSON_PATH

BASE_OUT_DIR = FIGURES_DIR / "motivation_all_layers"
DIR_FIG1 = BASE_OUT_DIR / "fig1_long_tail"
DIR_FIG2 = BASE_OUT_DIR / "fig2_pmi_comparison"
DIR_FIG3 = BASE_OUT_DIR / "fig3_markov_aggregated"
DIR_FIG4 = BASE_OUT_DIR / "fig4_hub_selection"  # 新增 Fig4 目录

for d in [DIR_FIG1, DIR_FIG2, DIR_FIG3, DIR_FIG4]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.titlesize": 14,
    "figure.dpi": 300, "axes.grid": False
})

print("⏳ 正在加载所有矩阵数据...")
all_intra_co = torch.load(RAW_CO_MATRIX_PATH, weights_only=True).numpy()
all_intra_pmi = torch.load(PMI_MATRIX_PATH, weights_only=True).numpy()
all_inter_markov = torch.load(MARKOV_MATRIX_PATH, weights_only=True).numpy()

with open(PLACEMENT_MAP_PATH, "r") as f:
    full_placement_map = json.load(f)

with open(HUB_JSON_PATH, "r", encoding="utf-8") as f:
    hub_scores_data = json.load(f)

num_layers = len(all_intra_co)

def get_layer_groups(layer_idx):
    """提取某层的 Hubs 和 4 个 GPU 的专家列表"""
    layer_map = full_placement_map[f"layer_{layer_idx}"]
    hubs = layer_map["shared_hubs_replicated_to_all_gpus"]
    gpu_experts = [layer_map["gpu_partitions"][f"gpu_{i}"]["experts"] for i in range(4)]
    return hubs, gpu_experts

# ==========================================
# 📊 构思一: 批量生成所有层的长尾分布图 (平滑曲线 + 自适应Y轴 + 高级悬浮标注)
# ==========================================
def plot_fig1_long_tail(l):
    activation_counts = np.diag(all_intra_co[l])
    sorted_indices = np.argsort(activation_counts)[::-1]
    freqs = activation_counts[sorted_indices] / np.sum(activation_counts) * 100
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x_coords = np.arange(len(freqs))
    
    # 1. B-spline 平滑曲线
    spl = make_interp_spline(x_coords, freqs, k=3)
    x_smooth = np.linspace(x_coords.min(), x_coords.max(), 300)
    y_smooth = spl(x_smooth)
    
    # 2. 绘制平滑曲线与散点
    ax.plot(x_smooth, y_smooth, color='#4c72b0', linestyle='-', linewidth=2, zorder=1, alpha=0.9)
    ax.scatter(x_coords, freqs, color='#4c72b0', s=20, edgecolor='white', linewidth=0.5, zorder=2)
    
    # 3. 自适应 Y 轴跨度 (留出上下 15% 的边距)
    y_min, y_max = min(freqs), max(freqs)
    y_margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 填充面积
    ax.fill_between(x_smooth, y_smooth, y2=y_min - y_margin, color='#4c72b0', alpha=0.1)
    
    # 4. 美观的高级���注 (首尾节点)
    first_val, last_val = freqs[0], freqs[-1]
    
    # 画大一号的空心圆圈强调首尾节点
    ax.scatter([0, len(freqs)-1], [first_val, last_val], facecolor='none', edgecolor='#4c72b0', s=100, linewidth=2, zorder=3)
    
    # 设计优雅的悬浮文本框样式
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#4c72b0", lw=1.2, alpha=0.95)
    
    # 标注首节点 (稍微往右放一点，避免遮挡 y 轴)
    ax.text(2, first_val, f"Max: {first_val:.2f}%", ha="left", va="center", 
            fontsize=10, fontweight='bold', color="#4c72b0", bbox=bbox_props, zorder=4)
            
    # 标注尾节点 (稍微往左放一点)
    ax.text(len(freqs)-3, last_val, f"Min: {last_val:.2f}%", ha="right", va="center", 
            fontsize=10, fontweight='bold', color="#4c72b0", bbox=bbox_props, zorder=4)
    
    ax.set_title(f'Expert Activation Frequency (Long Tail) - Layer {l}')
    ax.set_xlabel('Expert Rank (Sorted by Frequency)')
    ax.set_ylabel('Activation Share (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG1, f"layer_{l:02d}_tail.pdf"))
    plt.close(fig)

# ==========================================
# 📊 构思二: 批量生成 PMI 对比与重排热力图
# ==========================================
def plot_fig2_pmi_comparison(l):
    hubs, gpu_experts = get_layer_groups(l)
    pmi = all_intra_pmi[l]
    
    reordered_indices = hubs.copy()
    for g in gpu_experts: reordered_indices.extend(g)
    reordered_pmi = pmi[np.ix_(reordered_indices, reordered_indices)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 通用的坐标轴刻度设置 (0, 10, 20... 60)
    ticks = np.arange(0, 61, 10)
    
    # 左图：原生顺序
    sns.heatmap(pmi, cmap="YlGnBu", ax=axes[0], cbar=False)
    axes[0].set_title(f'Standard EP (Layer {l})')
    axes[0].set_xticks(ticks); axes[0].set_xticklabels(ticks)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(ticks)
    
    # 右图：重排顺序与物理框选
    sns.heatmap(reordered_pmi, cmap="YlGnBu", ax=axes[1], cbar_kws={'label': 'PMI Score'})
    axes[1].set_title(f'G2MoE Topology-Aware (Layer {l})')
    axes[1].set_xticks(ticks); axes[1].set_xticklabels(ticks)
    axes[1].set_yticks(ticks); axes[1].set_yticklabels(ticks)
    
    # 画框
    ax = axes[1]
    # 1. 红色虚线框出 Hub 安全区
    h_size = len(hubs)
    ax.add_patch(patches.Rectangle((0, 0), 60, h_size, fill=False, edgecolor='red', lw=2, ls='--'))
    ax.add_patch(patches.Rectangle((0, 0), h_size, 60, fill=False, edgecolor='red', lw=2, ls='--'))
    
    # 2. 画出 4 个 GPU 对角线框
    curr = h_size
    colors = ['#55a868', '#c44e52', '#8172b3', '#ccb974']
    for i, g_exp in enumerate(gpu_experts):
        s = len(g_exp)
        ax.add_patch(patches.Rectangle((curr, curr), s, s, fill=False, edgecolor=colors[i], lw=2.5))
        curr += s

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG2, f"layer_{l:02d}_pmi.pdf"))
    plt.close(fig)

# ==========================================
# 📊 构思三: 系统级聚合马尔可夫热力图
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fig3_markov_aggregated_physical(l):
    if l >= num_layers - 1: return
    
    src_hubs, src_gpus = get_layer_groups(l)
    dst_hubs, dst_gpus = get_layer_groups(l+1)
    
    src_groups = [src_hubs] + src_gpus
    dst_groups = [dst_hubs] + dst_gpus
    
    markov = all_inter_markov[l]
    raw_matrix = np.zeros((5, 5))
    
    # 1. 计算原始 5x5 矩阵
    for i in range(5):
        for j in range(5):
            sub_mat = markov[np.ix_(src_groups[i], dst_groups[j])]
            raw_matrix[i, j] = np.sum(sub_mat)
            
    # 2. 提取基础的纯 GPU 4x4 流量矩阵
    gpu_matrix = raw_matrix[1:5, 1:5].copy() 
    
    # 提取涉及 Hub 的流量
    hub_to_hub = raw_matrix[0, 0]
    gpus_to_hub = raw_matrix[1:5, 0]  # shape: (4,)
    hub_to_gpus = raw_matrix[0, 1:5]  # shape: (4,)
    
    # 计算 Hub 内数据的物理驻留概率权重 (基于有哪些 GPU 正在向 Hub 输出数据)
    hub_sum = np.sum(gpus_to_hub)
    w_i = gpus_to_hub / hub_sum if hub_sum > 0 else np.ones(4) / 4
    
    # 3. 按物理真值重新分配 Hub 流量
    for i in range(4):
        # 途径 1: GPU i -> Hub (物理上停留在本地)
        gpu_matrix[i, i] += gpus_to_hub[i]
        
        # 途径 3: Hub -> Hub (物理上继续停留在各自的本地)
        gpu_matrix[i, i] += hub_to_hub * w_i[i]
        
        for j in range(4):
            # 途径 2: Hub -> GPU j 
            # (数据原本以 w_i 的概率驻留在 GPU i 的 Hub 中，现在要传给 GPU j)
            gpu_matrix[i, j] += hub_to_gpus[j] * w_i[i]
            
    # 4. 行归一化计算流出转移概率
    row_sums = gpu_matrix.sum(axis=1, keepdims=True)
    agg_matrix = np.divide(gpu_matrix, row_sums, out=np.zeros_like(gpu_matrix), where=row_sums!=0)
    
    # 5. 画图
    labels = ["GPU 0", "GPU 1", "GPU 2", "GPU 3"]
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(agg_matrix, cmap="OrRd", annot=True, fmt=".2f", ax=ax, 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Transition Probability'})
    
    ax.set_title(f'Physical Device Flow (Layer {l} $\\rightarrow$ {l+1})\n(Based on Local Hub Assumption)')
    ax.set_xlabel(f'Target Physical GPUs (Layer {l+1})')
    ax.set_ylabel(f'Source Physical GPUs (Layer {l})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG3, f"layer_{l:02d}_to_{l+1:02d}_markov_physical.pdf"))
    plt.close(fig)

# ==========================================
# 📊 构思四: 批量生成 Hub 挑选分数与截断图 (平滑曲线 + 自适应Y轴，不标极值)
# ==========================================
def plot_fig4_hub_selection(l):
    hubs, _ = get_layer_groups(l)
    layer_data = hub_scores_data[f"layer_{l}"]
    sorted_experts = layer_data["hub_experts"]
    sorted_scores = layer_data["hub_scores"]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x_coords = np.arange(len(sorted_scores))
    
    # 1. B-spline 平滑曲线
    spl = make_interp_spline(x_coords, sorted_scores, k=3)
    x_smooth = np.linspace(x_coords.min(), x_coords.max(), 300)
    y_smooth = spl(x_smooth)
    
    hubs_int = [int(h) for h in hubs]
    colors = ['#c44e52' if int(exp) in hubs_int else '#4c72b0' for exp in sorted_experts]
    
    # 2. 绘制平滑曲线与带颜色的散点
    ax.plot(x_smooth, y_smooth, color='gray', linestyle='-', linewidth=1.5, zorder=1, alpha=0.6)
    ax.scatter(x_coords, sorted_scores, c=colors, s=40, edgecolor='white', linewidth=0.5, zorder=2)
    
    # 3. 自适应 Y 轴跨度
    y_min, y_max = min(sorted_scores), max(sorted_scores)
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 4. 画出 Hub 截断虚线
    num_hubs = len(hubs_int)
    ax.axvline(x=num_hubs - 0.5, color='gray', linestyle='--', alpha=0.8)
    
    # 优雅地放置截断文本
    bbox_cutoff = dict(boxstyle="round,pad=0.3", fc="#f2f2f2", ec="gray", lw=1, alpha=0.8)
    ax.text(num_hubs + 1, y_max - y_margin*0.5, 'Hub\nSelection\nCutoff', 
            ha='left', va='top', color='#333333', fontsize=10, bbox=bbox_cutoff)
    
    ax.set_title(f'Amplified Hub Score Distribution - Layer {l}')
    ax.set_xlabel('Expert Rank (Sorted by Hub Score)')
    ax.set_ylabel('Amplified Hub Score')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c44e52', markersize=8, label='Selected Global Hubs'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4c72b0', markersize=8, label='Specialized Experts')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG4, f"layer_{l:02d}_hub_scores.pdf"))
    plt.close(fig)

# ==========================================
# 🚀 执行主循环
# ==========================================
if __name__ == "__main__":
    print("🚀 开始批量生成顶会级 Motivation 神图...")
    for l in tqdm(range(num_layers), desc="Processing Layers"):
        plot_fig1_long_tail(l)
        plot_fig2_pmi_comparison(l)
        plot_fig3_markov_aggregated_physical(l)
        plot_fig4_hub_selection(l)

    print("\n🎉 大功告成！所有图表已完整生成至 ./paper_figures/motivation_all_layers 目录！")
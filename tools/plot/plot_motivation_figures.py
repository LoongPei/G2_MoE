import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as patches
from tqdm import tqdm

# ==========================================
# 1. 全局配置与数据加载
# ==========================================
from g2moe.config import RAW_CO_MATRIX_PATH, PMI_MATRIX_PATH, MARKOV_MATRIX_PATH, PLACEMENT_MAP_PATH

BASE_OUT_DIR = FIGURES_DIR / "motivation_all_layers"
DIR_FIG1 = BASE_OUT_DIR / "fig1_long_tail"
DIR_FIG2 = BASE_OUT_DIR / "fig2_pmi_comparison"
DIR_FIG3 = BASE_OUT_DIR / "fig3_network_islands"
DIR_FIG4 = BASE_OUT_DIR / "fig4_markov_aggregated"

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

num_layers = len(all_intra_co)

def get_layer_groups(layer_idx):
    """提取某层的 Hubs 和 4 个 GPU 的专家列表"""
    layer_map = full_placement_map[f"layer_{layer_idx}"]
    hubs = layer_map["shared_hubs_replicated_to_all_gpus"]
    gpu_experts = [layer_map["gpu_partitions"][f"gpu_{i}"]["experts"] for i in range(4)]
    return hubs, gpu_experts

# ==========================================
# 📊 构思一: 批量生成所有层的长尾分布图
# ==========================================
def plot_fig1_long_tail(l):
    hubs, _ = get_layer_groups(l)
    activation_counts = np.diag(all_intra_co[l])
    sorted_indices = np.argsort(activation_counts)[::-1]
    freqs = activation_counts[sorted_indices] / np.sum(activation_counts) * 100
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#c44e52' if idx in hubs else '#4c72b0' for idx in sorted_indices]
    ax.bar(range(len(freqs)), freqs, color=colors, edgecolor='black', linewidth=0.3)
    
    ax.set_title(f'Expert Activation Frequency - Layer {l}')
    ax.set_xlabel('Expert Rank')
    ax.set_ylabel('Activation Share (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], color='#c44e52', lw=6, label='Global Hubs'),
        mlines.Line2D([0], [0], color='#4c72b0', lw=6, label='Specialized')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
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
# 📊 构思三: 彻底重构的物理岛屿与核心拓扑图
# ==========================================
def plot_fig3_network_islands(l):
    hubs, gpu_experts = get_layer_groups(l)
    pmi = all_intra_pmi[l]
    activation = np.diag(all_intra_co[l])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    G = nx.Graph()
    pos = {}; node_colors = []; node_sizes = []
    
    # 画布背景岛屿
    ax.add_patch(patches.Circle((0, 0), 0.5, color='#f0f0f0', zorder=0)) # Hub 核心区
    quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    gpu_colors = ['#4c72b0', '#55a868', '#8172b3', '#ccb974']
    for i, (cx, cy) in enumerate(quadrants):
        ax.add_patch(patches.Circle((cx*1.5, cy*1.5), 0.8, color=gpu_colors[i], alpha=0.1, zorder=0))

    # 布置节点
    max_act = np.max(activation)
    for i, h in enumerate(hubs):
        angle = 2 * np.pi * i / len(hubs)
        pos[h] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))
        node_colors.append('#c44e52')
        node_sizes.append(activation[h]/max_act * 600 + 100)
        G.add_node(h)
        
    for g_idx, g_exp in enumerate(gpu_experts):
        cx, cy = quadrants[g_idx][0] * 1.5, quadrants[g_idx][1] * 1.5
        for i, exp in enumerate(g_exp):
            angle = 2 * np.pi * i / len(g_exp)
            pos[exp] = (cx + 0.5 * np.cos(angle), cy + 0.5 * np.sin(angle))
            node_colors.append(gpu_colors[g_idx])
            node_sizes.append(activation[exp]/max_act * 600 + 100)
            G.add_node(exp)

    # 布置连线 (Top 5%，按通信类型上色)
    threshold = np.percentile(pmi[pmi > 0], 95)
    edges_local, edges_hub, edges_remote = [], [], []
    weights_local, weights_hub, weights_remote = [], [], []
    
    for i in range(60):
        for j in range(i+1, 60):
            if pmi[i, j] > threshold:
                w = pmi[i, j] * 2
                G.add_edge(i, j)
                if i in hubs or j in hubs:
                    edges_hub.append((i, j)); weights_hub.append(w)
                else:
                    # 检查是否在同一个 GPU
                    same_gpu = False
                    for g_exp in gpu_experts:
                        if i in g_exp and j in g_exp: same_gpu = True; break
                    if same_gpu:
                        edges_local.append((i, j)); weights_local.append(w)
                    else:
                        edges_remote.append((i, j)); weights_remote.append(w)

    # 分批画线，强制开启 arrows=True 和 arrowstyle='-' 来支持优雅的弧线
    nx.draw_networkx_edges(G, pos, edgelist=edges_local, width=weights_local, edge_color='#2ca02c', alpha=0.6, arrows=True, arrowstyle='-', connectionstyle="arc3,rad=0.2", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_hub, width=weights_hub, edge_color='#1f77b4', alpha=0.4, arrows=True, arrowstyle='-', connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_remote, width=weights_remote, edge_color='#d62728', style='dashed', alpha=0.8, arrows=True, arrowstyle='-', connectionstyle="arc3,rad=0.3", ax=ax)
    
    # 画节点 (删除了报错的 zorder 参数，按执行顺序自然覆盖在连线上)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='white', linewidths=1.5, ax=ax)
    
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], color='#2ca02c', lw=3, label='Intra-GPU (0 Traffic)'),
        mlines.Line2D([0], [0], color='#1f77b4', lw=3, label='To Hub (0 Traffic)'),
        mlines.Line2D([0], [0], color='#d62728', lw=3, ls='--', label='Cross-GPU (Bottleneck)'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c44e52', markersize=10, label='Hub Core'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8172b3', markersize=10, label='GPU Islands')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f'Hardware-Aligned Routing Islands (Layer {l})')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG3, f"layer_{l:02d}_islands.pdf"))
    plt.close(fig)

# ==========================================
# 📊 构思四: 系统级聚合马尔可夫热力图
# ==========================================
def plot_fig4_markov_aggregated(l):
    if l >= num_layers - 1: return
    
    src_hubs, src_gpus = get_layer_groups(l)
    dst_hubs, dst_gpus = get_layer_groups(l+1)
    
    src_groups = [src_hubs] + src_gpus
    dst_groups = [dst_hubs] + dst_gpus
    
    markov = all_inter_markov[l]
    agg_matrix = np.zeros((5, 5))
    
    for i in range(5):
        for j in range(5):
            sub_mat = markov[np.ix_(src_groups[i], dst_groups[j])]
            agg_matrix[i, j] = np.sum(sub_mat)
            
    # 归一化，使其按行求和为1 (表示流出概率)
    row_sums = agg_matrix.sum(axis=1, keepdims=True)
    agg_matrix = np.divide(agg_matrix, row_sums, out=np.zeros_like(agg_matrix), where=row_sums!=0)
    
    labels = ["Hubs", "GPU 0", "GPU 1", "GPU 2", "GPU 3"]
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(agg_matrix, cmap="OrRd", annot=True, fmt=".2f", ax=ax, 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Transition Probability'})
    
    ax.set_title(f'System-Level Markov Flow (Layer {l} $\\rightarrow$ {l+1})')
    ax.set_xlabel(f'Target Physical Groups (Layer {l+1})')
    ax.set_ylabel(f'Source Physical Groups (Layer {l})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG4, f"layer_{l:02d}_to_{l+1:02d}_markov.pdf"))
    plt.close(fig)

# ==========================================
# 🚀 执行主循环
# ==========================================
if __name__ == "__main__":
    print("🚀 开始批量生成顶会级 Motivation 神图...")
    for l in tqdm(range(num_layers), desc="Processing Layers"):
        plot_fig1_long_tail(l)
        plot_fig2_pmi_comparison(l)
        plot_fig3_network_islands(l)
        plot_fig4_markov_aggregated(l)
        
    print("\n🎉 大功告成！所有 24 层（超 90 张高清 PDF 图表）已完整生成至 ./paper_figures/motivation_all_layers 目录！")
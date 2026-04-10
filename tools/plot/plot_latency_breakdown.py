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

from g2moe.config import EVAL_RESULTS_DIR, FIGURES_DIR

# ==========================================
# 配置与基础数据
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

DOMAINS = ["wiki", "code", "math"]
DOMAIN_TITLES = ["Wikitext (General NLP)", "Code (Programming)", "Math (Reasoning)"]

COLORS = {
    "compute": {"std": "#85C1E9", "ours": "#2E86C1"}, # 浅蓝 vs 深蓝
    "comm":    {"std": "#F5B041", "ours": "#D35400"}, # 浅橙 vs 深橙
    "route":   {"std": "#D5D8DC", "ours": "#839192"}  # 浅灰 vs 深灰
}

def load_exp_data(exp_name):
    """根据实验名称读取对应子文件夹的 JSON 数据"""
    std_path = RESULTS_DIR / exp_name / "matrix_standard_ep_breakdown.json"
    ours_path = RESULTS_DIR / exp_name / "matrix_ours_breakdown.json"
    
    with open(std_path, "r") as f:
        std_data = json.load(f)
    with open(ours_path, "r") as f:
        ours_data = json.load(f)
        
    return std_data, ours_data

def get_metric_safe(data, domain, seq_key, bs_key, metric):
    """安全获取指标数据（防止某些配置由于 OOM 没跑完而导致报错）"""
    try:
        return data[domain][seq_key][bs_key][metric]
    except KeyError:
        return 0.0  # 如果缺失，说明触发了 OOM 或未运行，返回 0

# ==========================================
# 通用绘图函数：堆叠柱状图
# ==========================================
def plot_comprehensive_stacked_bar(std_data, ours_data, exp_name, x_labels, seq_keys, bs_keys, xlabel_text):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    x = np.arange(len(x_labels))
    width = 0.35
    
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        
        # 提取数据
        std_comp = [get_metric_safe(std_data, domain, sq, bs, "compute_ms") for sq, bs in zip(seq_keys, bs_keys)]
        std_rout = [get_metric_safe(std_data, domain, sq, bs, "route_ms") for sq, bs in zip(seq_keys, bs_keys)]
        std_comm = [get_metric_safe(std_data, domain, sq, bs, "comm_ms") for sq, bs in zip(seq_keys, bs_keys)]
        
        ours_comp = [get_metric_safe(ours_data, domain, sq, bs, "compute_ms") for sq, bs in zip(seq_keys, bs_keys)]
        ours_rout = [get_metric_safe(ours_data, domain, sq, bs, "route_ms") for sq, bs in zip(seq_keys, bs_keys)]
        ours_comm = [get_metric_safe(ours_data, domain, sq, bs, "comm_ms") for sq, bs in zip(seq_keys, bs_keys)]
        
        # 绘制 Standard EP 柱子 (无斜线)
        bottom_std = np.zeros(len(x_labels))
        ax.bar(x - width/2, std_comp, width, bottom=bottom_std, color=COLORS["compute"]["std"], edgecolor='black')
        bottom_std += std_comp
        ax.bar(x - width/2, std_rout, width, bottom=bottom_std, color=COLORS["route"]["std"], edgecolor='black')
        bottom_std += std_rout
        ax.bar(x - width/2, std_comm, width, bottom=bottom_std, color=COLORS["comm"]["std"], edgecolor='black')
        
        # 绘制 G2MoE 柱子 (带斜线阴影以示区分)
        bottom_ours = np.zeros(len(x_labels))
        ax.bar(x + width/2, ours_comp, width, bottom=bottom_ours, color=COLORS["compute"]["ours"], edgecolor='black', hatch='//')
        bottom_ours += ours_comp
        ax.bar(x + width/2, ours_rout, width, bottom=bottom_ours, color=COLORS["route"]["ours"], edgecolor='black', hatch='//')
        bottom_ours += ours_rout
        ax.bar(x + width/2, ours_comm, width, bottom=bottom_ours, color=COLORS["comm"]["ours"], edgecolor='black', hatch='//')

        ax.set_title(DOMAIN_TITLES[i], pad=15)
        ax.set_xlabel(xlabel_text)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.set_ylabel("Latency per Step (ms)")

    # 图例
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
    
    out_path = f"{OUTPUT_DIR}/fig_{exp_name}_breakdown_bars.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ [{exp_name}] 堆叠柱状图已生成: {out_path}")
    plt.close()

# ==========================================
# 通用绘图函数：通信趋势折线图
# ==========================================
def plot_comprehensive_comm_trend(std_data, ours_data, exp_name, x_numeric, x_labels, seq_keys, bs_keys, xlabel_text):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        
        std_comm = [get_metric_safe(std_data, domain, sq, bs, "comm_ms") for sq, bs in zip(seq_keys, bs_keys)]
        ours_comm = [get_metric_safe(ours_data, domain, sq, bs, "comm_ms") for sq, bs in zip(seq_keys, bs_keys)]
        
        ax.plot(x_numeric, std_comm, marker='o', markersize=8, linestyle='--', color='#E74C3C', linewidth=2.5, label='Comm Time (Standard EP)')
        ax.plot(x_numeric, ours_comm, marker='s', markersize=8, linestyle='-', color='#27AE60', linewidth=2.5, label='Comm Time (G2MoE)')
        
        # 填充优化区域
        std_comm_arr, ours_comm_arr = np.array(std_comm), np.array(ours_comm)
        ax.fill_between(x_numeric, ours_comm_arr, std_comm_arr, where=(std_comm_arr >= ours_comm_arr), color='#27AE60', alpha=0.15, label='Communication Saved', interpolate=True)

        ax.set_title(f"{DOMAIN_TITLES[i]} Comm Trend")
        ax.set_xlabel(xlabel_text)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(x_labels)
        ax.grid(linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Communication Latency (ms)")
            ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/fig_{exp_name}_comm_trend.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ [{exp_name}] 通信趋势折线图已生成: {out_path}")
    plt.close()


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 实验一：SeqLen (固定 Global Batch = 4, 变动 SeqLen)
    # ---------------------------------------------------------
    try:
        std_seqlen, ours_seqlen = load_exp_data("seqlen")
        seq_lens = ["1024", "2048", "4096", "8192"]
        sq_keys = [f"seq_{s}" for s in seq_lens]
        bs_keys = ["global_bs_4"] * len(seq_lens)
        
        plot_comprehensive_stacked_bar(std_seqlen, ours_seqlen, "seqlen", seq_lens, sq_keys, bs_keys, "Sequence Length (BS=4)")
        plot_comprehensive_comm_trend(std_seqlen, ours_seqlen, "seqlen", [1, 2, 3, 4], seq_lens, sq_keys, bs_keys, "Sequence Length (BS=4)")
    except Exception as e:
        print(f"❌ 无法处理 seqlen 实验数据: {e}")

    # ---------------------------------------------------------
    # 实验二：BatchSize (固定 SeqLen = 1024, 变动 Global Batch)
    # ---------------------------------------------------------
    try:
        std_batchsize, ours_batchsize = load_exp_data("batchsize")
        batches = ["4", "8", "16", "32"]
        bs_keys_2 = [f"global_bs_{b}" for b in batches]
        sq_keys_2 = ["seq_1024"] * len(batches)
        
        plot_comprehensive_stacked_bar(std_batchsize, ours_batchsize, "batchsize", batches, sq_keys_2, bs_keys_2, "Global Batch Size (Seq=1024)")
        plot_comprehensive_comm_trend(std_batchsize, ours_batchsize, "batchsize", [4, 8, 16, 32], batches, sq_keys_2, bs_keys_2, "Global Batch Size (Seq=1024)")
    except Exception as e:
        print(f"❌ 无法处理 batchsize 实验数据: {e}")

    print("\n🎉 OSDI/MLSys 级两组测视图表全部绘制完成！")
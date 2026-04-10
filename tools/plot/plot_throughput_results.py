import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# 动态定位项目根目录并导入 g2moe
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import FIGURES_DIR, MATRIX_DIR
from g2moe.config import OUTPUT_DIR as GLOBAL_OUTPUT_DIR

# ==========================================
# 1. 全局配置
# ==========================================
OUTPUT_DIR = FIGURES_DIR / "performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 顶会风格设置
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

# ==========================================
# 2. 通用数据读取器
# ==========================================
def load_data(data_dir, x_axis_key):
    """
    x_axis_key 可以是 'global_batch' 或 'seq_len'
    """
    data = {"baseline": {}, "standard_ep": {}, "ours": {}}
    if not os.path.exists(data_dir): 
        print(f"⚠️ 警告: 未找到数据目录 {data_dir}")
        return data
    
    for file in os.listdir(data_dir):
        if file.endswith("_data.json"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                item = json.load(f)
                mode = item["mode"]
                x_val = item[x_axis_key]
                data[mode][x_val] = item
    return data

# ==========================================
# 3. 通用吞吐量画图函数
# ==========================================
def plot_throughput(data, x_axis_name, title, filename_prefix):
    # 提取所有出现的 X 轴坐标 (比如 [4, 8, 16, 32, 64] 或 [1024, 2048, 4096, 8192])
    x_vals = sorted(list(set(
        list(data["baseline"].keys()) + 
        list(data["standard_ep"].keys()) + 
        list(data["ours"].keys())
    )))
    
    if not x_vals: return

    # 提取吞吐量，缺失的用 0 代替 (意味着 OOM)
    tp_base = [data["baseline"].get(x, {}).get("throughput_tokens_per_sec", 0) for x in x_vals]
    tp_std = [data["standard_ep"].get(x, {}).get("throughput_tokens_per_sec", 0) for x in x_vals]
    tp_ours = [data["ours"].get(x, {}).get("throughput_tokens_per_sec", 0) for x in x_vals]

    x_indices = np.arange(len(x_vals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x_indices - width, tp_base, width, label='Native Pipeline', color='#4c72b0', edgecolor='black')
    ax.bar(x_indices, tp_std, width, label='Standard EP', color='#55a868', edgecolor='black')
    ax.bar(x_indices + width, tp_ours, width, label='G2MoE (Ours)', color='#c44e52', edgecolor='black', hatch='//')

    ax.set_ylabel('Throughput (Tokens/sec)')
    ax.set_xlabel(x_axis_name)
    ax.set_title(title)
    
    # 横坐标转换为字符串以便于展示
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(val) for val in x_vals])
    ax.legend(loc='upper left')

    # 给 OOM 的柱子上打个标
    for i, tp in enumerate(tp_base):
        if tp == 0 and x_vals[i] == max(x_vals):  # 如果最大规模时得分为0，判定为OOM
            ax.text(x_indices[i] - width, max(tp_std + tp_ours)*0.05, "OOM", 
                    ha='center', va='bottom', color='black', fontsize=12, fontweight='bold', rotation=90)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_throughput.pdf")
    plt.savefig(out_path)
    print(f"✅ 吞吐量对比图已生成: {out_path}")
    plt.close()

# ==========================================
# 4. 通用通信削减画图函数
# ==========================================
def plot_traffic_reduction(data, x_axis_name, title, filename_prefix):
    x_vals = sorted(list(data["ours"].keys()))
    if not x_vals: return
    
    traffic_std = [data["standard_ep"].get(x, {}).get("total_traffic_mb", 0) for x in x_vals]
    traffic_ours = [data["ours"].get(x, {}).get("total_traffic_mb", 0) for x in x_vals]
    
    # 忽略没有跑出数据的节点
    valid_indices = [i for i, v in enumerate(traffic_std) if v > 0]
    x_vals = [x_vals[i] for i in valid_indices]
    traffic_std = [traffic_std[i] for i in valid_indices]
    traffic_ours = [traffic_ours[i] for i in valid_indices]

    x_indices = np.arange(len(x_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(x_indices - width/2, traffic_std, width, label='Standard EP', color='#8c8c8c', edgecolor='black')
    ax.bar(x_indices + width/2, traffic_ours, width, label='G2MoE (Ours)', color='#dd8452', edgecolor='black', hatch='\\\\')

    ax.set_ylabel('Cross-GPU Traffic Volume (MB)')
    ax.set_xlabel(x_axis_name)
    ax.set_title(title)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(val) for val in x_vals])
    ax.legend(loc='upper left')

    # 在顶部标注削减百分比
    for i in range(len(x_vals)):
        if traffic_std[i] > 0:
            reduction = (traffic_std[i] - traffic_ours[i]) / traffic_std[i] * 100
            ax.text(x_indices[i], max(traffic_std[i], traffic_ours[i]) * 1.02, f"-{reduction:.1f}%", 
                    ha='center', va='bottom', color='red', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_traffic_reduction.pdf")
    plt.savefig(out_path)
    print(f"✅ 通信削减对比图已生成: {out_path}")
    plt.close()

# ==========================================
# 🚀 主程序执行
# ==========================================
if __name__ == "__main__":
    print("🚀 开始生成系统性能评估图表...")

    # 任务 1: Batch Size 扩展性测试图表
    dir_batch = GLOBAL_OUTPUT_DIR / "throughput" / "batchsize"
    if os.path.exists(dir_batch):
        print("\n[1/2] 处理 Batch Size 扩展性数据...")
        data_batch = load_data(dir_batch, x_axis_key="global_batch")
        plot_throughput(data_batch, "Global Batch Size", "Throughput Scaling (Fixed Seq=1024)", "batch_scaling")
        plot_traffic_reduction(data_batch, "Global Batch Size", "Traffic Reduction (Fixed Seq=1024)", "batch_scaling")

    # 任务 2: Sequence Length 扩展性测试图表
    dir_seqlen = GLOBAL_OUTPUT_DIR / "throughput" / "seqlen"
    if os.path.exists(dir_seqlen):
        print("\n[2/2] 处理 Sequence Length 扩展性数据...")
        data_seqlen = load_data(dir_seqlen, x_axis_key="seq_len")
        plot_throughput(data_seqlen, "Sequence Length", "Long-Context Throughput Scaling (Fixed Batch=8)", "seqlen_scaling")
        plot_traffic_reduction(data_seqlen, "Sequence Length", "Traffic Reduction (Fixed Batch=8)", "seqlen_scaling")

    print(f"\n🎉 所有图表已就绪！存放在 {OUTPUT_DIR} 目录中。")
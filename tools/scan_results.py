import os
import json
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import EVAL_RESULTS_DIR

RESULTS_DIR = EVAL_RESULTS_DIR / "comprehensive_ablation"

def get_vol(mode, domain="wiki", h=4, k=4, b=1, s=8192):
    path = RESULTS_DIR / f"{mode}_{domain}_H{h}_K{k}_B{b}_S{s}.json"
    if not path.exists(): return "MISSING"
    with open(path, "r") as f:
        data = json.load(f)
        if data.get("status") == "OOM": return "OOM"
        return round(np.mean(data["metrics"]["step_total_traffic_mb"]), 1)

print("========== 战役 1: 多领域 (Wiki/Code/Math) ==========")
for d in ["wiki", "code", "math"]:
    print(f"[{d.upper()}] Base: {get_vol('standard_ep', d)} | Topo: {get_vol('ep_topo', d)} | Hub: {get_vol('ep_hub', d)} | Ours: {get_vol('ours', d)}")

print("\n========== 战役 2: Hub Size 线性度 ==========")
for h in [0, 4, 8, 12, 16]:
    print(f"Hub={h}: {get_vol('ep_hub', 'wiki', h)}")

print("\n========== 战役 3: 扩展墙 (BS & SeqLen) ==========")
for b in [1, 2, 4, 8]:
    for s in [2048, 4096, 8192]:
        base = get_vol("standard_ep", "wiki", 4, 4, b, s)
        ours = get_vol("ours", "wiki", 4, 4, b, s)
        ratio = f"{round((base-ours)/base*100,1)}%" if isinstance(base, float) and isinstance(ours, float) else "N/A"
        print(f"BS={b}, Seq={s} -> Base: {base} | Ours: {ours} | Reduction: {ratio}")

print("\n========== 战役 4: 路由爆炸 (Top-K) ==========")
for k in [2, 4, 8]:
    base = get_vol("standard_ep", "wiki", 4, k)
    ours = get_vol("ours", "wiki", 4, k)
    print(f"Top-{k} -> Base: {base} | Ours: {ours}")
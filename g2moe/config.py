import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ==========================================
# 1. 外部依赖路径 (模型与数据集)
# ==========================================
CACHE_DIR = os.getenv("G2MOE_CACHE_DIR", "/opt/data/longpei/cache/G2MoE")
HF_DATASETS_CACHE = os.getenv("G2MOE_HF_DATASETS_CACHE", "/opt/data/longpei/cache/huggingface/c2r_eval")

# 默认模型绝对路径
DEFAULT_MODEL_NAME = os.getenv("G2MOE_DEFAULT_MODEL", "Qwen1.5-MoE-A2.7B")
DEFAULT_MODEL_PATH = os.path.join(CACHE_DIR, DEFAULT_MODEL_NAME)

# ==========================================
# 2. 项目内部路径 (实验输出与矩阵)
# ==========================================
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 矩阵与拓扑策略路径
MATRIX_DIR = OUTPUT_DIR / "matrices"
RAW_CO_MATRIX_PATH = MATRIX_DIR / "raw_co_occurrence.pt"
PMI_MATRIX_PATH = MATRIX_DIR / "intra_pmi_matrix.pt"
MARKOV_MATRIX_PATH = MATRIX_DIR / "inter_markov_matrix.pt"
HUB_JSON_PATH = MATRIX_DIR / "Hub_Analysis" / "identified_hub_experts.json"
PLACEMENT_MAP_PATH = MATRIX_DIR / "Placement_Strategy" / "affinity_placement_map.json"

# 评估结果路径
EVAL_RESULTS_DIR = OUTPUT_DIR / "eval_results"

# 图表输出路径
FIGURES_DIR = OUTPUT_DIR / "figures"

def ensure_dirs():
    """确保所有输出目录存在"""
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ensure_dirs()
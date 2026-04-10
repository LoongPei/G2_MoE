# G²MoE: Graph-to-Grid Topology-Aware Routing for Mixture-of-Experts

This repository contains the official implementation of **G2MoE**, a system-level optimization for Mixture-of-Experts (MoE) models. G2MoE aligns expert routing with underlying physical hardware topology (e.g., PCIe/NVLink inter-GPU bandwidth) to minimize communication bottlenecks via a highly decoupled, Adapter-based execution engine.

## 📂 Repository Structure

```text
G2MoE/
├── g2moe/                  # Core library
│   ├── adapters/           # Model-specific logic (e.g., Qwen1.5, Mixtral)
│   ├── core/               # Universal Engine (HybridMoEWrapper, Factory)
│   ├── solver/             # Mathematical Solvers (Gurobi MIQP)
│   └── utils/              # Common utilities (Data loaders)
├── tools/                  # Offline pipeline (Topology, Matrices, Plotting)
├── experiments/            # Main entrypoints for evaluation and benchmarking
├── scripts/                # Bash scripts for automated execution
└── outputs/                # Auto-generated artifacts (matrices, results, figures)
```

## 🛠️ Step 0: Environment Setup

We provide the exact environment frozen from our testing machines to guarantee reproducibility.

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate gmoe
```
*Alternatively, using pip:*
```bash
pip install -r requirements.txt
```

## 📥 Step 1: Preparation (Models & Configuration)

G2MoE uses a `.env` file to manage paths dynamically, decoupling code from local hardcoded absolute paths.

1. Create a `.env` file in the project root:
```env
# .env
G2MOE_CACHE_DIR=/path/to/your/model_and_data_cache
G2MOE_HF_DATASETS_CACHE=/path/to/your/huggingface_datasets_cache
G2MOE_DEFAULT_MODEL=Qwen1.5-MoE-A2.7B
```

2. Download the base model weights from HuggingFace to your `G2MOE_CACHE_DIR`. For instance, download `Qwen/Qwen1.5-MoE-A2.7B` and rename/symlink it to match `G2MOE_DEFAULT_MODEL`.

3. Ensure you have the datasets ready. The `lm-eval` harness and our custom data loaders will automatically fetch necessary data to `G2MOE_HF_DATASETS_CACHE` if internet access is available, or load them offline if already downloaded.

## ⚙️ Step 2: Offline Topology Pipeline

Before running inference, G2MoE requires profiling the model's routing behavior and solving for the optimal physical expert placement. Run the following commands sequentially:

**1. Generate Natural Workload Dataset:**
Extracts a 5000-document slice from The Pile to simulate realistic routing distribution.
```bash
python tools/get_dataset.py
```

**2. Profile Matrices (Co-occurrence, PMI, Markov):**
Captures the intra-layer and inter-layer routing correlations.
```bash
python tools/generate_matrix.py
```

**3. Identify Heavy-Hitter Experts (Hubs):**
Analyzes the long-tail distribution of expert usage.
```bash
python tools/analyze_hubs.py
```

**4. 3D Physical Placement (MIQP Solver):**
Uses Gurobi to map experts to physical GPUs based on hardware topology.
```bash
python tools/solve_placement.py
```
*Outputs are saved to `outputs/matrices/`.*

## 🚀 Step 3: Benchmarking & Profiling

Evaluate the system-level throughput and perform a micro-level latency breakdown of Compute, Comm, and Route.

**Throughput Benchmark:**
```bash
torchrun --nproc_per_node=4 experiments/throughput/throughput_benchmark.py --mode ours --global_batch 16 --seq_len 1024
```

```bash
torchrun --nproc_per_node=4 experiments/throughput/throughput_benchmark.py --mode standard_ep --global_batch 16 --seq_len 1024
```

```bash
python experiments/throughput/throughput_benchmark.py --mode baseline --global_batch 16 --seq_len 1024
```
```bash
# 实验脚本
. /opt/data/xie/projects/G2_MoE/scripts/run_benchmark.sh
```

You can run multiple experiments to compare the performance of various methods by modifying the batch_size and seqlen parameters. 

Note: When mode=baseline, the program's startup command is different from the other two modes.
当mode=ours或mode=standard_ep时，global_batch必须是4的倍数

**Micro-Latency Breakdown:**

```bash
torchrun --nproc_per_node=4 experiments/throughput/latency_breakdown.py --mode ours --seq_lens 1024 2048 4096 8192 --global_batches 4 --exp_name batchsize
```
```bash
# 实验脚本
. /opt/data/xie/projects/G2_MoE/scripts/run_latence.sh
```


## 🧪 Step 4: Comprehensive Ablation & Downstream Tasks

**Automated Ablation Suite:**
Run our OSDI-grade automated script that sweeps across multiple domains, batch sizes, sequence lengths, and top-k routing parameters:
```bash
bash scripts/run_all_ablations.sh
```

**Downstream Accuracy Fidelity Check:**
Verify that G2MoE achieves **zero degradation** on standardized tasks (SciQ, RTE, etc.) using `lm-eval`:
```bash
# Evaluate baseline
torchrun --nproc_per_node=4 experiments/downstream/eval_downstream_baseline.py
# Evaluate G2MoE
torchrun --nproc_per_node=4 experiments/downstream/eval_downstream_ours.py
```

## 📊 Step 5: Visualization

Once the experiments finish, generate paper-ready PDF figures (saved in `outputs/figures/`):

```bash
# Example: Plot comprehensive ablation results
python tools/plot/plot_comprehensive_figures.py

# Example: Plot micro-latency breakdowns
python tools/plot/plot_latency_breakdown.py

# You can run any plotting script under tools/plot/
```

## 📜 License & Citation
*(Include your license and BibTeX citation block here once published).*

import json
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from g2moe.config import PLACEMENT_MAP_PATH, HUB_JSON_PATH
from g2moe.core.wrapper import HybridMoEWrapper
from g2moe.adapters.qwen import QwenMoEAdapter

def build_g2moe_model(model_path, mode="ours", rank=0, world_size=1, target_hub_size=4, target_top_k=None):
    """
    统一的模型组���工厂。负责加载 HF 模型，读取物理拓扑矩阵，并无缝植入 G2MoE Wrapper。
    """
    # 1. 加载 HuggingFace 模型骨架
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cpu", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    # 2. 自动选择适配器 (目前默认 Qwen，后续可根据 model.config.model_type 动态选择)
    adapter = QwenMoEAdapter()
    
    # 3. 读取底层物理拓扑策略
    with open(PLACEMENT_MAP_PATH, "r", encoding="utf-8") as f:
        placement_map = json.load(f)
    
    hub_analysis = None
    if mode == "ep_hub":
        with open(HUB_JSON_PATH, "r", encoding="utf-8") as f:
            hub_analysis = json.load(f)

    # 4. 逐层植入 G2MoE 核心引擎
    layers = adapter.get_layers(model)
    for l_idx, layer in enumerate(layers):
        # 兼容不同命名，提取真正的 MLP 模块
        orig_mlp = layer.mlp if hasattr(layer, 'mlp') else layer
        
        # 动态修改 Top-K (如果有提供)
        if target_top_k is not None:
            router = adapter.get_router(orig_mlp)
            if hasattr(router, "top_k"):
                router.top_k = target_top_k
                
        layer_map = placement_map[f"layer_{l_idx}"]
        real_gpu_partitions = [layer_map["gpu_partitions"][f"gpu_{g}"]["experts"] for g in range(world_size)]
        
        # --- 策略路由 ---
        if mode == "standard_ep":
            hub_ids = []
            exp_per_gpu = 60 // world_size
            global_map = [list(range(g * exp_per_gpu, (g + 1) * exp_per_gpu)) for g in range(world_size)]
            
        elif mode == "ep_hub":
            all_available_hubs = hub_analysis[f"layer_{l_idx}"]["hub_experts"]
            hub_ids = [int(e) for e in all_available_hubs[:target_hub_size]]
            remaining_experts = [e for e in range(60) if e not in hub_ids]
            chunk = len(remaining_experts) // world_size
            global_map = [remaining_experts[g * chunk : (g + 1) * chunk] for g in range(world_size)]
                    
        elif mode == "ep_topo":
            hub_ids = []
            global_map = [part.copy() for part in real_gpu_partitions]
            for i, h_id in enumerate(layer_map["shared_hubs_replicated_to_all_gpus"]):
                global_map[i % world_size].append(h_id)
                
        elif mode == "ours":
            hub_ids = layer_map["shared_hubs_replicated_to_all_gpus"]
            global_map = [part.copy() for part in real_gpu_partitions]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # --- 执行替换 ---
        new_mlp = HybridMoEWrapper(
            orig_mlp=orig_mlp,
            hub_ids=hub_ids, 
            my_specialized_ids=global_map[rank], 
            global_specialized_map=global_map,
            adapter=adapter
        )
        
        # 将新包装的模块注回模型中
        if hasattr(layer, 'mlp'):
            layer.mlp = new_mlp
        else:
            # 兼容其他非 qwen 命名架构
            setattr(layer, 'mlp', new_mlp)
            
    # 5. 挂载到指定 GPU 设备
    model.to(torch.device(f"cuda:{rank}"))
    return model
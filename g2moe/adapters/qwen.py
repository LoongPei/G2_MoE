import torch.nn.functional as F
from .base import BaseMoEAdapter

class QwenMoEAdapter(BaseMoEAdapter):
    """
    Qwen1.5-MoE 架构专属适配器
    """
    def get_layers(self, model):
        return model.model.layers
        
    def get_router(self, layer_mlp):
        return layer_mlp.gate
        
    def get_expert_weights(self, layer_mlp, expert_id: int):
        # Qwen 的专家权重以连续大矩阵存储
        return {
            "gate_up": layer_mlp.experts.gate_up_proj.data[expert_id].clone(),
            "down": layer_mlp.experts.down_proj.data[expert_id].clone()
        }
        
    def expert_forward(self, hidden_states, weights_dict):
        # Qwen 的 FFN 计算公式: down(SiLU(gate) * up)
        gate, up = F.linear(hidden_states, weights_dict["gate_up"]).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, weights_dict["down"])
        
    def get_shared_expert_components(self, layer_mlp):
        shared_expert = getattr(layer_mlp, 'shared_expert', None)
        shared_expert_gate = getattr(layer_mlp, 'shared_expert_gate', None)
        return shared_expert, shared_expert_gate
import torch.nn as nn

class BaseMoEAdapter:
    """
    MoE 模型适配器基类：屏蔽不同底层模型 (Qwen, Mixtral, DeepSeek 等) 的实现差异
    """
    
    def get_layers(self, model: nn.Module):
        """返回模型的 Transformer 层列表"""
        raise NotImplementedError
        
    def get_router(self, layer_mlp: nn.Module):
        """返回当前层的路由器 (Gate)"""
        raise NotImplementedError
        
    def get_expert_weights(self, layer_mlp: nn.Module, expert_id: int) -> dict:
        """
        提取特定专家的权重，返回一个字典。
        例如 Qwen: {"gate_up": tensor, "down": tensor}
        """
        raise NotImplementedError
        
    def expert_forward(self, hidden_states, weights_dict: dict):
        """
        利用提取出的权重字典，执行该模型专属的专家前向计算。
        """
        raise NotImplementedError

    def get_shared_expert_components(self, layer_mlp: nn.Module):
        """
        如果有共享专家 (Shared Expert, 如 DeepSeek, Qwen1.5-MoE)，则返回相关组件。
        返回 (shared_expert, shared_expert_gate)。如果没有则返回 (None, None)。
        """
        return None, None
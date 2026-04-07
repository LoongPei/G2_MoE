import torch
import torch.nn as nn
import torch.distributed as dist

class HybridMoEWrapper(nn.Module):
    """
    G2MoE 核心引擎：
    基于物理感知的拓扑路由。高度解耦，支持任意通过 Adapter 接入的 MoE 模型。
    内置了 Profiler 探针接口，可直接用于吞吐量和流量记录。
    """
    def __init__(self, orig_mlp, hub_ids, my_specialized_ids, global_specialized_map, adapter):
        super().__init__()
        self.adapter = adapter
        self.hub_ids = hub_ids
        self.my_specialized_ids = my_specialized_ids
        self.global_specialized_map = global_specialized_map
        
        self.gate = self.adapter.get_router(orig_mlp)
        self.shared_expert, self.shared_expert_gate = self.adapter.get_shared_expert_components(orig_mlp)
        
        # 🌟 动态权重容器：自动适配不同的模型架构结构 (比如 Qwen 的 2 个权重，或 Mixtral 的 3 个权重)
        sample_weights = self.adapter.get_expert_weights(orig_mlp, 0)
        weight_keys = sample_weights.keys()
        
        self.hub_weights = nn.ModuleDict({k: nn.ParameterDict() for k in weight_keys})
        for h_id in self.hub_ids:
            w_dict = self.adapter.get_expert_weights(orig_mlp, h_id)
            for k, v in w_dict.items():
                self.hub_weights[k][str(h_id)] = nn.Parameter(v)
                
        self.local_weights = nn.ModuleDict({k: nn.ParameterDict() for k in weight_keys})
        for s_id in self.my_specialized_ids:
            w_dict = self.adapter.get_expert_weights(orig_mlp, s_id)
            for k, v in w_dict.items():
                self.local_weights[k][str(s_id)] = nn.Parameter(v)

        # 挂载流量与耗时探针所需的基础变量
        self.current_comm_bytes = 0
        self.benchmark_comm_bytes = 0
        self.benchmark_tokens_bypassed = 0
        self.benchmark_tokens_sent = 0
        self.profiling_events = []

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        device = flat_hidden.device
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # 1. 路由选择
        router_logits, routing_weights, selected_experts = self.gate(flat_hidden)
        final_out = torch.zeros_like(flat_hidden)
        
        # 2. 共享专家 (Shared Expert)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(flat_hidden)
            if self.shared_expert_gate is not None:
                shared_out = shared_out * torch.sigmoid(self.shared_expert_gate(flat_hidden))
            final_out += shared_out

        # ==========================================
        # 赛道 A：Hub Experts (零通信开销，留在本地计算)
        # ==========================================
        for h_id in self.hub_ids:
            mask = (selected_experts == h_id)
            if not mask.any(): continue
            idx_x, idx_y = torch.where(mask)
            
            self.benchmark_tokens_bypassed += len(idx_x)
            
            tokens = flat_hidden[idx_x]
            w_dict = {k: self.hub_weights[k][str(h_id)] for k in self.hub_weights.keys()}
            expert_out = self.adapter.expert_forward(tokens, w_dict)
            
            final_out.index_add_(0, idx_x, (expert_out * routing_weights[idx_x, idx_y].unsqueeze(-1)).to(final_out.dtype))

        # ==========================================
        # 赛道 B：Specialized Experts (物理感知 All-To-All)
        # ==========================================
        send_buffers, send_eids, send_weights, local_indices = [[] for _ in range(world_size)], [[] for _ in range(world_size)], [[] for _ in range(world_size)], [[] for _ in range(world_size)]
        
        for target_gpu in range(world_size):
            for s_id in self.global_specialized_map[target_gpu]:
                mask = (selected_experts == s_id)
                if not mask.any(): continue
                idx_x, idx_y = torch.where(mask)
                send_buffers[target_gpu].append(flat_hidden[idx_x])
                send_eids[target_gpu].append(torch.full((len(idx_x),), s_id, dtype=torch.long, device=device))
                send_weights[target_gpu].append(routing_weights[idx_x, idx_y])
                local_indices[target_gpu].append(idx_x)

        send_tensors, send_eids_tensors, send_sizes = [], [], []
        for g in range(world_size):
            if send_buffers[g]:
                send_tensors.append(torch.cat(send_buffers[g]))
                send_eids_tensors.append(torch.cat(send_eids[g]))
                send_sizes.append(send_tensors[-1].size(0))
            else:
                send_tensors.append(torch.empty((0, hidden_dim), dtype=flat_hidden.dtype, device=device))
                send_eids_tensors.append(torch.empty((0,), dtype=torch.long, device=device))
                send_sizes.append(0)

        # 👉 流量探针统计 
        cross_card_send_sizes = [sz for g, sz in enumerate(send_sizes) if g != dist.get_rank()]
        self.current_comm_bytes = sum(cross_card_send_sizes) * hidden_dim * 2 
        self.benchmark_comm_bytes += self.current_comm_bytes
        self.benchmark_tokens_sent += sum(send_sizes)

        # ---------------- Dispatch ----------------
        send_sizes_tensor = torch.tensor(send_sizes, dtype=torch.long, device=device)
        recv_sizes_tensor = torch.empty(world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_sizes_tensor, send_sizes_tensor)
        recv_sizes = recv_sizes_tensor.tolist()
        
        recv_tensors = [torch.empty((sz, hidden_dim), dtype=flat_hidden.dtype, device=device) for sz in recv_sizes]
        recv_eids_tensors = [torch.empty(sz, dtype=torch.long, device=device) for sz in recv_sizes]
        
        dist.all_to_all(recv_tensors, send_tensors)
        dist.all_to_all(recv_eids_tensors, send_eids_tensors)

        # ---------------- Compute ----------------
        processed_tensors = []
        for recv_toks, recv_eids in zip(recv_tensors, recv_eids_tensors):
            if recv_toks.shape[0] == 0:
                processed_tensors.append(recv_toks)
                continue
                
            local_out = torch.zeros_like(recv_toks)
            for exp_id in torch.unique(recv_eids):
                e_id_str = str(exp_id.item())
                mask = (recv_eids == exp_id)
                current_state = recv_toks[mask]
                
                w_dict = {k: self.local_weights[k][e_id_str] for k in self.local_weights.keys()}
                expert_out = self.adapter.expert_forward(current_state, w_dict)
                local_out[mask] = expert_out.to(local_out.dtype)
                
            processed_tensors.append(local_out)

        # ---------------- Combine ----------------
        return_tensors = [torch.empty_like(t) for t in send_tensors]
        dist.all_to_all(return_tensors, processed_tensors)

        for g in range(world_size):
            if send_sizes[g] == 0: continue
            idx_offset = 0
            for w_list, orig_idx in zip(send_weights[g], local_indices[g]):
                chunk_size = len(orig_idx)
                chunk_res = return_tensors[g][idx_offset : idx_offset + chunk_size]
                chunk_res = (chunk_res * w_list.unsqueeze(-1)).to(final_out.dtype)
                final_out.index_add_(0, orig_idx, chunk_res)
                idx_offset += chunk_size

        return final_out.view(batch_size, seq_len, hidden_dim)
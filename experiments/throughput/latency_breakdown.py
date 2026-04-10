import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from g2moe.config import DEFAULT_MODEL_PATH, PLACEMENT_MAP_PATH, EVAL_RESULTS_DIR
from g2moe.utils.data import get_domain_tokens

# ==========================================
# 带毫秒级 CUDA 探针的特制 Wrapper
# ==========================================
class ProfilingHybridMoEWrapper(nn.Module):
    def __init__(self, orig_mlp, hub_ids, my_specialized_ids, global_specialized_map):
        super().__init__()
        self.hub_ids = hub_ids
        self.my_specialized_ids = my_specialized_ids
        self.global_specialized_map = global_specialized_map
        
        self.hub_gate_up, self.hub_down = nn.ParameterDict(), nn.ParameterDict()
        for h_id in self.hub_ids:
            self.hub_gate_up[str(h_id)] = nn.Parameter(orig_mlp.experts.gate_up_proj.data[h_id].clone())
            self.hub_down[str(h_id)] = nn.Parameter(orig_mlp.experts.down_proj.data[h_id].clone())

        self.local_gate_up, self.local_down = nn.ParameterDict(), nn.ParameterDict()
        for s_id in self.my_specialized_ids:
            self.local_gate_up[str(s_id)] = nn.Parameter(orig_mlp.experts.gate_up_proj.data[s_id].clone())
            self.local_down[str(s_id)] = nn.Parameter(orig_mlp.experts.down_proj.data[s_id].clone())

        self.gate = orig_mlp.gate
        if hasattr(orig_mlp, 'shared_expert'):
            self.shared_expert = orig_mlp.shared_expert
            self.shared_expert_gate = getattr(orig_mlp, 'shared_expert_gate', None)

        self.profiling_events = []

    def forward(self, hidden_states):
        evt_total_start = torch.cuda.Event(enable_timing=True)
        evt_total_end = torch.cuda.Event(enable_timing=True)
        evt_total_start.record()

        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        device = flat_hidden.device
        world_size = dist.get_world_size()
        
        router_logits, routing_weights, selected_experts = self.gate(flat_hidden)
        final_out = torch.zeros_like(flat_hidden)
        
        evt_comp1_start = torch.cuda.Event(enable_timing=True)
        evt_comp1_end = torch.cuda.Event(enable_timing=True)
        evt_comp1_start.record()
        
        if hasattr(self, 'shared_expert'):
            shared_out = self.shared_expert(flat_hidden)
            if hasattr(self, 'shared_expert_gate'):
                shared_out = shared_out * torch.sigmoid(self.shared_expert_gate(flat_hidden))
            final_out += shared_out

        for h_id in self.hub_ids:
            mask = (selected_experts == h_id)
            if not mask.any(): continue
            idx_x, idx_y = torch.where(mask)
            gate, up = F.linear(flat_hidden[idx_x], self.hub_gate_up[str(h_id)]).chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, self.hub_down[str(h_id)])
            final_out.index_add_(0, idx_x, (expert_out * routing_weights[idx_x, idx_y].unsqueeze(-1)).to(final_out.dtype))
        evt_comp1_end.record()

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

        evt_comm1_start = torch.cuda.Event(enable_timing=True)
        evt_comm1_end = torch.cuda.Event(enable_timing=True)
        evt_comm1_start.record()
        send_sizes_tensor = torch.tensor(send_sizes, dtype=torch.long, device=device)
        recv_sizes_tensor = torch.empty(world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_sizes_tensor, send_sizes_tensor)
        recv_sizes = recv_sizes_tensor.tolist()
        
        recv_tensors = [torch.empty((sz, hidden_dim), dtype=flat_hidden.dtype, device=device) for sz in recv_sizes]
        recv_eids_tensors = [torch.empty(sz, dtype=torch.long, device=device) for sz in recv_sizes]
        dist.all_to_all(recv_tensors, send_tensors)
        dist.all_to_all(recv_eids_tensors, send_eids_tensors)
        evt_comm1_end.record()

        evt_comp2_start = torch.cuda.Event(enable_timing=True)
        evt_comp2_end = torch.cuda.Event(enable_timing=True)
        evt_comp2_start.record()
        processed_tensors = []
        for recv_toks, recv_eids in zip(recv_tensors, recv_eids_tensors):
            if recv_toks.shape[0] == 0:
                processed_tensors.append(recv_toks)
                continue
            local_out = torch.zeros_like(recv_toks)
            for exp_id in torch.unique(recv_eids):
                e_id_str = str(exp_id.item())
                mask = (recv_eids == exp_id)
                gate, up = F.linear(recv_toks[mask], self.local_gate_up[e_id_str]).chunk(2, dim=-1)
                local_out[mask] = F.linear(F.silu(gate) * up, self.local_down[e_id_str]).to(local_out.dtype)
            processed_tensors.append(local_out)
        evt_comp2_end.record()

        evt_comm2_start = torch.cuda.Event(enable_timing=True)
        evt_comm2_end = torch.cuda.Event(enable_timing=True)
        evt_comm2_start.record()
        return_tensors = [torch.empty_like(t) for t in send_tensors]
        dist.all_to_all(return_tensors, processed_tensors)
        evt_comm2_end.record()

        for g in range(world_size):
            if send_sizes[g] == 0: continue
            idx_offset = 0
            for w_list, orig_idx in zip(send_weights[g], local_indices[g]):
                chunk_size = len(orig_idx)
                chunk_res = return_tensors[g][idx_offset : idx_offset + chunk_size]
                chunk_res = (chunk_res * w_list.unsqueeze(-1)).to(final_out.dtype)
                final_out.index_add_(0, orig_idx, chunk_res)
                idx_offset += chunk_size

        evt_total_end.record()

        self.profiling_events.append({
            "total": (evt_total_start, evt_total_end),
            "comp": [(evt_comp1_start, evt_comp1_end), (evt_comp2_start, evt_comp2_end)],
            "comm": [(evt_comm1_start, evt_comm1_end), (evt_comm2_start, evt_comm2_end)]
        })

        return final_out.view(batch_size, seq_len, hidden_dim)

def load_model(mode, rank, world_size):
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True)
    with open(PLACEMENT_MAP_PATH, "r", encoding="utf-8") as f:
        placement_map = json.load(f)
        
    for l_idx, layer in enumerate(model.model.layers):
        layer_map = placement_map[f"layer_{l_idx}"]
        if mode == "ours":
            hub_ids = layer_map["shared_hubs_replicated_to_all_gpus"]
            global_map = [layer_map["gpu_partitions"][f"gpu_{g}"]["experts"] for g in range(world_size)]
        else:
            hub_ids = []
            exp_per_gpu = 60 // world_size
            global_map = [list(range(g * exp_per_gpu, (g + 1) * exp_per_gpu)) for g in range(world_size)]
            
        orig_mlp = layer.mlp if hasattr(layer, 'mlp') else layer
        new_mlp = ProfilingHybridMoEWrapper(orig_mlp, hub_ids, global_map[rank], global_map)
        
        if hasattr(layer, 'mlp'):
            layer.mlp = new_mlp
        else:
            setattr(layer, 'mlp', new_mlp)
        
    model.to(torch.device(f"cuda:{rank}"))
    return model

def run_matrix_profiling(mode, seq_lens, global_batches, exp_name):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    model = load_model(mode, rank, world_size)
    model.eval()

    DOMAINS = ["code", "math", "wiki"]
    STEPS = 10
    
    results_matrix = {}

    for domain in DOMAINS:
        results_matrix[domain] = {}
        
        # 计算所需的最大 token 数量
        max_local_bs = max(global_batches) // world_size
        if max_local_bs == 0:
            max_local_bs = 1
            
        max_tokens_needed = max_local_bs * max(seq_lens) * (STEPS + 2) * world_size
        
        if rank == 0: 
            print(f"\n=========================================\n📥 正在从本地极速加载 [{domain.upper()}] 领域语料...\n=========================================")
            
        tokens = get_domain_tokens(tokenizer, domain, max_tokens_needed)
        
        for seq_len in seq_lens:
            results_matrix[domain][f"seq_{seq_len}"] = {}
            if rank == 0: print(f" 👉 开始测试 Sequence Length: {seq_len}")
            
            for gb in global_batches:
                bs = gb // world_size
                if bs == 0:
                    if rank == 0: print(f"  ⚠️ Global batch {gb} 小于 GPU 数量 {world_size}，跳过该配置。")
                    continue
                    
                if rank == 0: print(f"  ➤ 测试 [Global Batch = {gb} | Local Batch = {bs} | Seq = {seq_len}] ...")
                torch.cuda.empty_cache()
                oom_detected = torch.tensor([0], dtype=torch.int, device=f"cuda:{rank}")
                
                try:
                    for layer in model.model.layers: getattr(layer, 'mlp', layer).profiling_events.clear()
                    
                    # 预热
                    dummy_start = rank * bs * seq_len
                    dummy_input = tokens[dummy_start : dummy_start + bs * seq_len].view(bs, seq_len).to(f"cuda:{rank}")
                    with torch.no_grad():
                        for _ in range(2): model(dummy_input)
                    torch.cuda.synchronize()
                    
                    for layer in model.model.layers: getattr(layer, 'mlp', layer).profiling_events.clear()

                    # 正式测速
                    base_offset = 2 * bs * seq_len * world_size
                    for i in range(STEPS):
                        step_start = base_offset + (i * bs * seq_len * world_size)
                        rank_start = step_start + (rank * bs * seq_len)
                        chunk = tokens[rank_start : rank_start + bs * seq_len].view(bs, seq_len).to(f"cuda:{rank}")
                        with torch.no_grad():
                            model(chunk)

                    torch.cuda.synchronize()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_detected[0] = 1
                        torch.cuda.empty_cache()
                    else:
                        raise e 

                dist.all_reduce(oom_detected, op=dist.ReduceOp.MAX)

                if oom_detected.item() > 0:
                    if rank == 0:
                        print(f"  ⚠️ [Global Batch = {gb} | Seq = {seq_len}] 触发显存极限 (OOM)！已启动全局熔断，跳入下一配置。")
                    torch.cuda.empty_cache()
                    break

                dist.barrier() 
                
                total_ms_sum, comp_ms_sum, comm_ms_sum = 0.0, 0.0, 0.0
                total_layers_tracked = 0
                for layer in model.model.layers:
                    mlp = getattr(layer, 'mlp', layer)
                    for evts in mlp.profiling_events:
                        total_ms_sum += evts["total"][0].elapsed_time(evts["total"][1])
                        comp_ms_sum += sum(s.elapsed_time(e) for s, e in evts["comp"])
                        comm_ms_sum += sum(s.elapsed_time(e) for s, e in evts["comm"])
                        total_layers_tracked += 1

                avg_total = total_ms_sum / total_layers_tracked
                avg_comp = comp_ms_sum / total_layers_tracked
                avg_comm = comm_ms_sum / total_layers_tracked
                avg_route = avg_total - avg_comp - avg_comm
                
                results_matrix[domain][f"seq_{seq_len}"][f"global_bs_{gb}"] = {
                    "compute_ms": round(avg_comp, 3),
                    "comm_ms": round(avg_comm, 3),
                    "route_ms": round(avg_route, 3),
                    "total_ms": round(avg_total, 3)
                }

                if rank == 0:
                    # 使用 exp_name 创建对应的子文件夹
                    out_dir = EVAL_RESULTS_DIR / "latency_breakdown" / exp_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"matrix_{mode}_breakdown.json"
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(results_matrix, f, indent=2)
            
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["standard_ep", "ours"], required=True)
    parser.add_argument("--seq_lens", type=int, nargs='+', default=[1024], help="List of Sequence Lengths to test. e.g. --seq_lens 512 1024 2048")
    parser.add_argument("--global_batches", type=int, nargs='+', default=[8, 16, 32, 64], help="List of Global Batch Sizes to test. e.g. --global_batches 16 32")
    parser.add_argument("--exp_name", type=str, default="default_exp", help="Name of the experiment, used as subfolder for saving results.")
    args = parser.parse_args()
    
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    run_matrix_profiling(args.mode, args.seq_lens, args.global_batches, args.exp_name)
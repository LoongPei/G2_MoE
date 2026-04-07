import numpy as np
import gurobipy as gp
from gurobipy import GRB

class G2MoEPlacementSolver:
    """
    G2MoE 3D 物理拓扑感知求解器 (基于 Gurobi MIQP)
    """
    def __init__(self, num_gpus=4, num_layers=24, num_experts=60, time_limit=6000, mip_gap=0.03):
        self.num_gpus = num_gpus
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.time_limit = time_limit
        self.mip_gap = mip_gap

    def solve(self, D, pmi_matrices, markov_matrices, hub_data, specialized_experts, expert_loads, dynamic_lambda):
        """
        执行 MIQP 求解，返回解析好的 placement_map 字典。
        """
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 1)
        env.start()
        model = gp.Model("MoE_3D_Physical_Placement", env=env)
        
        model.setParam(GRB.Param.TimeLimit, self.time_limit)  
        model.setParam(GRB.Param.MIPGap, self.mip_gap)    

        # 1. 定义决策变量 X[l][i][k]
        X = {}
        for l in range(self.num_layers):
            X[l] = {}
            for i in specialized_experts[l]:
                X[l][i] = {k: model.addVar(vtype=GRB.BINARY, name=f"X_{l}_{i}_{k}") for k in range(self.num_gpus)}

        # 2. 添加约束
        for l in range(self.num_layers):
            exact_cap = len(specialized_experts[l]) // self.num_gpus
            for i in specialized_experts[l]:
                # 唯一性约束: 每个专家只能放在一个 GPU
                model.addConstr(gp.quicksum(X[l][i][k] for k in range(self.num_gpus)) == 1)
            for k in range(self.num_gpus):
                # 绝对容量均衡约束: 每个 GPU 必须分到严格相等的专家数
                model.addConstr(gp.quicksum(X[l][i][k] for i in specialized_experts[l]) == exact_cap)

        print("🧩 [Solver] 正在构建 3D 时空二次目标函数...")
        obj = gp.QuadExpr()
        
        # 层内协同代价 (PMI)
        for l in range(self.num_layers):
            specs, pmi = specialized_experts[l], pmi_matrices[l]
            max_pmi = np.max(pmi) + 1e-9
            for idx_1, i in enumerate(specs):
                for j in specs[idx_1 + 1:]:
                    w = pmi[i, j] / max_pmi
                    if w > 0.01: 
                        for k in range(self.num_gpus):
                            for m in range(self.num_gpus):
                                if D[k, m] > 0: 
                                    obj += w * D[k, m] * X[l][i][k] * X[l][j][m]

        # 层间转移代价 (Markov)
        for l in range(self.num_layers - 1):
            specs_curr, specs_next = specialized_experts[l], specialized_experts[l+1]
            markov, max_markov = markov_matrices[l], np.max(markov_matrices[l]) + 1e-9
            for i in specs_curr:
                for j in specs_next:
                    w = markov[i, j] / max_markov
                    if w > 0.05: 
                        for k in range(self.num_gpus):
                            for m in range(self.num_gpus):
                                if D[k, m] > 0:
                                    obj += dynamic_lambda * w * D[k, m] * X[l][i][k] * X[l+1][j][m]

        model.setObjective(obj, GRB.MINIMIZE)
        
        print("\n🔥 [Solver] Gurobi 发动机点火！开始求解全局物理最优调度...")
        model.optimize()

        # 3. 解析结果并返回
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            final_placement_map = {}
            for l in range(self.num_layers):
                hubs = hub_data[f"layer_{l}"]["hub_experts"][:4]
                layer_placement = {
                    "shared_hubs_replicated_to_all_gpus": hubs,
                    "gpu_partitions": {f"gpu_{k}": {"experts": [], "total_load": 0} for k in range(self.num_gpus)}
                }
                for i in specialized_experts[l]:
                    for k in range(self.num_gpus):
                        if X[l][i][k].X > 0.5: 
                            layer_placement["gpu_partitions"][f"gpu_{k}"]["experts"].append(i)
                            layer_placement["gpu_partitions"][f"gpu_{k}"]["total_load"] += float(expert_loads[l][i])
                            break
                final_placement_map[f"layer_{l}"] = layer_placement
            return final_placement_map
        else:
            print("❌ [Solver] 求解失败，可能存在约束冲突。")
            return None
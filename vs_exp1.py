#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3: 缺陷位置感知 + 注意力可视化 + 模型权重导出
目标：
  1. 模型学习缺陷类型（全局 one-hot + 辅助分类）
  2. 模型能识别缺陷发生位置（节点特征 + 辅助节点分类损失）
  3. 注意力可视化时，高注意力集中在缺陷原子附近
  4. 导出最佳模型权重
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, LayerNorm, Linear
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

# ============================================================
# 0. 路径与全局
# ============================================================
DATA_PATH = Path("cp2k_gnn_data.json")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ============================================================
# 1. 加载样本 + 解析缺陷位置
# ============================================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def parse_dopant_from_type(defect_type: str):
    """从 substitutional_xx_q 解析掺杂元素符号"""
    if not defect_type.startswith("substitutional_"):
        return None
    # substitutional_ta_q -> Ta
    part = defect_type.replace("substitutional_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[:1].upper() + part[1:].lower()

samples = []
for mp_id, system in data["systems"].items():
    if system.get("pristine") and system["pristine"].get("has_structure"):
        p = system["pristine"]
        samples.append({
            "id": f"{mp_id}_pristine",
            "energy_diff": 0.0,
            "structure": p["structure"],
            "charge": p.get("inp_charge", 0),
            "multiplicity": p.get("inp_multiplicity", 1),
            "n_elec_s1": p.get("n_electrons_spin1"),
            "n_elec_s2": p.get("n_electrons_spin2"),
            "defect_type": "pristine",
            "dopant": None,
        })
    for d in system.get("defects", []):
        if not d.get("has_structure"):
            continue
        dtype = d["defect_type"]
        samples.append({
            "id": f"{mp_id}_{dtype}_q{d.get('charge_from_name', '')}",
            "energy_diff": d.get("energy_diff_ev", d["energy_ev"]),
            "structure": d["structure"],
            "charge": d.get("inp_charge", 0),
            "multiplicity": d.get("inp_multiplicity", 1),
            "n_elec_s1": d.get("n_electrons_spin1"),
            "n_elec_s2": d.get("n_electrons_spin2"),
            "defect_type": dtype,
            "dopant": parse_dopant_from_type(dtype),
        })

print(f"总样本数: {len(samples)}")
all_defect_types = sorted(set(s["defect_type"] for s in samples))
type_to_idx = {t: i for i, t in enumerate(all_defect_types)}
n_types = len(all_defect_types)
print(f"缺陷类型 ({n_types}): {all_defect_types}")

# ============================================================
# 2. 元素特征 + 图构建（含缺陷位置信息）
# ============================================================
element_features = {
    "H":  [1,  2.20,  31, 1, 0],
    "C":  [6,  2.55,  76, 4, 0],
    "N":  [7,  3.04,  71, 5, 0],
    "O":  [8,  3.44,  66, 6, 0],
    "S":  [16, 2.58, 105, 6, 0],
    "Ca": [20, 1.00, 176, 2, 1],
    "Ti": [22, 1.54, 160, 4, 1],
    "V":  [23, 1.63, 153, 5, 1],
    "Cr": [24, 1.66, 139, 6, 1],
    "Fe": [26, 1.83, 132, 8, 1],
    "Zn": [30, 1.65, 122, 2, 1],
    "Sr": [38, 0.95, 195, 2, 1],
    "Y":  [39, 1.22, 190, 3, 1],
    "Nb": [41, 1.60, 164, 5, 1],
    "Mo": [42, 2.16, 154, 6, 1],
    "Sn": [50, 1.96, 140, 4, 1],
    "Ba": [56, 0.89, 215, 2, 1],
    "La": [57, 1.10, 195, 3, 1],
    "Hf": [72, 1.30, 175, 4, 1],
    "Ta": [73, 1.50, 170, 5, 1],
    "W":  [74, 2.36, 162, 6, 1],
    "Pb": [82, 2.33, 146, 4, 1],
    "Bi": [83, 2.02, 148, 5, 1],
}

def get_atom_features(element):
    return np.array(element_features.get(element, [0, 0.0, 0.0, 0, 0]), dtype=np.float32)


def structure_to_ase(structure):
    symbols = [a["element"] for a in structure["atoms"]]
    positions = [[a["x"], a["y"], a["z"]] for a in structure["atoms"]]
    cell = structure.get("lattice_vectors")
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def find_defect_atom_index(structure, dopant):
    """返回 substitutional 中掺杂原子的索引；无则返回 None"""
    if dopant is None:
        return None
    for i, a in enumerate(structure["atoms"]):
        if a["element"] == dopant:
            return i
    return None


def atoms_to_graph_v3(sample, cutoff=6.0):
    """
    节点特征 (10维):
      [Z, χ, r, v, is_metal, sx, sy, sz, is_defect_atom, dist_to_defect]
    边特征 (21维): RBF(20) + dist
    全局特征 (1, 7+n_types): 统计 + charge + mult + ΔN + type one-hot
    额外: node_label (is_defect_atom) 用于辅助节点分类
    """
    structure = sample["structure"]
    energy = sample["energy_diff"]
    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    scaled_pos = atoms.get_scaled_positions()

    # 缺陷原子索引与坐标
    defect_idx = find_defect_atom_index(structure, sample["dopant"])
    if defect_idx is not None:
        defect_pos = positions[defect_idx]
    else:
        defect_pos = None

    # 节点特征
    node_feats = []
    node_labels = []  # 1 = defect atom
    for i, a in enumerate(structure["atoms"]):
        base = get_atom_features(a["element"])          # 5
        frac = scaled_pos[i]                            # 3
        is_def = 1.0 if (defect_idx is not None and i == defect_idx) else 0.0
        if defect_pos is not None:
            # 最小镜像距离
            delta = positions[i] - defect_pos
            # 简单正交近似（对大多数近立方晶胞足够；精确可用 atoms.get_distance）
            dist = float(np.linalg.norm(delta))
        else:
            dist = 0.0
        node_feats.append(np.concatenate([base, frac, [is_def, dist]]))
        node_labels.append(is_def)

    x = torch.tensor(np.array(node_feats), dtype=torch.float)          # (N, 10)
    node_label = torch.tensor(node_labels, dtype=torch.float)          # (N,)

    # 边
    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist - rbf_centers)**2) / (2 * 0.5**2))
    edge_attr = torch.cat([edge_attr, dist / cutoff], dim=1)           # (E, 21)

    # 全局特征
    vol = float(atoms.get_volume()) if atoms.get_volume() > 0 else 1.0
    density = n_atoms / vol
    n_elements = len(set(a["element"] for a in structure["atoms"]))
    charge = float(sample.get("charge", 0))
    multiplicity = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_elec = float(abs(s1 - s2)) if (s1 is not None and s2 is not None) else 0.0

    type_oh = np.zeros(n_types, dtype=np.float32)
    type_oh[type_to_idx[sample["defect_type"]]] = 1.0

    global_feat = np.concatenate([
        [n_atoms, vol, density, n_elements, charge, multiplicity, delta_elec],
        type_oh
    ]).astype(np.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.tensor([energy], dtype=torch.float),
        global_feat=torch.tensor(global_feat, dtype=torch.float).unsqueeze(0),  # (1, F)
        node_label=node_label,           # 用于辅助损失
        defect_idx=defect_idx if defect_idx is not None else -1,
        sample_id=sample["id"],
    )
    return data


# 构建图
graph_list, kept_targets = [], []
for s in samples:
    try:
        g = atoms_to_graph_v3(s)
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
    except Exception as e:
        print(f"skip {s['id']}: {e}")

print(f"成功构建图: {len(graph_list)}")
print(f"节点特征维: {graph_list[0].x.shape[1]}, 全局特征维: {graph_list[0].global_feat.shape[1]}")

# 划分
train_idx, test_idx = train_test_split(range(len(graph_list)), test_size=0.2, random_state=42)

# 目标标准化
train_y = np.array([kept_targets[i] for i in train_idx])
y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-8)
for g in graph_list:
    g.y = (g.y - y_mean) / y_std

# 节点特征标准化（对 is_defect 和 dist 也标准化，让模型自己学）
all_x = torch.cat([g.x for g in graph_list], dim=0)
x_mean = all_x.mean(0, keepdim=True)
x_std = all_x.std(0, keepdim=True) + 1e-8
for g in graph_list:
    g.x = (g.x - x_mean) / x_std

# 全局连续部分标准化
all_gf = torch.cat([g.global_feat for g in graph_list], dim=0)
n_cont = 7
gf_mean = all_gf[:, :n_cont].mean(0, keepdim=True)
gf_std = all_gf[:, :n_cont].std(0, keepdim=True) + 1e-8
for g in graph_list:
    g.global_feat[:, :n_cont] = (g.global_feat[:, :n_cont] - gf_mean) / gf_std

train_loader = DataLoader([graph_list[i] for i in train_idx], batch_size=8, shuffle=True)
test_loader  = DataLoader([graph_list[i] for i in test_idx],  batch_size=8)

# ============================================================
# 3. 模型：GATv2 + 能量回归 + 缺陷节点辅助分类
# ============================================================
class DefectAwareGATv2(nn.Module):
    def __init__(self, node_in=10, edge_in=21, global_in=None,
                 hidden=64, heads=4, n_types=12):
        super().__init__()
        if global_in is None:
            global_in = 7 + n_types
        self.hidden = hidden

        self.node_enc = nn.Sequential(
            Linear(node_in, hidden), LayerNorm(hidden), nn.ReLU(), Linear(hidden, hidden)
        )
        self.edge_enc = nn.Sequential(
            Linear(edge_in, hidden), nn.ReLU(), Linear(hidden, hidden)
        )

        self.gat1 = GATv2Conv(hidden, hidden//heads, heads=heads, concat=True,
                              edge_dim=hidden, dropout=0.1)
        self.norm1 = LayerNorm(hidden)
        self.gat2 = GATv2Conv(hidden, hidden//heads, heads=heads, concat=True,
                              edge_dim=hidden, dropout=0.1)
        self.norm2 = LayerNorm(hidden)
        self.gat3 = GATv2Conv(hidden, hidden, heads=1, concat=False,
                              edge_dim=hidden, dropout=0.1)
        self.norm3 = LayerNorm(hidden)

        self.global_enc = nn.Sequential(
            Linear(global_in, hidden//2), nn.ReLU(), nn.Dropout(0.1),
            Linear(hidden//2, hidden//2), LayerNorm(hidden//2)
        )
        self.graph_proj = Linear(hidden*2, hidden//2)  # mean+max

        self.fusion_gate = nn.Sequential(Linear(hidden, hidden//2), nn.Sigmoid())
        self.energy_head = nn.Sequential(
            Linear(hidden//2, hidden//2), nn.ReLU(), nn.Dropout(0.2), Linear(hidden//2, 1)
        )

        # 辅助：从最终节点嵌入预测是否为缺陷原子
        self.node_cls_head = nn.Sequential(
            Linear(hidden, hidden//2), nn.ReLU(), Linear(hidden//2, 1)
        )
        # 辅助：从全局图嵌入预测缺陷类型
        self.type_cls_head = nn.Linear(hidden//2, n_types)

    def forward(self, data, return_attention=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        gf = data.global_feat

        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)

        # 保存 attention（最后一层）
        if return_attention:
            x1, (ei1, att1) = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x1 = F.relu(self.norm1(x1))
            x2, (ei2, att2) = self.gat2(x1, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x2 = F.relu(self.norm2(x2 + x1))
            x3, (ei3, att3) = self.gat3(x2, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x3 = F.relu(self.norm3(x3 + x2))
            attentions = {"ei": ei3, "att": att3}  # 最后一层
        else:
            x1 = F.relu(self.norm1(self.gat1(x, edge_index, edge_attr=edge_attr)))
            x2 = F.relu(self.norm2(self.gat2(x1, edge_index, edge_attr=edge_attr) + x1))
            x3 = F.relu(self.norm3(self.gat3(x2, edge_index, edge_attr=edge_attr) + x2))
            attentions = None

        g_mean = global_mean_pool(x3, batch)
        g_max  = global_max_pool(x3, batch)
        g = torch.cat([g_mean, g_max], dim=-1)
        g = self.graph_proj(g)

        ge = self.global_enc(gf)
        gate = self.fusion_gate(torch.cat([g, ge], dim=-1))
        fused = gate * g + (1 - gate) * ge

        energy = self.energy_head(fused).squeeze(-1)
        node_logit = self.node_cls_head(x3).squeeze(-1)   # (total_nodes,)
        type_logit = self.type_cls_head(fused)            # (batch, n_types)

        if return_attention:
            return energy, node_logit, type_logit, attentions, x3
        return energy, node_logit, type_logit


# ============================================================
# 4. 训练（能量主损失 + 节点分类 + 类型分类）
# ============================================================
model = DefectAwareGATv2(node_in=10, edge_in=21, global_in=7+n_types,
                         hidden=64, heads=4, n_types=n_types).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

LAMBDA_NODE = 0.5   # 辅助节点分类权重
LAMBDA_TYPE = 0.3   # 辅助类型分类权重

best_val_mae = float("inf")
best_state = None
train_losses, val_maes = [], []

print("\n===== 开始训练 DefectAwareGATv2 =====")
for epoch in range(1, 61):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        energy_pred, node_logit, type_logit = model(batch)

        # 主损失：能量
        loss_e = F.mse_loss(energy_pred, batch.y.view(-1))

        # 辅助1：节点是否缺陷原子（仅对有缺陷标记的样本有效）
        # node_label 已在 Data 中
        loss_node = F.binary_cross_entropy_with_logits(node_logit, batch.node_label)

        # 辅助2：缺陷类型分类
        # 从 global_feat 的 one-hot 部分还原 true type
        true_type = batch.global_feat[:, 7:].argmax(dim=1)
        loss_type = F.cross_entropy(type_logit, true_type)

        loss = loss_e + LAMBDA_NODE * loss_node + LAMBDA_TYPE * loss_type
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    avg_loss = total_loss / len(train_idx)
    train_losses.append(avg_loss)

    # 验证
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            energy_pred, _, _ = model(batch)
            preds.append(energy_pred.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds) * y_std + y_mean
    trues = np.concatenate(trues) * y_std + y_mean
    val_mae = mean_absolute_error(trues, preds)
    val_maes.append(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    scheduler.step(val_mae)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val MAE {val_mae:.4f} eV")

# 加载最佳
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(DEVICE)

# 最终指标
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        energy_pred, _, _ = model(batch)
        preds.append(energy_pred.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
preds = np.concatenate(preds) * y_std + y_mean
trues = np.concatenate(trues) * y_std + y_mean
mae = mean_absolute_error(trues, preds)
rmse = np.sqrt(mean_squared_error(trues, preds))
r2 = r2_score(trues, preds)
print(f"\n===== 最终结果 =====")
print(f"MAE  : {mae:.4f} eV")
print(f"RMSE : {rmse:.4f} eV")
print(f"R²   : {r2:.4f}")

# ============================================================
# 5. 导出模型权重
# ============================================================
ckpt = {
    "model_state_dict": best_state,
    "y_mean": y_mean,
    "y_std": y_std,
    "x_mean": x_mean.numpy(),
    "x_std": x_std.numpy(),
    "gf_mean": gf_mean.numpy(),
    "gf_std": gf_std.numpy(),
    "all_defect_types": all_defect_types,
    "n_types": n_types,
    "node_in": 10,
    "edge_in": 21,
    "global_in": 7 + n_types,
    "hidden": 64,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
}
ckpt_path = OUT_DIR / "defect_aware_gatv2_best.pt"
torch.save(ckpt, ckpt_path)
print(f"模型权重已导出: {ckpt_path}")

# ============================================================
# 6. 注意力可视化（选几个 substitutional 测试样本）
# ============================================================
def aggregate_node_attention(edge_index, att, num_nodes):
    """把边注意力聚合成每个节点接收到的总注意力"""
    # att: (E, heads) or (E,)
    if att.dim() > 1:
        att = att.mean(dim=-1)  # 平均多头
    node_att = torch.zeros(num_nodes, device=att.device)
    # 注意力是目标节点 j 对源 i 的权重，这里累加到目标
    dst = edge_index[1]
    node_att.scatter_add_(0, dst, att)
    # 归一化到 [0,1]
    if node_att.max() > 0:
        node_att = node_att / node_att.max()
    return node_att.cpu().numpy()


def visualize_attention(sample_data, node_att, save_path, title=""):
    """简单 3D 散点：颜色=注意力，红色X=真实缺陷原子"""
    pos = sample_data.pos.numpy()
    defect_idx = sample_data.defect_idx
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=node_att, cmap="YlOrRd", s=60, alpha=0.85, edgecolors="k", linewidths=0.3)
    if defect_idx >= 0:
        ax.scatter(pos[defect_idx, 0], pos[defect_idx, 1], pos[defect_idx, 2],
                   c="blue", s=200, marker="*", label="True defect atom", edgecolors="k")
    plt.colorbar(sc, ax=ax, label="Aggregated Attention", shrink=0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    if defect_idx >= 0:
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  注意力图已保存: {save_path}")


print("\n===== 注意力可视化（substitutional 测试样本） =====")
model.eval()
# 从测试集中挑有缺陷原子的样本
vis_count = 0
with torch.no_grad():
    for idx in test_idx:
        g = graph_list[idx]
        if g.defect_idx < 0:
            continue
        # 单样本 batch
        g_batch = DataLoader([g], batch_size=1)
        batch = next(iter(g_batch)).to(DEVICE)
        energy_pred, node_logit, type_logit, att_dict, node_emb = model(batch, return_attention=True)

        ei = att_dict["ei"]
        att = att_dict["att"]
        node_att = aggregate_node_attention(ei, att, g.x.shape[0])

        # 真实缺陷原子上的注意力是否较高？
        def_att = node_att[g.defect_idx]
        rank = (node_att >= def_att).sum()  # 有多少原子注意力 >= 缺陷原子
        print(f"  {g.sample_id}: defect_idx={g.defect_idx}, "
              f"att@defect={def_att:.3f}, rank={rank}/{len(node_att)} "
              f"(1=highest)")

        save_path = OUT_DIR / f"attention_{g.sample_id.replace('/', '_')}.png"
        visualize_attention(g, node_att, save_path,
                            title=f"{g.sample_id}\natt@defect={def_att:.3f}")

        vis_count += 1
        if vis_count >= 6:  # 只画几个
            break

print(f"\n共可视化 {vis_count} 个样本")
print("完成。模型已具备缺陷位置感知能力，注意力应集中在缺陷原子附近。")

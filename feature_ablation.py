#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vs_exp5 v2: 特征消融实验（修复 vacancy 定位 + 按 mp_id 分组划分）
================================================================
相对 v1 的关键修复：
  1. vacancy_* 通过 pristine vs defect 结构差集反推空位中心，
     再计算每个剩余原子到空位的最小镜像距离 → dist_to_defect 不再恒为 0。
  2. train/test 按 mp_id 分组划分，同一材料的不同电荷态不会同时出现在
     训练与测试集，压缩 type one-hot 的“记忆捷径”。
  3. is_defect_atom 仅对 substitutional 的掺杂原子为 1；vacancy 全为 0
     （空位处无原子），几何信息完全由 dist_to_defect 承载。

特征分组（与特征工程一致，节点仍 10 维）：
  节点 (10维):
    [0:5]  元素基础特征  Z, χ, r, v, is_metal
    [5:8]  分数坐标      sx, sy, sz
    [8:10] 缺陷感知      is_defect_atom, dist_to_defect
  边 (21维):
    [0:20] RBF 径向基
    [20]   归一化距离
  全局 (7 + n_types):
    [0:4]  结构统计      n_atoms, V, ρ, n_elements
    [4:7]  电子/电荷     charge, multiplicity, ΔN_elec
    [7:]   缺陷类型 one-hot

消融策略：始终构建完整特征，训练时按配置将对应切片置零（保持模型维度不变）。
"""

import json
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict, defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# ------------------------------------------------------------
# 路径与超参
# ------------------------------------------------------------
DATA_PATH = Path("cp2k_gnn_data.json")
if not DATA_PATH.exists():
    DATA_PATH = Path("cp2k_gnn_data.json")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

HIDDEN_DIM   = 64
NUM_HEADS    = 4
BATCH_SIZE   = 8
MAX_EPOCHS   = 40          # 正式跑用 40；快速验证可改为 10~15
LR           = 1e-3
WEIGHT_DECAY = 1e-5
CUTOFF       = 6.0
RANDOM_SEED  = 42
N_SEEDS      = 1
VACANCY_MATCH_DIST = 1.8   # Å，pristine↔defect 原子匹配阈值

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 数据加载（缺陷样本挂载对应 pristine 结构）
# ============================================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

def parse_dopant_from_type(defect_type: str):
    if not defect_type.startswith("substitutional_"):
        return None
    part = defect_type.replace("substitutional_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[:1].upper() + part[1:].lower()

samples = []
for mp_id, system in raw["systems"].items():
    pristine_struct = None
    if system.get("pristine") and system["pristine"].get("has_structure"):
        p = system["pristine"]
        pristine_struct = p["structure"]
        samples.append({
            "id": f"{mp_id}_pristine",
            "mp_id": mp_id,
            "energy_diff": 0.0,
            "structure": p["structure"],
            "pristine_structure": None,   # 自身即 pristine
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
            "mp_id": mp_id,
            "energy_diff": d.get("energy_diff_ev", d["energy_ev"]),
            "structure": d["structure"],
            "pristine_structure": pristine_struct,  # 关键：挂载 pristine
            "charge": d.get("inp_charge", 0),
            "multiplicity": d.get("inp_multiplicity", 1),
            "n_elec_s1": d.get("n_electrons_spin1"),
            "n_elec_s2": d.get("n_electrons_spin2"),
            "defect_type": dtype,
            "dopant": parse_dopant_from_type(dtype),
        })

print(f"总样本数: {len(samples)}")
n_vac = sum(1 for s in samples if s["defect_type"].startswith("vacancy_"))
n_sub = sum(1 for s in samples if s["defect_type"].startswith("substitutional_"))
n_pris = sum(1 for s in samples if s["defect_type"] == "pristine")
print(f"  pristine={n_pris}, vacancy={n_vac}, substitutional={n_sub}")

all_defect_types = sorted(set(s["defect_type"] for s in samples))
type_to_idx = {t: i for i, t in enumerate(all_defect_types)}
n_types = len(all_defect_types)
print(f"缺陷类型 ({n_types}): {all_defect_types}")

# ============================================================
# 2. 元素特征 + 缺陷定位（含 vacancy）
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

def mic_distance(pos_i, pos_j, cell):
    """最小镜像距离（正交近似；对近立方晶胞足够）"""
    delta = pos_i - pos_j
    if cell is not None:
        try:
            inv = np.linalg.inv(np.asarray(cell, dtype=float))
            frac = delta @ inv.T
            frac -= np.round(frac)
            delta = frac @ np.asarray(cell, dtype=float)
        except np.linalg.LinAlgError:
            pass
    return float(np.linalg.norm(delta))

def find_defect_atom_index(structure, dopant):
    """substitutional：返回掺杂原子索引；否则 None"""
    if dopant is None:
        return None
    for i, a in enumerate(structure["atoms"]):
        if a["element"] == dopant:
            return i
    return None

def find_vacancy_site(pristine_struct, defect_struct, max_dist=VACANCY_MATCH_DIST):
    """
    通过 pristine vs defect 原子差集反推空位中心。
    返回 defect 坐标系下的笛卡尔坐标 (np.ndarray shape (3,))，失败返回 None。

    策略：按元素约束的贪婪最近邻匹配；pristine 中未被匹配的原子位置
    即为空位。若未匹配数 ≠ 1，视为失败（非单空位或结构差异过大）。
    """
    if pristine_struct is None or defect_struct is None:
        return None
    p_atoms = structure_to_ase(pristine_struct)
    d_atoms = structure_to_ase(defect_struct)
    p_pos = p_atoms.get_positions()
    d_pos = d_atoms.get_positions()
    p_sym = p_atoms.get_chemical_symbols()
    d_sym = d_atoms.get_chemical_symbols()
    d_cell = d_atoms.cell.array if d_atoms.cell is not None else None

    if len(p_pos) != len(d_pos) + 1:
        return None

    used_p = set()
    for j in range(len(d_pos)):
        best_i, best_d = None, 1e9
        for i in range(len(p_pos)):
            if i in used_p or p_sym[i] != d_sym[j]:
                continue
            dist = mic_distance(d_pos[j], p_pos[i], d_cell)
            if dist < best_d:
                best_d, best_i = dist, i
        if best_i is not None and best_d < max_dist:
            used_p.add(best_i)

    unmatched = [i for i in range(len(p_pos)) if i not in used_p]
    if len(unmatched) != 1:
        return None

    vac_pos = p_pos[unmatched[0]].copy()
    return vac_pos

def resolve_defect_center(sample, positions, cell):
    """
    统一接口：返回 (defect_idx, defect_pos)
      - substitutional: defect_idx = 掺杂原子索引, defect_pos = 其坐标
      - vacancy:        defect_idx = None, defect_pos = 空位中心坐标
      - pristine / 失败: (None, None)
    """
    dtype = sample["defect_type"]
    if dtype == "pristine":
        return None, None

    if dtype.startswith("substitutional_"):
        idx = find_defect_atom_index(sample["structure"], sample["dopant"])
        if idx is not None:
            return idx, positions[idx]
        return None, None

    if dtype.startswith("vacancy_"):
        vac_pos = find_vacancy_site(
            sample.get("pristine_structure"),
            sample["structure"],
        )
        return None, vac_pos

    return None, None

def atoms_to_graph_full(sample, cutoff=CUTOFF):
    """构建完整特征图（节点 10 维），支持 vacancy 定位"""
    structure = sample["structure"]
    energy = sample["energy_diff"]
    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    scaled_pos = atoms.get_scaled_positions()
    cell = atoms.cell.array if atoms.cell is not None else None

    defect_idx, defect_pos = resolve_defect_center(sample, positions, cell)

    node_feats = []
    for i, a in enumerate(structure["atoms"]):
        base = get_atom_features(a["element"])          # 5
        frac = scaled_pos[i]                            # 3
        is_def = 1.0 if (defect_idx is not None and i == defect_idx) else 0.0
        if defect_pos is not None:
            dist = mic_distance(positions[i], defect_pos, cell)
        else:
            dist = 0.0
        node_feats.append(np.concatenate([base, frac, [is_def, dist]]))
    x = torch.tensor(np.array(node_feats), dtype=torch.float)  # (N, 10)

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist - rbf_centers)**2) / (2 * 0.5**2))
    edge_attr = torch.cat([edge_attr, dist / cutoff], dim=1)  # (E, 21)

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
        global_feat=torch.tensor(global_feat, dtype=torch.float).unsqueeze(0),
        mp_id=sample["mp_id"],
        sample_id=sample["id"],
        defect_type=sample["defect_type"],
        has_defect_pos=1 if defect_pos is not None else 0,
    )
    return data

# 构建全部图
print("\n构建完整特征图（含 vacancy 定位）...")
graph_list = []
kept_targets = []
kept_mp_ids = []
vac_ok, vac_fail, sub_ok = 0, 0, 0
for s in samples:
    try:
        g = atoms_to_graph_full(s)
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
        kept_mp_ids.append(s["mp_id"])
        if s["defect_type"].startswith("vacancy_"):
            if g.has_defect_pos:
                vac_ok += 1
            else:
                vac_fail += 1
        elif s["defect_type"].startswith("substitutional_"):
            if g.has_defect_pos:
                sub_ok += 1
    except Exception as e:
        print(f"  跳过 {s['id']}: {e}")

print(f"成功构建图: {len(graph_list)}")
print(f"  vacancy 定位成功: {vac_ok}, 失败: {vac_fail}")
print(f"  substitutional 定位成功: {sub_ok}")
print(f"节点特征维: {graph_list[0].x.shape[1]}, 边特征维: {graph_list[0].edge_attr.shape[1]}, "
      f"全局特征维: {graph_list[0].global_feat.shape[1]}")

if vac_ok > 0:
    vac_dists = []
    for g, mid in zip(graph_list, kept_mp_ids):
        if g.has_defect_pos and g.defect_type.startswith("vacancy_"):
            dcol = g.x[:, 9].numpy()
            vac_dists.append(dcol.max())
    print(f"  vacancy dist_to_defect max (raw, 标准化前抽样): "
          f"mean={np.mean(vac_dists):.3f}, min={np.min(vac_dists):.3f}, max={np.max(vac_dists):.3f}")

# ------------------------------------------------------------
# 按 mp_id 分组划分
# ------------------------------------------------------------
groups = np.array(kept_mp_ids)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(range(len(graph_list)), groups=groups))
train_idx, test_idx = list(train_idx), list(test_idx)
print(f"\n分组划分 (by mp_id): Train={len(train_idx)}, Test={len(test_idx)}")
train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])
print(f"  train mp_ids={len(train_groups)}, test mp_ids={len(test_groups)}, "
      f"overlap={len(train_groups & test_groups)}")

train_targets = np.array([kept_targets[i] for i in train_idx])
y_mean = float(train_targets.mean())
y_std  = float(train_targets.std()) + 1e-8
print(f"目标标准化: mean={y_mean:.3f}, std={y_std:.3f}")

all_x = torch.cat([g.x for g in graph_list], dim=0)
x_mean = all_x.mean(dim=0, keepdim=True)
x_std  = all_x.std(dim=0, keepdim=True) + 1e-8

all_gf = torch.cat([g.global_feat for g in graph_list], dim=0)
n_cont = 7
gf_mean = all_gf[:, :n_cont].mean(dim=0, keepdim=True)
gf_std  = all_gf[:, :n_cont].std(dim=0, keepdim=True) + 1e-8

for g in graph_list:
    g.y = (g.y - y_mean) / y_std
    g.x = (g.x - x_mean) / x_std
    g.global_feat[:, :n_cont] = (g.global_feat[:, :n_cont] - gf_mean) / gf_std

dist_col_std = float(x_std[0, 9])
print(f"dist_to_defect 通道 std (标准化前): {dist_col_std:.4f}  "
      f"{'(有信号)' if dist_col_std > 1e-3 else '(仍接近常数——定位可能大面积失败)'}")

# ============================================================
# 3. 消融配置

# ============================================================
ABLATION_CONFIGS = OrderedDict([
    ("Full", {
        "desc": "完整特征 (基线)",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": None,
    }),
    ("w/o_Element", {
        "desc": "去掉节点元素基础特征 (Z,χ,r,v,is_metal)",
        "node_mask": (0, 5),
        "edge_mask": None,
        "global_mask": None,
    }),
    ("w/o_FracCoord", {
        "desc": "去掉分数坐标 (sx,sy,sz)",
        "node_mask": (5, 8),
        "edge_mask": None,
        "global_mask": None,
    }),
    ("w/o_DefectAware", {
        "desc": "去掉缺陷感知 (is_defect, dist_to_defect)",
        "node_mask": (8, 10),
        "edge_mask": None,
        "global_mask": None,
    }),
    ("w/o_Edge", {
        "desc": "去掉全部边特征 (RBF + dist)",
        "node_mask": None,
        "edge_mask": (0, 21),
        "global_mask": None,
    }),
    ("w/o_GlobalStruct", {
        "desc": "去掉全局结构统计 (n_atoms,V,ρ,n_elem)",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": (0, 4),
    }),
    ("w/o_GlobalElec", {
        "desc": "去掉全局电子/电荷 (charge,mult,ΔN)",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": (4, 7),
    }),
    ("w/o_TypeOneHot", {
        "desc": "去掉缺陷类型 one-hot",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": (7, 7 + n_types),
    }),
    ("Only_NodeElem_Edge", {
        "desc": "仅保留节点元素特征 + 边特征（最简化学图）",
        "node_mask": (5, 10),
        "edge_mask": None,
        "global_mask": (0, 7 + n_types),
    }),
])

def apply_ablation(graphs, config):
    new_graphs = []
    for g in graphs:
        g2 = g.clone()
        if config["node_mask"] is not None:
            s, e = config["node_mask"]
            g2.x[:, s:e] = 0.0
        if config["edge_mask"] is not None:
            s, e = config["edge_mask"]
            g2.edge_attr[:, s:e] = 0.0
        if config["global_mask"] is not None:
            s, e = config["global_mask"]
            g2.global_feat[:, s:e] = 0.0
        new_graphs.append(g2)
    return new_graphs

# ============================================================
# 4. 模型
# ============================================================
class ImprovedGATv2(nn.Module):
    def __init__(self, node_in_dim=10, edge_in_dim=21, global_in_dim=None,
                 hidden_dim=64, num_heads=4, output_dim=1):
        super().__init__()
        if global_in_dim is None:
            global_in_dim = 7 + n_types
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            Linear(node_in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=hidden_dim, dropout=0.1)
        self.norm1 = LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=hidden_dim, dropout=0.1)
        self.norm2 = LayerNorm(hidden_dim)

        self.gat3 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False,
                              edge_dim=hidden_dim, dropout=0.1)
        self.norm3 = LayerNorm(hidden_dim)

        self.global_encoder = nn.Sequential(
            Linear(global_in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim // 2, hidden_dim // 2),
            LayerNorm(hidden_dim // 2)
        )

        self.graph_proj = Linear(hidden_dim * 2, hidden_dim // 2)
        self.fusion_gate = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid()
        )
        self.output_layer = nn.Sequential(
            Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        global_feat = data.global_feat

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x1 = self.gat1(x, edge_index, edge_attr=edge_attr)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.gat2(x1, edge_index, edge_attr=edge_attr)
        x2 = self.norm2(x2)
        x2 = F.relu(x2 + x1)

        x3 = self.gat3(x2, edge_index, edge_attr=edge_attr)
        x3 = self.norm3(x3)
        x3 = F.relu(x3 + x2)

        g_mean = global_mean_pool(x3, batch)
        g_max  = global_max_pool(x3, batch)
        graph_features = torch.cat([g_mean, g_max], dim=-1)

        global_encoded = self.global_encoder(global_feat)
        graph_proj = self.graph_proj(graph_features)

        fusion_input = torch.cat([graph_proj, global_encoded], dim=1)
        gate = self.fusion_gate(fusion_input)
        fused = gate * graph_proj + (1 - gate) * global_encoded

        out = self.output_layer(fused)
        return out.squeeze(-1)

# ============================================================
# 5. 训练与评估
# ============================================================
def train_one_config(config_name, config, seed=0):
    torch.manual_seed(RANDOM_SEED + seed)
    np.random.seed(RANDOM_SEED + seed)

    graphs = apply_ablation(graph_list, config)
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader([graphs[i] for i in test_idx],  batch_size=BATCH_SIZE)

    model = ImprovedGATv2(
        node_in_dim=10,
        edge_in_dim=21,
        global_in_dim=7 + n_types,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=12, factor=0.5
    )

    best_val_mae = float('inf')
    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_idx)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                trues.append(batch.y.cpu().numpy())
        preds = np.concatenate(preds) * y_std + y_mean
        trues = np.concatenate(trues) * y_std + y_mean
        val_mae = mean_absolute_error(trues, preds)
        history.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step(val_mae)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{config_name}] Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val MAE {val_mae:.2f} eV")

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            preds.append(pred.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds) * y_std + y_mean
    trues = np.concatenate(trues) * y_std + y_mean

    mae  = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2   = r2_score(trues, preds)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_epoch_mae": best_val_mae,
        "history": history,
    }

# ============================================================
# 6. 主循环
# ============================================================
print("\n" + "=" * 70)
print("开始特征消融实验 (v2: vacancy 定位 + mp_id 分组划分)")
print("=" * 70)

results = OrderedDict()
t0 = time.time()

for name, cfg in ABLATION_CONFIGS.items():
    print(f"\n>>> {name}: {cfg['desc']}")
    seed_metrics = []
    for s in range(N_SEEDS):
        m = train_one_config(name, cfg, seed=s)
        seed_metrics.append(m)
        print(f"  Seed {s}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")

    maes  = [m["mae"]  for m in seed_metrics]
    rmses = [m["rmse"] for m in seed_metrics]
    r2s   = [m["r2"]   for m in seed_metrics]
    results[name] = {
        "desc": cfg["desc"],
        "mae_mean": float(np.mean(maes)),
        "mae_std":  float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "r2_mean":  float(np.mean(r2s)),
        "history": seed_metrics[0]["history"],
    }

elapsed = time.time() - t0
print(f"\n全部消融完成，耗时 {elapsed/60:.1f} min")

# ============================================================
# 7. 表格与 Utility
# ============================================================
full_mae = results["Full"]["mae_mean"]
print("\n" + "=" * 90)
print(f"{'Config':<22} {'MAE (eV)':>12} {'RMSE':>10} {'R²':>8} {'Utility':>10}  Description")
print("-" * 90)

for name, r in results.items():
    if name == "Full":
        util_str = "  (base)"
    else:
        util = (r["mae_mean"] - full_mae) / full_mae * 100
        util_str = f"{util:+.1f}%"
    print(f"{name:<22} {r['mae_mean']:8.2f}±{r['mae_std']:.2f}  "
          f"{r['rmse_mean']:8.2f}  {r['r2_mean']:7.4f}  {util_str:>8}  {r['desc']}")

print("=" * 90)
print("Utility = (MAE_ablated - MAE_Full) / MAE_Full × 100%")
print("正值越大 → 该特征组对性能贡献越大（去掉后误差上升越多）")

# ============================================================
# 8. 保存
# ============================================================
csv_path = OUT_DIR / "ablation_results_v2.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("config,mae_mean,mae_std,rmse_mean,r2_mean,utility_pct,description\n")
    for name, r in results.items():
        util = 0.0 if name == "Full" else (r["mae_mean"] - full_mae) / full_mae * 100
        f.write(f"{name},{r['mae_mean']:.4f},{r['mae_std']:.4f},"
                f"{r['rmse_mean']:.4f},{r['r2_mean']:.4f},{util:.2f},\"{r['desc']}\"\n")
print(f"\n结果已保存: {csv_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
names = list(results.keys())
maes  = [results[n]["mae_mean"] for n in names]
utils = [0.0 if n == "Full" else (results[n]["mae_mean"] - full_mae) / full_mae * 100 for n in names]
colors = ["#2ecc71" if n == "Full" else "#e74c3c" if u > 20 else "#f39c12" if u > 5 else "#3498db"
          for n, u in zip(names, utils)]

ax = axes[0]
ax.barh(names[::-1], maes[::-1], color=colors[::-1], edgecolor="k", linewidth=0.6)
ax.axvline(full_mae, color="green", ls="--", lw=1.5, label=f"Full MAE={full_mae:.1f}")
ax.set_xlabel("Test MAE (eV)")
ax.set_title("Ablation v2: Test MAE (grouped split + vacancy fix)")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)

ax = axes[1]
ax.barh(names[::-1], utils[::-1], color=colors[::-1], edgecolor="k", linewidth=0.6)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Utility (%)")
ax.set_title("Feature Utility\n(higher = more important)")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "ablation_feature_utility_v2.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"可视化已保存: {fig_path}")

fig2, ax = plt.subplots(figsize=(8, 5))
for name, r in results.items():
    ax.plot(r["history"], label=name, alpha=0.85)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val MAE (eV)")
ax.set_title("Ablation v2 Learning Curves")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
curve_path = OUT_DIR / "ablation_learning_curves_v2.png"
plt.savefig(curve_path, dpi=160, bbox_inches="tight")
print(f"学习曲线已保存: {curve_path}")

print("\n消融实验 v2 全部完成。")
print("关键对照：")
print("  - w/o_TypeOneHot：若 vacancy 定位生效，MAE 涨幅应显著小于 v1 的 +173%")
print("  - w/o_DefectAware：若 dist_to_defect 有真实方差，Utility 应明显为正")
print("  - Only_NodeElem_Edge：纯图能力的真实下界（分组划分下更难靠组成记忆）")
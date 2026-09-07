#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vs_exp8: 强制图分支学习（无 type one-hot）+ 电子环境主特征
================================================================
设计原则（基于消融结论 + 刚性剔除假设）：
  1. Full 模型定义中彻底不含 defect_type one-hot（不是消融 mask，是特征表里没有）。
  2. 全局只保留物理标量：charge, multiplicity, ΔN_elec + Mulliken 体系统计。
  3. 节点以电子环境为主：
       - Mulliken net / Hirshfeld net / spin（自洽布居）
       - 价电子 / d 电子数 / KIND-q（化学先验，部分替代 type）
       - 修复后的缺陷定位：is_defect_atom, dist_to_defect
  4. 坐标类特征默认关闭（USE_FRAC_COORD=False）：刚性下几乎无信息。
  5. train/test 按 mp_id 分组，避免同材料电荷态泄漏。

节点特征布局（默认 12 维，无分数坐标）：
  [0:5]   Z, χ, r, v, is_metal
  [5:7]   n_d, kind_q
  [7:10]  q_mull, q_hirsh, spin
  [10:12] is_defect, dist_to_defect
  若 USE_FRAC_COORD=True 再在中间插入 sx,sy,sz → 15 维。

全局特征（12 维，无 type）：
  charge, multiplicity, ΔN_elec,
  mean|q_mull|, max|q_mull|, std(q_mull),
  shell_mean_q, shell_max_abs_q,
  n_atoms, density,
  vac_nd, vac_kq

用法：
  python vs_exp8_no_type_electronic.py
  数据需含 mulliken/hirshfeld 字段（与 exp7 相同 JSON）。
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, LayerNorm, Linear
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

# ============================================================
# 0. 路径与超参
# ============================================================
DATA_PATH = Path("cp2k_gnn_data.json")
if not DATA_PATH.exists():
    DATA_PATH = Path("cp2k_gnn_data.json")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

HIDDEN_DIM = 96
NUM_HEADS = 4
BATCH_SIZE = 8
MAX_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
CUTOFF = 6.0
SHELL_CUTOFF = 3.0
RANDOM_SEED = 42
N_SEEDS = 1
VACANCY_MATCH_DIST = 1.8
USE_FRAC_COORD = False
PATIENCE = 25

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 元素化学先验（价电子 / d 电子 / KIND-q）
# ============================================================
ELEMENT_PRIOR = {
    # element: (Z, χ, r_pm, valence_e, is_metal, n_d, kind_q)
    "H":  (1,  2.20,  31, 1, 0, 0, 1),
    "C":  (6,  2.55,  76, 4, 0, 0, 4),
    "N":  (7,  3.04,  71, 5, 0, 0, 5),
    "O":  (8,  3.44,  66, 6, 0, 0, 6),
    "S":  (16, 2.58, 105, 6, 0, 0, 6),
    "Ca": (20, 1.00, 176, 2, 1, 0, 10),
    "Ti": (22, 1.54, 160, 4, 1, 2, 12),
    "V":  (23, 1.63, 153, 5, 1, 3, 13),
    "Cr": (24, 1.66, 139, 6, 1, 5, 14),
    "Fe": (26, 1.83, 132, 8, 1, 6, 16),
    "Zn": (30, 1.65, 122, 2, 1, 10, 12),
    "Sr": (38, 0.95, 195, 2, 1, 0, 10),
    "Y":  (39, 1.22, 190, 3, 1, 1, 11),
    "Nb": (41, 1.60, 164, 5, 1, 4, 13),
    "Mo": (42, 2.16, 154, 6, 1, 5, 14),
    "Sn": (50, 1.96, 140, 4, 1, 0, 14),
    "Ba": (56, 0.89, 215, 2, 1, 0, 10),
    "La": (57, 1.10, 195, 3, 1, 1, 11),
    "Hf": (72, 1.30, 175, 4, 1, 2, 12),
    "Ta": (73, 1.50, 170, 5, 1, 3, 13),
    "W":  (74, 2.36, 162, 6, 1, 4, 14),
    "Pb": (82, 2.33, 146, 4, 1, 0, 14),
    "Bi": (83, 2.02, 148, 5, 1, 0, 15),
}

def get_element_vector(element):
    t = ELEMENT_PRIOR.get(element, (0, 0.0, 0.0, 0, 0, 0, 0))
    return np.array(t, dtype=np.float32)

# ============================================================
# 2. 数据加载
# ============================================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

def parse_dopant_from_type(defect_type: str):
    if not defect_type.startswith("substitutional_"):
        return None
    part = defect_type.replace("substitutional_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[:1].upper() + part[1:].lower()

def parse_vacancy_element(defect_type: str):
    if not defect_type.startswith("vacancy_"):
        return None
    part = defect_type.replace("vacancy_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[:1].upper() + part[1:].lower()

samples = []
n_with_charges = 0
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
            "pristine_structure": None,
            "charge": p.get("inp_charge", 0),
            "multiplicity": p.get("inp_multiplicity", 1),
            "n_elec_s1": p.get("n_electrons_spin1"),
            "n_elec_s2": p.get("n_electrons_spin2"),
            "defect_type": "pristine",
            "dopant": None,
            "vacancy_element": None,
            "mulliken": p.get("mulliken"),
            "hirshfeld": p.get("hirshfeld"),
        })
        if p.get("has_charges") or p.get("mulliken"):
            n_with_charges += 1

    for d in system.get("defects", []):
        if not d.get("has_structure"):
            continue
        dtype = d["defect_type"]
        samples.append({
            "id": f"{mp_id}_{dtype}_q{d.get('charge_from_name', '')}",
            "mp_id": mp_id,
            "energy_diff": d.get("energy_diff_ev", d["energy_ev"]),
            "structure": d["structure"],
            "pristine_structure": pristine_struct,
            "charge": d.get("inp_charge", 0),
            "multiplicity": d.get("inp_multiplicity", 1),
            "n_elec_s1": d.get("n_electrons_spin1"),
            "n_elec_s2": d.get("n_electrons_spin2"),
            "defect_type": dtype,
            "dopant": parse_dopant_from_type(dtype),
            "vacancy_element": parse_vacancy_element(dtype),
            "mulliken": d.get("mulliken"),
            "hirshfeld": d.get("hirshfeld"),
        })
        if d.get("has_charges") or d.get("mulliken"):
            n_with_charges += 1

print(f"总样本数: {len(samples)}, 带电荷分析: {n_with_charges}")
n_vac = sum(1 for s in samples if s["defect_type"].startswith("vacancy_"))
n_sub = sum(1 for s in samples if s["defect_type"].startswith("substitutional_"))
print(f"  pristine={sum(1 for s in samples if s['defect_type']=='pristine')}, "
      f"vacancy={n_vac}, substitutional={n_sub}")

# ============================================================
# 3. 图构建工具
# ============================================================
def structure_to_ase(structure):
    symbols = [a["element"] for a in structure["atoms"]]
    positions = [[a["x"], a["y"], a["z"]] for a in structure["atoms"]]
    cell = structure.get("lattice_vectors")
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def mic_distance(pos_i, pos_j, cell):
    delta = np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float)
    if cell is not None:
        try:
            cell = np.asarray(cell, dtype=float)
            inv = np.linalg.inv(cell)
            frac = delta @ inv.T
            frac -= np.round(frac)
            delta = frac @ cell
        except np.linalg.LinAlgError:
            pass
    return float(np.linalg.norm(delta))

def find_defect_atom_index(structure, dopant):
    if dopant is None:
        return None
    for i, a in enumerate(structure["atoms"]):
        if a["element"] == dopant:
            return i
    return None

def find_vacancy_site(pristine_struct, defect_struct, max_dist=VACANCY_MATCH_DIST):
    if pristine_struct is None or defect_struct is None:
        return None
    p_atoms = structure_to_ase(pristine_struct)
    d_atoms = structure_to_ase(defect_struct)
    p_pos, d_pos = p_atoms.get_positions(), d_atoms.get_positions()
    p_sym, d_sym = p_atoms.get_chemical_symbols(), d_atoms.get_chemical_symbols()
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
    return p_pos[unmatched[0]].copy()

def resolve_defect_center(sample, positions, cell):
    dtype = sample["defect_type"]
    if dtype == "pristine":
        return None, None
    if dtype.startswith("substitutional_"):
        idx = find_defect_atom_index(sample["structure"], sample["dopant"])
        if idx is not None:
            return idx, positions[idx]
        return None, None
    if dtype.startswith("vacancy_"):
        vac_pos = find_vacancy_site(sample.get("pristine_structure"), sample["structure"])
        return None, vac_pos
    return None, None

def extract_per_atom_charges(sample, n_atoms):
    mull_net = np.zeros(n_atoms, dtype=np.float32)
    hirsh_net = np.zeros(n_atoms, dtype=np.float32)
    spin = np.zeros(n_atoms, dtype=np.float32)
    has_mull = False

    mull = sample.get("mulliken")
    if mull and "atoms" in mull:
        for atom in mull["atoms"]:
            idx = int(atom["atom_index"]) - 1
            if 0 <= idx < n_atoms:
                mull_net[idx] = float(atom.get("net_charge", 0.0))
                spin[idx] = float(atom.get("spin_moment", 0.0))
                has_mull = True

    hirsh = sample.get("hirshfeld")
    if hirsh and "atoms" in hirsh:
        for atom in hirsh["atoms"]:
            idx = int(atom["atom_index"]) - 1
            if 0 <= idx < n_atoms:
                hirsh_net[idx] = float(atom.get("net_charge", 0.0))

    return mull_net, hirsh_net, spin, has_mull

def atoms_to_graph(sample, cutoff=CUTOFF):
    structure = sample["structure"]
    energy = sample["energy_diff"]
    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    scaled_pos = atoms.get_scaled_positions()
    cell = atoms.cell.array if atoms.cell is not None else None

    defect_idx, defect_pos = resolve_defect_center(sample, positions, cell)
    mull_net, hirsh_net, spin, has_mull = extract_per_atom_charges(sample, n_atoms)

    removed_el = sample.get("vacancy_element")
    removed_prior = get_element_vector(removed_el) if removed_el else np.zeros(7, dtype=np.float32)

    node_feats = []
    for i, a in enumerate(structure["atoms"]):
        el_vec = get_element_vector(a["element"])
        base = el_vec[:5]
        chem = el_vec[5:7]
        elec = np.array([mull_net[i], hirsh_net[i], spin[i]], dtype=np.float32)
        is_def = 1.0 if (defect_idx is not None and i == defect_idx) else 0.0
        dist = mic_distance(positions[i], defect_pos, cell) if defect_pos is not None else 0.0
        defect = np.array([is_def, dist], dtype=np.float32)

        if USE_FRAC_COORD:
            frac = scaled_pos[i].astype(np.float32)
            feat = np.concatenate([base, chem, frac, elec, defect])
        else:
            feat = np.concatenate([base, chem, elec, defect])
        node_feats.append(feat)

    x = torch.tensor(np.array(node_feats), dtype=torch.float)
    node_in_dim = x.shape[1]

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist_e = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist_e - rbf_centers) ** 2) / (2 * 0.5 ** 2))
    edge_attr = torch.cat([edge_attr, dist_e / cutoff], dim=1)

    vol = float(atoms.get_volume()) if atoms.get_volume() > 0 else 1.0
    density = n_atoms / vol
    charge = float(sample.get("charge", 0))
    multiplicity = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_elec = float(abs(s1 - s2)) if (s1 is not None and s2 is not None) else 0.0

    abs_mull = np.abs(mull_net)
    mean_abs_q = float(abs_mull.mean()) if n_atoms > 0 else 0.0
    max_abs_q = float(abs_mull.max()) if n_atoms > 0 else 0.0
    std_q = float(mull_net.std()) if n_atoms > 1 else 0.0

    if defect_pos is not None:
        shell_mask = np.array([
            mic_distance(positions[i], defect_pos, cell) < SHELL_CUTOFF
            for i in range(n_atoms)
        ])
        if shell_mask.any():
            shell_q = mull_net[shell_mask]
            shell_mean_q = float(shell_q.mean())
            shell_max_abs_q = float(np.abs(shell_q).max())
        else:
            shell_mean_q, shell_max_abs_q = 0.0, 0.0
    else:
        shell_mean_q, shell_max_abs_q = 0.0, 0.0

    if sample["defect_type"].startswith("vacancy_"):
        vac_nd, vac_kq = float(removed_prior[5]), float(removed_prior[6])
    elif defect_idx is not None:
        el_vec = get_element_vector(structure["atoms"][defect_idx]["element"])
        vac_nd, vac_kq = float(el_vec[5]), float(el_vec[6])
    else:
        vac_nd, vac_kq = 0.0, 0.0

    global_feat = np.array([
        charge, multiplicity, delta_elec,
        mean_abs_q, max_abs_q, std_q,
        shell_mean_q, shell_max_abs_q,
        float(n_atoms), density,
        vac_nd, vac_kq,
    ], dtype=np.float32)

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
        has_mulliken=1 if has_mull else 0,
    )
    return data, node_in_dim

# ============================================================
# 4. 构建图
# ============================================================
print("\n构建图（无 type one-hot，电子环境主特征）...")
graph_list, kept_targets, kept_mp_ids = [], [], []
vac_ok = vac_fail = sub_ok = 0
node_in_dim = None
n_mull_graphs = 0

for s in samples:
    try:
        g, dim = atoms_to_graph(s)
        if node_in_dim is None:
            node_in_dim = dim
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
        kept_mp_ids.append(s["mp_id"])
        if g.has_mulliken:
            n_mull_graphs += 1
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

GLOBAL_IN_DIM = int(graph_list[0].global_feat.shape[1])
print(f"成功构建图: {len(graph_list)}")
print(f"  节点维={node_in_dim}, 边维={graph_list[0].edge_attr.shape[1]}, 全局维={GLOBAL_IN_DIM}（无 type）")
print(f"  vacancy 定位成功={vac_ok}, 失败={vac_fail}; substitutional={sub_ok}")
print(f"  含 Mulliken 的图: {n_mull_graphs}/{len(graph_list)}")
print(f"  USE_FRAC_COORD={USE_FRAC_COORD}")

groups = np.array(kept_mp_ids)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(range(len(graph_list)), groups=groups))
train_idx, test_idx = list(train_idx), list(test_idx)
print(f"\n分组划分: Train={len(train_idx)}, Test={len(test_idx)}, "
      f"mp overlap={len(set(groups[train_idx]) & set(groups[test_idx]))}")

train_targets = np.array([kept_targets[i] for i in train_idx])
y_mean = float(train_targets.mean())
y_std = float(train_targets.std()) + 1e-8
print(f"目标标准化: mean={y_mean:.3f}, std={y_std:.3f}")

all_x = torch.cat([g.x for g in graph_list], dim=0)
x_mean = all_x.mean(dim=0, keepdim=True)
x_std = all_x.std(dim=0, keepdim=True) + 1e-8

all_gf = torch.cat([g.global_feat for g in graph_list], dim=0)
gf_mean = all_gf.mean(dim=0, keepdim=True)
gf_std = all_gf.std(dim=0, keepdim=True) + 1e-8

for g in graph_list:
    g.y = (g.y - y_mean) / y_std
    g.x = (g.x - x_mean) / x_std
    g.global_feat = (g.global_feat - gf_mean) / gf_std

print(f"节点各维 std(标准化前): {x_std.numpy().ravel().round(3)}")

# ============================================================
# 5. 模型
# ============================================================
class ElectronAwareGATv2(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim=21, global_in_dim=GLOBAL_IN_DIM,
                 hidden_dim=96, num_heads=4, output_dim=1, global_dropout=0.3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            Linear(node_in_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
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
            nn.Dropout(global_dropout),
            Linear(hidden_dim // 2, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
        )
        self.graph_proj = Linear(hidden_dim * 2, hidden_dim // 2)
        self.output_layer = nn.Sequential(
            Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        global_feat = data.global_feat

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x1 = F.relu(self.norm1(self.gat1(x, edge_index, edge_attr=edge_attr)))
        x2 = F.relu(self.norm2(self.gat2(x1, edge_index, edge_attr=edge_attr)) + x1)
        x3 = F.relu(self.norm3(self.gat3(x2, edge_index, edge_attr=edge_attr)) + x2)

        g_mean = global_mean_pool(x3, batch)
        g_max = global_max_pool(x3, batch)
        graph_proj = self.graph_proj(torch.cat([g_mean, g_max], dim=-1))
        global_encoded = self.global_encoder(global_feat)

        fused = graph_proj + global_encoded
        return self.output_layer(fused).squeeze(-1)

# ============================================================
# 6. 消融配置
# ============================================================
NODE_DIM = node_in_dim
assert NODE_DIM in (12, 15), f" unexpected node dim {NODE_DIM}"

if USE_FRAC_COORD:
    SL_ELEM, SL_CHEM, SL_ELEC, SL_DEF = (0, 5), (5, 7), (10, 13), (13, 15)
else:
    SL_ELEM, SL_CHEM, SL_ELEC, SL_DEF = (0, 5), (5, 7), (7, 10), (10, 12)

ABLATION_CONFIGS = OrderedDict([
    ("Full", {
        "desc": "无type基线：元素+化学先验+Mulliken+缺陷定位+物理全局",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": None,
    }),
    ("w/o_Mulliken", {
        "desc": "去掉节点 Mulliken/Hirshfeld/spin",
        "node_mask": SL_ELEC,
        "edge_mask": None,
        "global_mask": (3, 8),
    }),
    ("w/o_ChemPrior", {
        "desc": "去掉 n_d + KIND_q（节点）及全局 vac_nd/vac_kq",
        "node_mask": SL_CHEM,
        "edge_mask": None,
        "global_mask": (10, 12),
    }),
    ("w/o_DefectAware", {
        "desc": "去掉 is_defect + dist_to_defect",
        "node_mask": SL_DEF,
        "edge_mask": None,
        "global_mask": None,
    }),
    ("w/o_GlobalElec", {
        "desc": "去掉 charge/mult/ΔN",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": (0, 3),
    }),
    ("GraphOnly", {
        "desc": "仅图：节点全特征+边，全局全置零",
        "node_mask": None,
        "edge_mask": None,
        "global_mask": (0, GLOBAL_IN_DIM),
    }),
])

def apply_ablation(graphs, config):
    out = []
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
        out.append(g2)
    return out

# ============================================================
# 7. 训练
# ============================================================
def train_one_config(config_name, config, seed=0):
    torch.manual_seed(RANDOM_SEED + seed)
    np.random.seed(RANDOM_SEED + seed)

    graphs = apply_ablation(graph_list, config)
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=BATCH_SIZE)

    model = ElectronAwareGATv2(
        node_in_dim=NODE_DIM,
        edge_in_dim=21,
        global_in_dim=GLOBAL_IN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        global_dropout=0.3,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=12, factor=0.5
    )

    best_val_mae = float("inf")
    best_state = None
    history = []
    bad_epochs = 0

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
        avg_loss = total_loss / max(len(train_idx), 1)

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
        scheduler.step(val_mae)

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{config_name}] Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val MAE {val_mae:.2f} eV")

        if bad_epochs >= PATIENCE:
            print(f"  [{config_name}] early stop at epoch {epoch}")
            break

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

    return {
        "mae": mean_absolute_error(trues, preds),
        "rmse": float(np.sqrt(mean_squared_error(trues, preds))),
        "r2": r2_score(trues, preds),
        "history": history,
    }

# ============================================================
# 8. 主循环
# ============================================================
print("\n" + "=" * 70)
print("vs_exp8: 无 type one-hot | Mulliken + 化学先验 | 加法融合强逼图分支")
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
    results[name] = {
        "desc": cfg["desc"],
        "mae_mean": float(np.mean([m["mae"] for m in seed_metrics])),
        "mae_std": float(np.std([m["mae"] for m in seed_metrics])),
        "rmse_mean": float(np.mean([m["rmse"] for m in seed_metrics])),
        "r2_mean": float(np.mean([m["r2"] for m in seed_metrics])),
        "history": seed_metrics[0]["history"],
    }

print(f"\n全部完成，耗时 {(time.time() - t0) / 60:.1f} min")

full_mae = results["Full"]["mae_mean"]
print("\n" + "=" * 95)
print(f"{'Config':<22} {'MAE (eV)':>12} {'RMSE':>10} {'R²':>8} {'Utility':>10}  Description")
print("-" * 95)
for name, r in results.items():
    if name == "Full":
        util_str = "  (base)"
    else:
        util = (r["mae_mean"] - full_mae) / full_mae * 100
        util_str = f"{util:+.1f}%"
    print(f"{name:<22} {r['mae_mean']:8.2f}±{r['mae_std']:.2f}  "
          f"{r['rmse_mean']:8.2f}  {r['r2_mean']:7.4f}  {util_str:>8}  {r['desc']}")
print("=" * 95)
print("Utility = (MAE_ablated - MAE_Full) / MAE_Full × 100%")
print("关注: w/o_Mulliken 与 GraphOnly —— 电子特征与纯图能力")

csv_path = OUT_DIR / "exp8_no_type_electronic_results.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("config,mae_mean,mae_std,rmse_mean,r2_mean,utility_pct,description\n")
    for name, r in results.items():
        util = 0.0 if name == "Full" else (r["mae_mean"] - full_mae) / full_mae * 100
        f.write(f"{name},{r['mae_mean']:.4f},{r['mae_std']:.4f},"
                f"{r['rmse_mean']:.4f},{r['r2_mean']:.4f},{util:.2f},\"{r['desc']}\"\n")
print(f"\n结果: {csv_path}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names = list(results.keys())
maes = [results[n]["mae_mean"] for n in names]
utils = [0.0 if n == "Full" else (results[n]["mae_mean"] - full_mae) / full_mae * 100 for n in names]
colors = ["#2ecc71" if n == "Full" else "#e74c3c" if u > 15 else "#f39c12" if u > 5 else "#3498db"
          for n, u in zip(names, utils)]

axes[0].barh(names[::-1], maes[::-1], color=colors[::-1], edgecolor="k", linewidth=0.6)
axes[0].axvline(full_mae, color="green", ls="--", lw=1.5, label=f"Full={full_mae:.1f}")
axes[0].set_xlabel("Test MAE (eV)")
axes[0].set_title("exp8: No type-onehot | electronic features")
axes[0].legend(loc="lower right")
axes[0].grid(axis="x", alpha=0.3)

axes[1].barh(names[::-1], utils[::-1], color=colors[::-1], edgecolor="k", linewidth=0.6)
axes[1].axvline(0, color="k", lw=1)
axes[1].set_xlabel("Utility (%)")
axes[1].set_title("Feature utility (no type shortcut)")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "exp8_feature_utility.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"图: {fig_path}")

fig2, ax = plt.subplots(figsize=(8, 5))
for name, r in results.items():
    ax.plot(r["history"], label=name, alpha=0.85)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val MAE (eV)")
ax.set_title("exp8 learning curves")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
curve_path = OUT_DIR / "exp8_learning_curves.png"
plt.savefig(curve_path, dpi=160, bbox_inches="tight")
print(f"曲线: {curve_path}")

print("\n保存 Full 权重 ...")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
graphs = apply_ablation(graph_list, ABLATION_CONFIGS["Full"])
train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=BATCH_SIZE)
model = ElectronAwareGATv2(NODE_DIM, 21, GLOBAL_IN_DIM, HIDDEN_DIM, NUM_HEADS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=12, factor=0.5)
best_state, best_mae, bad = None, float("inf"), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch), batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            preds.append(model(batch).cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds) * y_std + y_mean
    trues = np.concatenate(trues) * y_std + y_mean
    mae = mean_absolute_error(trues, preds)
    scheduler.step(mae)
    if mae < best_mae - 1e-6:
        best_mae = mae
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
    if epoch % 20 == 0:
        print(f"  [save] Epoch {epoch} Val MAE {mae:.2f}")
    if bad >= PATIENCE:
        break

weight_path = OUT_DIR / "exp8_no_type_electronic_best.pt"
torch.save({
    "state_dict": best_state,
    "node_in_dim": NODE_DIM,
    "global_in_dim": GLOBAL_IN_DIM,
    "y_mean": y_mean,
    "y_std": y_std,
    "x_mean": x_mean,
    "x_std": x_std,
    "gf_mean": gf_mean,
    "gf_std": gf_std,
    "use_frac_coord": USE_FRAC_COORD,
    "mae": best_mae,
    "config": "Full_no_type_electronic",
}, weight_path)
print(f"权重: {weight_path} (MAE={best_mae:.3f})")
print("\nexp8 全部完成。")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vs_exp9 Stage B: 在位点读出框架上叠加 ALIGNN 式配位 / 键角环境
================================================================
依赖 Stage A 的「宿主图 + defect_idx」路径，在读出头追加：

  B1 位点壳层化学直方图 + CN + 邻居平均 χ / valence / n_d
  B2 以 i' 为顶点的键角 cosθ → RBF 池化（轻量三体，仅位点邻域）

消融：
  A-only（无壳层无键角）
  A + ShellHist
  A + ShellHist + AngleRBF   ← Full Stage B

无 type one-hot；划分仍按 mp_id；Mulliken 不强制。

用法：
  python vs_exp9_stageB_coord_env.py
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict, Counter
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
from torch_geometric.utils import scatter
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

HIDDEN_DIM   = 96
NUM_HEADS    = 4
BATCH_SIZE   = 8
MAX_EPOCHS   = 80
LR           = 1e-3
WEIGHT_DECAY = 1e-5
CUTOFF       = 6.0
SHELL_CUTOFF = 3.5          # 位点壳层半径 (Å)
ANGLE_RBF_DIM = 16          # 键角 RBF 通道数
RANDOM_SEED  = 42
N_SEEDS      = 1
VACANCY_MATCH_DIST = 1.8
PATIENCE     = 25
K_NEIGHBOR_POOL = True

# 元素表顺序（固定直方图维度）
ELEM_LIST = [
    "H", "C", "N", "O", "S", "Ca", "Ti", "V", "Cr", "Fe", "Zn",
    "Sr", "Y", "Nb", "Mo", "Sn", "Ba", "La", "Hf", "Ta", "W", "Pb", "Bi"
]
ELEM_TO_I = {e: i for i, e in enumerate(ELEM_LIST)}
N_ELEM_HIST = len(ELEM_LIST)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 元素化学先验
# ============================================================
ELEMENT_PRIOR = {
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
    "Sn": (50, 1.96, 140, 4, 1, 10, 14),
    "Ba": (56, 0.89, 215, 2, 1, 0, 10),
    "La": (57, 1.10, 195, 3, 1, 1, 11),
    "Hf": (72, 1.30, 175, 4, 1, 2, 12),
    "Ta": (73, 1.50, 170, 5, 1, 3, 13),
    "W":  (74, 2.36, 162, 6, 1, 4, 14),
    "Pb": (82, 2.33, 146, 4, 1, 10, 14),
    "Bi": (83, 2.02, 148, 5, 1, 10, 15),
}

def get_prior(element):
    t = ELEMENT_PRIOR.get(element, (0, 0.0, 0.0, 0, 0, 0, 0))
    return np.array(t, dtype=np.float32)

# ============================================================
# 2. 几何工具
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
            inv = np.linalg.inv(np.asarray(cell, dtype=float))
            frac = delta @ inv.T
            frac -= np.round(frac)
            delta = frac @ np.asarray(cell, dtype=float)
        except np.linalg.LinAlgError:
            pass
    return float(np.linalg.norm(delta))

def mic_vector(pos_i, pos_j, cell):
    delta = np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float)
    if cell is not None:
        try:
            inv = np.linalg.inv(np.asarray(cell, dtype=float))
            frac = delta @ inv.T
            frac -= np.round(frac)
            delta = frac @ np.asarray(cell, dtype=float)
        except np.linalg.LinAlgError:
            pass
    return delta

def find_vacancy_index_in_pristine(pristine_struct, defect_struct, max_dist=VACANCY_MATCH_DIST):
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
    return unmatched[0]

def find_defect_atom_index(structure, dopant):
    if dopant is None:
        return None
    for i, a in enumerate(structure["atoms"]):
        if a["element"] == dopant:
            return i
    return None

def parse_dopant_from_type(defect_type: str):
    if not defect_type.startswith("substitutional_"):
        return None
    part = defect_type.replace("substitutional_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[:1].upper() + part[1:].lower()

# ============================================================
# 3. 壳层化学 + 键角描述子（Stage B 核心）
# ============================================================
def compute_shell_and_angles(structure, defect_idx, shell_cutoff=SHELL_CUTOFF,
                             angle_rbf_dim=ANGLE_RBF_DIM):
    """
    以 defect_idx 为中心：
      - element histogram (固定 N_ELEM_HIST)
      - CN
      - mean χ, mean valence, mean n_d of neighbors
      - cosθ of all triples (j, i', k) → RBF pool → angle_rbf_dim
    若 defect_idx is None 或越界，返回全零向量。
    """
    hist = np.zeros(N_ELEM_HIST, dtype=np.float32)
    cn = 0.0
    mean_chi = 0.0
    mean_val = 0.0
    mean_nd = 0.0
    ang_feat = np.zeros(angle_rbf_dim, dtype=np.float32)

    if defect_idx is None or defect_idx < 0:
        # hist( N ) + cn(1) + meanχ/v/nd(3) + ang(RBF)
        return np.concatenate([hist, [cn, mean_chi, mean_val, mean_nd], ang_feat])

    atoms = structure_to_ase(structure)
    positions = atoms.get_positions()
    cell = atoms.cell.array if atoms.cell is not None else None
    symbols = atoms.get_chemical_symbols()
    n = len(positions)
    if defect_idx >= n:
        return np.concatenate([hist, [cn, mean_chi, mean_val, mean_nd], ang_feat])

    center = positions[defect_idx]
    neigh_idx = []
    for j in range(n):
        if j == defect_idx:
            continue
        d = mic_distance(positions[j], center, cell)
        if d < shell_cutoff and d > 1e-6:
            neigh_idx.append(j)

    cn = float(len(neigh_idx))
    if cn == 0:
        return np.concatenate([hist, [cn, mean_chi, mean_val, mean_nd], ang_feat])

    chis, vals, nds = [], [], []
    for j in neigh_idx:
        el = symbols[j]
        if el in ELEM_TO_I:
            hist[ELEM_TO_I[el]] += 1.0
        prior = get_prior(el)
        chis.append(prior[1])
        vals.append(prior[3])
        nds.append(prior[5])
    mean_chi = float(np.mean(chis))
    mean_val = float(np.mean(vals))
    mean_nd = float(np.mean(nds))
    # 归一化直方图（可选，避免大 CN 主导）
    hist = hist / max(cn, 1.0)

    # 键角：所有 (j, k) 对，j < k
    cosines = []
    for a in range(len(neigh_idx)):
        for b in range(a + 1, len(neigh_idx)):
            j, k = neigh_idx[a], neigh_idx[b]
            vj = mic_vector(positions[j], center, cell)
            vk = mic_vector(positions[k], center, cell)
            nj = np.linalg.norm(vj)
            nk = np.linalg.norm(vk)
            if nj < 1e-8 or nk < 1e-8:
                continue
            c = float(np.dot(vj, vk) / (nj * nk))
            c = max(-1.0, min(1.0, c))
            cosines.append(c)

    if cosines:
        cosines = np.array(cosines, dtype=np.float32)
        # RBF on cosθ ∈ [-1, 1]
        centers = np.linspace(-1.0, 1.0, angle_rbf_dim, dtype=np.float32)
        gamma = 1.0 / (0.25 ** 2)  # 宽度可调
        # (n_angles, rbf_dim)
        rbf = np.exp(-gamma * (cosines[:, None] - centers[None, :]) ** 2)
        ang_feat = rbf.mean(axis=0).astype(np.float32)

    return np.concatenate([hist, [cn, mean_chi, mean_val, mean_nd], ang_feat])

# site_env 维度
SITE_ENV_DIM = N_ELEM_HIST + 1 + 3 + ANGLE_RBF_DIM  # hist + cn + 3 means + ang

# ============================================================
# 4. 数据加载
# ============================================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

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
            "pristine_structure": None,
            "charge": p.get("inp_charge", 0),
            "multiplicity": p.get("inp_multiplicity", 1),
            "n_elec_s1": p.get("n_electrons_spin1"),
            "n_elec_s2": p.get("n_electrons_spin2"),
            "defect_type": "pristine",
            "dopant": None,
            "use_host_graph": False,
        })
    for d in system.get("defects", []):
        if not d.get("has_structure"):
            continue
        dtype = d["defect_type"]
        is_vac = dtype.startswith("vacancy_")
        samples.append({
            "id": f"{mp_id}_{dtype}_q{d.get('charge_from_name', '')}",
            "mp_id": mp_id,
            "energy_diff": d.get("energy_diff_ev", d.get("energy_ev", 0.0)),
            "structure": d["structure"],
            "pristine_structure": pristine_struct,
            "charge": d.get("inp_charge", 0),
            "multiplicity": d.get("inp_multiplicity", 1),
            "n_elec_s1": d.get("n_electrons_spin1"),
            "n_elec_s2": d.get("n_electrons_spin2"),
            "defect_type": dtype,
            "dopant": parse_dopant_from_type(dtype),
            "use_host_graph": is_vac,
        })

print(f"总样本数: {len(samples)}")
n_vac = sum(1 for s in samples if s["defect_type"].startswith("vacancy_"))
n_sub = sum(1 for s in samples if s["defect_type"].startswith("substitutional_"))
n_pris = sum(1 for s in samples if s["defect_type"] == "pristine")
print(f"  pristine={n_pris}, vacancy={n_vac}, substitutional={n_sub}")

# ============================================================
# 5. 图构建（含 site_env）
# ============================================================
def atoms_to_graph_stageB(sample, cutoff=CUTOFF, include_shell=True, include_angle=True):
    use_host = sample.get("use_host_graph", False)
    if use_host and sample.get("pristine_structure") is not None:
        structure = sample["pristine_structure"]
        defect_idx = find_vacancy_index_in_pristine(
            sample["pristine_structure"], sample["structure"]
        )
        use_site = defect_idx is not None
    else:
        structure = sample["structure"]
        defect_idx = None
        use_site = False
        if sample["defect_type"].startswith("substitutional_"):
            defect_idx = find_defect_atom_index(structure, sample["dopant"])
            use_site = False

    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    cell = atoms.cell.array if atoms.cell is not None else None

    node_feats = []
    for i, a in enumerate(structure["atoms"]):
        prior = get_prior(a["element"])
        is_def = 1.0 if (defect_idx is not None and i == defect_idx) else 0.0
        node_feats.append(np.concatenate([prior, [is_def]]))
    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist - rbf_centers) ** 2) / (2 * 0.5 ** 2))
    # 可选：边是否关联缺陷位点
    edge_to_def = np.zeros(len(i_idx), dtype=np.float32)
    if defect_idx is not None:
        for e, (ii, jj) in enumerate(zip(i_idx, j_idx)):
            if ii == defect_idx or jj == defect_idx:
                edge_to_def[e] = 1.0
    edge_attr = torch.cat([
        edge_attr,
        dist / cutoff,
        torch.tensor(edge_to_def, dtype=torch.float).unsqueeze(1),
    ], dim=1)  # 22

    vol = float(atoms.get_volume()) if atoms.get_volume() > 0 else 1.0
    density = n_atoms / vol
    charge = float(sample.get("charge", 0))
    multiplicity = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_elec = float(abs(s1 - s2)) if (s1 is not None and s2 is not None) else 0.0
    global_feat = np.array(
        [charge, multiplicity, delta_elec, n_atoms, density], dtype=np.float32
    )

    # Stage B 位点环境
    site_env = compute_shell_and_angles(
        structure, defect_idx if use_site else None,
        shell_cutoff=SHELL_CUTOFF, angle_rbf_dim=ANGLE_RBF_DIM,
    )
    if not include_shell:
        # 置零 hist + cn + means
        site_env[: N_ELEM_HIST + 4] = 0.0
    if not include_angle:
        site_env[N_ELEM_HIST + 4 :] = 0.0

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.tensor([sample["energy_diff"]], dtype=torch.float),
        global_feat=torch.tensor(global_feat, dtype=torch.float).unsqueeze(0),
        site_env=torch.tensor(site_env, dtype=torch.float).unsqueeze(0),
        mp_id=sample["mp_id"],
        sample_id=sample["id"],
        defect_type=sample["defect_type"],
        defect_idx=torch.tensor(defect_idx if defect_idx is not None else -1, dtype=torch.long),
        use_site_readout=torch.tensor(1 if use_site else 0, dtype=torch.long),
        n_atoms=n_atoms,
    )
    return data

print("\n构建 Stage B 图（含壳层/键角描述子）...")
graph_list = []
kept_targets = []
kept_mp_ids = []
vac_ok = 0
for s in samples:
    try:
        g = atoms_to_graph_stageB(s, include_shell=True, include_angle=True)
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
        kept_mp_ids.append(s["mp_id"])
        if s["defect_type"].startswith("vacancy_") and int(g.use_site_readout) == 1:
            vac_ok += 1
    except Exception as e:
        print(f"  跳过 {s['id']}: {e}")

print(f"成功构建图: {len(graph_list)}, vacancy 位点成功: {vac_ok}")
print(f"节点维: {graph_list[0].x.shape[1]}, 边维: {graph_list[0].edge_attr.shape[1]}, "
      f"全局: {graph_list[0].global_feat.shape[1]}, site_env: {graph_list[0].site_env.shape[1]}")

# 划分
groups = np.array(kept_mp_ids)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(range(len(graph_list)), groups=groups))
train_idx, test_idx = list(train_idx), list(test_idx)
print(f"\n分组划分: Train={len(train_idx)}, Test={len(test_idx)}")
train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])
print(f"  overlap mp_ids={len(train_groups & test_groups)}")

train_targets = np.array([kept_targets[i] for i in train_idx])
y_mean = float(train_targets.mean())
y_std = float(train_targets.std()) + 1e-8

all_x = torch.cat([g.x for g in graph_list], dim=0)
x_mean = all_x.mean(dim=0, keepdim=True)
x_std = all_x.std(dim=0, keepdim=True) + 1e-8
all_gf = torch.cat([g.global_feat for g in graph_list], dim=0)
gf_mean = all_gf.mean(dim=0, keepdim=True)
gf_std = all_gf.std(dim=0, keepdim=True) + 1e-8
all_se = torch.cat([g.site_env for g in graph_list], dim=0)
se_mean = all_se.mean(dim=0, keepdim=True)
se_std = all_se.std(dim=0, keepdim=True) + 1e-8

for g in graph_list:
    g.y = (g.y - y_mean) / y_std
    g.x = (g.x - x_mean) / x_std
    g.global_feat = (g.global_feat - gf_mean) / gf_std
    g.site_env = (g.site_env - se_mean) / se_std

# ============================================================
# 6. 模型：SiteReadout + SiteEnv
# ============================================================
class StageB_SiteEnvGATv2(nn.Module):
    def __init__(self, node_in_dim=8, edge_in_dim=22, global_in_dim=5,
                 site_env_dim=SITE_ENV_DIM, hidden_dim=96, num_heads=4,
                 k_neighbor_pool=True, use_site_env=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbor_pool = k_neighbor_pool
        self.use_site_env = use_site_env

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
            Linear(hidden_dim // 2, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
        )
        self.site_env_encoder = nn.Sequential(
            Linear(site_env_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
        )

        site_in = hidden_dim * 2 if k_neighbor_pool else hidden_dim
        self.site_proj = Linear(site_in, hidden_dim // 2)
        self.graph_proj = Linear(hidden_dim * 2, hidden_dim // 2)

        # 融合：local + global + site_env
        fuse_in = (hidden_dim // 2) * 3 if use_site_env else (hidden_dim // 2) * 2
        self.fusion = nn.Sequential(
            Linear(fuse_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        global_feat = data.global_feat
        site_env = data.site_env
        num_graphs = int(batch.max().item()) + 1

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x1 = F.relu(self.norm1(self.gat1(x, edge_index, edge_attr=edge_attr)))
        x2 = F.relu(self.norm2(self.gat2(x1, edge_index, edge_attr=edge_attr)) + x1)
        x3 = F.relu(self.norm3(self.gat3(x2, edge_index, edge_attr=edge_attr)) + x2)

        global_encoded = self.global_encoder(global_feat)
        env_encoded = self.site_env_encoder(site_env) if self.use_site_env else None

        defect_idx = data.defect_idx
        use_site = data.use_site_readout

        g_mean = global_mean_pool(x3, batch)
        g_max = global_max_pool(x3, batch)
        graph_proj = self.graph_proj(torch.cat([g_mean, g_max], dim=-1))

        counts = scatter(torch.ones_like(batch), batch, dim=0, dim_size=num_graphs, reduce="sum")
        offsets = torch.cumsum(counts, dim=0) - counts
        site_list = []
        for b in range(num_graphs):
            di = int(defect_idx[b].item())
            if use_site[b] == 1 and di >= 0:
                global_i = offsets[b] + di
                v = x3[global_i]
                if self.k_neighbor_pool:
                    src, dst = edge_index
                    e_mask = (batch[src] == b) & (dst == global_i)
                    neigh = src[e_mask]
                    v_nb = x3[neigh].mean(dim=0) if neigh.numel() > 0 else torch.zeros_like(v)
                    v = torch.cat([v, v_nb], dim=-1)
                site_list.append(v)
            else:
                dim = self.hidden_dim * 2 if self.k_neighbor_pool else self.hidden_dim
                site_list.append(torch.zeros(dim, device=x3.device))
        site_feat = torch.stack(site_list, dim=0)
        site_proj = self.site_proj(site_feat)

        use_site_f = use_site.float().unsqueeze(-1)
        local = use_site_f * site_proj + (1.0 - use_site_f) * graph_proj

        if self.use_site_env:
            fused = torch.cat([local, global_encoded, env_encoded], dim=1)
        else:
            fused = torch.cat([local, global_encoded], dim=1)
        out = self.fusion(fused)
        return out.squeeze(-1)

# ============================================================
# 7. 训练
# ============================================================
def rebuild_graphs(include_shell, include_angle):
    graphs = []
    for s in samples:
        try:
            g = atoms_to_graph_stageB(s, include_shell=include_shell, include_angle=include_angle)
            graphs.append(g)
        except Exception:
            continue
    for g in graphs:
        g.y = (g.y - y_mean) / y_std
        g.x = (g.x - x_mean) / x_std
        g.global_feat = (g.global_feat - gf_mean) / gf_std
        g.site_env = (g.site_env - se_mean) / se_std
    return graphs

def train_one(config_name, include_shell=True, include_angle=True, use_site_env=True, seed=0):
    torch.manual_seed(RANDOM_SEED + seed)
    np.random.seed(RANDOM_SEED + seed)

    if include_shell and include_angle and use_site_env:
        graphs = graph_list
    else:
        graphs = rebuild_graphs(include_shell, include_angle)

    train_loader = DataLoader([graphs[i] for i in train_idx if i < len(graphs)],
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader([graphs[i] for i in test_idx if i < len(graphs)],
                             batch_size=BATCH_SIZE)

    model = StageB_SiteEnvGATv2(
        node_in_dim=8,
        edge_in_dim=22,
        global_in_dim=5,
        site_env_dim=SITE_ENV_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        k_neighbor_pool=K_NEIGHBOR_POOL,
        use_site_env=use_site_env,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=12, factor=0.5
    )

    best_val_mae = float("inf")
    best_state = None
    history = []
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_g = 0.0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_g += batch.num_graphs
        avg_loss = total_loss / max(n_g, 1)

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
            wait = 0
        else:
            wait += 1
        scheduler.step(val_mae)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{config_name}] Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val MAE {val_mae:.2f} eV")
        if wait >= PATIENCE:
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
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2, "history": history}

# ============================================================
# 8. 主循环消融
# ============================================================
CONFIGS = OrderedDict([
    ("A_only", {
        "include_shell": False, "include_angle": False, "use_site_env": False,
        "desc": "仅 Stage A 位点读出（无壳层无键角）",
    }),
    ("A_plus_Shell", {
        "include_shell": True, "include_angle": False, "use_site_env": True,
        "desc": "位点读出 + 壳层化学直方图",
    }),
    ("A_plus_Shell_Angle", {
        "include_shell": True, "include_angle": True, "use_site_env": True,
        "desc": "位点读出 + 壳层 + 键角 RBF（Full Stage B）",
    }),
])

print("\n" + "=" * 70)
print("Stage B 实验：配位壳层 + 键角环境")
print("=" * 70)

results = OrderedDict()
t0 = time.time()
for name, cfg in CONFIGS.items():
    print(f"\n>>> {name}: {cfg['desc']}")
    m = train_one(
        name,
        include_shell=cfg["include_shell"],
        include_angle=cfg["include_angle"],
        use_site_env=cfg["use_site_env"],
        seed=0,
    )
    results[name] = {**m, "desc": cfg["desc"]}
    print(f"  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")

elapsed = time.time() - t0
print(f"\n全部完成，耗时 {elapsed/60:.1f} min")

base_mae = results["A_only"]["mae"]
print("\n" + "=" * 90)
print(f"{'Config':<22} {'MAE (eV)':>12} {'RMSE':>10} {'R²':>8} {'Δ vs A':>10}  Description")
print("-" * 90)
for name, r in results.items():
    delta = (r["mae"] - base_mae) / base_mae * 100 if name != "A_only" else 0.0
    delta_str = "  (base)" if name == "A_only" else f"{delta:+.1f}%"
    print(f"{name:<22} {r['mae']:12.2f}  {r['rmse']:10.2f}  {r['r2']:7.4f}  {delta_str:>8}  {r['desc']}")
print("=" * 90)
print("Δ < 0 表示相对 A-only 改善；键角应有非负 Utility（去掉后误差不降）")

csv_path = OUT_DIR / "stageB_coord_env_results.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("config,mae,rmse,r2,delta_vs_A_pct,description\n")
    for name, r in results.items():
        delta = 0.0 if name == "A_only" else (r["mae"] - base_mae) / base_mae * 100
        f.write(f"{name},{r['mae']:.4f},{r['rmse']:.4f},{r['r2']:.4f},{delta:.2f},\"{r['desc']}\"\n")
print(f"\n结果已保存: {csv_path}")

fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(results.keys())
maes = [results[n]["mae"] for n in names]
colors = ["#3498db", "#f39c12", "#2ecc71"]
ax.barh(names[::-1], maes[::-1], color=colors[::-1], edgecolor="k", linewidth=0.6)
ax.axvline(base_mae, color="gray", ls="--", lw=1.5, label="A-only")
ax.set_xlabel("Test MAE (eV)")
ax.set_title("Stage B: Shell hist + Angle RBF ablation")
ax.legend()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig_path = OUT_DIR / "stageB_mae_bar.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"可视化已保存: {fig_path}")

print("\nStage B 脚本完成。验收建议：")
print("  1. A-only vs A+Shell vs A+Shell+Angle 消融表")
print("  2. 抽样打印某 vacancy 的 CN、键角 cos 分布是否与晶体学一致")
print("  3. 仍按 mp_id 分组，无 type one-hot")

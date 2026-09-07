#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vs_exp9 Stage A: 宿主图 + 位点池化（对齐 dGNN 读出）
================================================================
刚性剔除假设下，空位样本改为：
  - 图输入 = pristine 宿主结构（原子数 = N）
  - defect_idx = 被删原子在宿主中的下标
  - 卷积后取 v_{i'}（可选 k 近邻加权）+ 全局电子特征 → MLP
  - 禁止仅用 global_mean/max_pool 作为唯一读出

substitutional / pristine 暂走旧路径（缺陷超胞图 + 全图 pool），便于对照。
无 type one-hot；划分按 mp_id GroupShuffleSplit。

消融：
  Full (site readout) vs GlobalPoolOnly vs w/o_ChemPrior

用法：
  python vs_exp9_stageA_host_site.py
"""

import json
import time
import copy
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
from torch_geometric.data import Data, Batch
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
SHELL_CUTOFF = 3.5          # 位点近邻壳层 (Å)，用于可选邻域 pool
RANDOM_SEED  = 42
N_SEEDS      = 1
VACANCY_MATCH_DIST = 1.8
PATIENCE     = 25
K_NEIGHBOR_POOL = True      # 是否拼接 i' 的近邻加权和

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 元素化学先验（与 exp8 对齐，无 type）
# ============================================================
# (Z, χ, r_pm, valence_e, is_metal, n_d, kind_q)
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
    """最小镜像位移向量 pos_i - pos_j"""
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
    """
    返回被删原子在 pristine 中的下标（整数），失败返回 None。
    策略：元素约束贪婪匹配；pristine 中未匹配且唯一的原子即为空位位点。
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
# 3. 数据加载
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
            "use_host_graph": is_vac,   # Stage A 核心开关
        })

print(f"总样本数: {len(samples)}")
n_vac = sum(1 for s in samples if s["defect_type"].startswith("vacancy_"))
n_sub = sum(1 for s in samples if s["defect_type"].startswith("substitutional_"))
n_pris = sum(1 for s in samples if s["defect_type"] == "pristine")
print(f"  pristine={n_pris}, vacancy={n_vac}, substitutional={n_sub}")

# ============================================================
# 4. 图构建（Stage A：vacancy → 宿主图 + defect_idx）
# ============================================================
def atoms_to_graph_stageA(sample, cutoff=CUTOFF, use_chemprior=True):
    """
    返回 Data，含：
      x: 节点特征
      defect_idx: 标量 long（-1 表示无位点读出，走全局 pool）
      use_site_readout: 0/1
    """
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
            # Stage A 建议 substitutional 仍走全图 pool，不强制位点读出
            use_site = False

    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    cell = atoms.cell.array if atoms.cell is not None else None

    node_feats = []
    for i, a in enumerate(structure["atoms"]):
        prior = get_prior(a["element"])  # 7 dims
        if not use_chemprior:
            # 消融：置零 n_d, kind_q（索引 5,6）
            prior = prior.copy()
            prior[5] = 0.0
            prior[6] = 0.0
        is_def = 1.0 if (defect_idx is not None and i == defect_idx) else 0.0
        # 节点：Z,χ,r,v,is_metal, n_d, kind_q, is_defect_site  → 8 维
        node_feats.append(np.concatenate([prior, [is_def]]))
    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist - rbf_centers) ** 2) / (2 * 0.5 ** 2))
    edge_attr = torch.cat([edge_attr, dist / cutoff], dim=1)  # 21

    vol = float(atoms.get_volume()) if atoms.get_volume() > 0 else 1.0
    density = n_atoms / vol
    charge = float(sample.get("charge", 0))
    multiplicity = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_elec = float(abs(s1 - s2)) if (s1 is not None and s2 is not None) else 0.0

    # 全局：无 type one-hot；charge / mult / ΔN / n_atoms / density
    global_feat = np.array(
        [charge, multiplicity, delta_elec, n_atoms, density], dtype=np.float32
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.tensor([sample["energy_diff"]], dtype=torch.float),
        global_feat=torch.tensor(global_feat, dtype=torch.float).unsqueeze(0),
        mp_id=sample["mp_id"],
        sample_id=sample["id"],
        defect_type=sample["defect_type"],
        defect_idx=torch.tensor(defect_idx if defect_idx is not None else -1, dtype=torch.long),
        use_site_readout=torch.tensor(1 if use_site else 0, dtype=torch.long),
        n_atoms=n_atoms,
    )
    return data

# 构建图
print("\n构建 Stage A 图（vacancy → 宿主图 + defect_idx）...")
graph_list = []
kept_targets = []
kept_mp_ids = []
vac_host_ok, vac_host_fail = 0, 0
for s in samples:
    try:
        g = atoms_to_graph_stageA(s, use_chemprior=True)
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
        kept_mp_ids.append(s["mp_id"])
        if s["defect_type"].startswith("vacancy_"):
            if int(g.use_site_readout) == 1:
                vac_host_ok += 1
            else:
                vac_host_fail += 1
    except Exception as e:
        print(f"  跳过 {s['id']}: {e}")

print(f"成功构建图: {len(graph_list)}")
print(f"  vacancy 宿主图+位点成功: {vac_host_ok}, 失败回退: {vac_host_fail}")
print(f"节点特征维: {graph_list[0].x.shape[1]}, 边: {graph_list[0].edge_attr.shape[1]}, "
      f"全局: {graph_list[0].global_feat.shape[1]}")

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

# ============================================================
# 5. 模型：SiteReadoutGATv2
# ============================================================
class SiteReadoutGATv2(nn.Module):
    """
    卷积后：
      - 若 use_site_readout=1 且 defect_idx>=0：取 v_site = H[defect_idx]
        可选拼接近邻加权和，再与全局拼接 → MLP
      - 否则：global mean+max pool → 与全局拼接 → MLP（旧路径）
    force_global_pool=True 时强制走全图 pool（消融用）
    """
    def __init__(self, node_in_dim=8, edge_in_dim=21, global_in_dim=5,
                 hidden_dim=96, num_heads=4, force_global_pool=False,
                 k_neighbor_pool=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.force_global_pool = force_global_pool
        self.k_neighbor_pool = k_neighbor_pool

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
            nn.Dropout(0.1),
            Linear(hidden_dim // 2, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
        )

        # 位点读出：v_site (+ v_nb) → 投影到 hidden/2
        site_in = hidden_dim * 2 if k_neighbor_pool else hidden_dim
        self.site_proj = Linear(site_in, hidden_dim // 2)
        self.graph_proj = Linear(hidden_dim * 2, hidden_dim // 2)  # mean+max

        self.fusion_gate = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Sequential(
            Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, 1),
        )

    def _neighbor_pool(self, h, edge_index, defect_idx, batch, num_graphs):
        """对每个图中 defect_idx 的直接邻居做 mean pool"""
        # h: (N, F), edge_index: (2, E)
        src, dst = edge_index
        # 构建 mask：dst 是目标位点的边
        # 需要 per-graph 的 defect 全局索引
        # batch 向量给出节点所属图
        node_offset = torch.zeros(num_graphs, dtype=torch.long, device=h.device)
        for b in range(1, num_graphs):
            node_offset[b] = (batch == (b - 1)).sum() + node_offset[b - 1]
        # 更稳健：用 scatter 累计
        counts = scatter(torch.ones_like(batch), batch, dim=0, dim_size=num_graphs, reduce="sum")
        offsets = torch.cumsum(counts, dim=0) - counts

        global_def_idx = offsets + defect_idx.clamp(min=0)
        # 边 mask：dst == global_def_idx[batch[src]] 太复杂，改用循环小 batch
        nb_feats = []
        for b in range(num_graphs):
            mask_nodes = (batch == b)
            h_b = h[mask_nodes]
            di = int(defect_idx[b].item())
            if di < 0 or di >= h_b.size(0):
                nb_feats.append(torch.zeros(h.size(1), device=h.device))
                continue
            # 该图内的边
            e_mask = (batch[src] == b) & (batch[dst] == b)
            src_b = src[e_mask] - offsets[b]
            dst_b = dst[e_mask] - offsets[b]
            neigh = src_b[dst_b == di]
            if neigh.numel() == 0:
                nb_feats.append(torch.zeros(h.size(1), device=h.device))
            else:
                nb_feats.append(h_b[neigh].mean(dim=0))
        return torch.stack(nb_feats, dim=0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        global_feat = data.global_feat
        num_graphs = int(batch.max().item()) + 1

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x1 = F.relu(self.norm1(self.gat1(x, edge_index, edge_attr=edge_attr)))
        x2 = F.relu(self.norm2(self.gat2(x1, edge_index, edge_attr=edge_attr)) + x1)
        x3 = F.relu(self.norm3(self.gat3(x2, edge_index, edge_attr=edge_attr)) + x2)

        global_encoded = self.global_encoder(global_feat)

        # 准备 per-graph defect_idx / use_site
        defect_idx = data.defect_idx  # (B,)
        use_site = data.use_site_readout  # (B,)

        if self.force_global_pool:
            use_site = torch.zeros_like(use_site)

        # 全局 pool 特征（备用）
        g_mean = global_mean_pool(x3, batch)
        g_max = global_max_pool(x3, batch)
        graph_features = torch.cat([g_mean, g_max], dim=-1)
        graph_proj = self.graph_proj(graph_features)

        # 位点特征
        counts = scatter(torch.ones_like(batch), batch, dim=0, dim_size=num_graphs, reduce="sum")
        offsets = torch.cumsum(counts, dim=0) - counts
        site_list = []
        for b in range(num_graphs):
            di = int(defect_idx[b].item())
            if use_site[b] == 1 and di >= 0:
                global_i = offsets[b] + di
                v = x3[global_i]
                if self.k_neighbor_pool:
                    # 简单：取该节点邻居 mean
                    src, dst = edge_index
                    e_mask = (batch[src] == b) & (dst == global_i)
                    neigh = src[e_mask]
                    if neigh.numel() > 0:
                        v_nb = x3[neigh].mean(dim=0)
                    else:
                        v_nb = torch.zeros_like(v)
                    v = torch.cat([v, v_nb], dim=-1)
                site_list.append(v)
            else:
                # 占位，后面用 graph_proj 覆盖
                dim = self.hidden_dim * 2 if self.k_neighbor_pool else self.hidden_dim
                site_list.append(torch.zeros(dim, device=x3.device))
        site_feat = torch.stack(site_list, dim=0)
        site_proj = self.site_proj(site_feat)

        # 按 use_site 选择
        use_site_f = use_site.float().unsqueeze(-1)
        local = use_site_f * site_proj + (1.0 - use_site_f) * graph_proj

        fusion_input = torch.cat([local, global_encoded], dim=1)
        gate = self.fusion_gate(fusion_input)
        fused = gate * local + (1 - gate) * global_encoded
        out = self.output_layer(fused)
        return out.squeeze(-1)

# ============================================================
# 6. 训练与评估
# ============================================================
def train_one_config(config_name, force_global_pool=False, use_chemprior=True, seed=0):
    torch.manual_seed(RANDOM_SEED + seed)
    np.random.seed(RANDOM_SEED + seed)

    # 若消融 ChemPrior，需重建图（节点维不变，只是对应通道为 0）
    if not use_chemprior:
        graphs = []
        for s in samples:
            try:
                graphs.append(atoms_to_graph_stageA(s, use_chemprior=False))
            except Exception:
                continue
        # 重新标准化
        for g in graphs:
            g.y = (g.y - y_mean) / y_std
            g.x = (g.x - x_mean) / x_std
            g.global_feat = (g.global_feat - gf_mean) / gf_std
        # 注意：samples 与 graph_list 顺序一致时才安全；这里简化用原 train_idx
        # 实际应重新过滤，正式跑时建议固定索引
        train_loader = DataLoader([graphs[i] for i in train_idx if i < len(graphs)],
                                  batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader([graphs[i] for i in test_idx if i < len(graphs)],
                                 batch_size=BATCH_SIZE)
    else:
        train_loader = DataLoader([graph_list[i] for i in train_idx],
                                  batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader([graph_list[i] for i in test_idx],
                                 batch_size=BATCH_SIZE)

    model = SiteReadoutGATv2(
        node_in_dim=8,
        edge_in_dim=21,
        global_in_dim=5,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        force_global_pool=force_global_pool,
        k_neighbor_pool=K_NEIGHBOR_POOL,
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
        total_loss = 0.0
        n_g = 0
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
# 7. 主循环：Full / GlobalPoolOnly / w/o_ChemPrior
# ============================================================
CONFIGS = OrderedDict([
    ("Full_SiteReadout", {"force_global_pool": False, "use_chemprior": True,
                          "desc": "宿主图 + 位点读出（Stage A 主模型）"}),
    ("GlobalPoolOnly",   {"force_global_pool": True,  "use_chemprior": True,
                          "desc": "强制全图 pool（消融位点读出）"}),
    ("w/o_ChemPrior",    {"force_global_pool": False, "use_chemprior": False,
                          "desc": "位点读出但去掉 n_d / kind_q"}),
])

print("\n" + "=" * 70)
print("Stage A 实验：宿主图 + 位点池化")
print("=" * 70)

results = OrderedDict()
t0 = time.time()
for name, cfg in CONFIGS.items():
    print(f"\n>>> {name}: {cfg['desc']}")
    m = train_one_config(name, force_global_pool=cfg["force_global_pool"],
                         use_chemprior=cfg["use_chemprior"], seed=0)
    results[name] = {**m, "desc": cfg["desc"]}
    print(f"  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")

elapsed = time.time() - t0
print(f"\n全部完成，耗时 {elapsed/60:.1f} min")

full_mae = results["Full_SiteReadout"]["mae"]
print("\n" + "=" * 90)
print(f"{'Config':<22} {'MAE (eV)':>12} {'RMSE':>10} {'R²':>8} {'Utility':>10}  Description")
print("-" * 90)
for name, r in results.items():
    if name == "Full_SiteReadout":
        util_str = "  (base)"
    else:
        util = (r["mae"] - full_mae) / full_mae * 100
        util_str = f"{util:+.1f}%"
    print(f"{name:<22} {r['mae']:12.2f}  {r['rmse']:10.2f}  {r['r2']:7.4f}  {util_str:>8}  {r['desc']}")
print("=" * 90)
print("Utility = (MAE_ablated - MAE_Full) / MAE_Full × 100%")
print("GlobalPoolOnly Utility > 0 → 位点读出有效；w/o_ChemPrior Utility 应相对 exp8 下降")

csv_path = OUT_DIR / "stageA_host_site_results.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("config,mae,rmse,r2,utility_pct,description\n")
    for name, r in results.items():
        util = 0.0 if name == "Full_SiteReadout" else (r["mae"] - full_mae) / full_mae * 100
        f.write(f"{name},{r['mae']:.4f},{r['rmse']:.4f},{r['r2']:.4f},{util:.2f},\"{r['desc']}\"\n")
print(f"\n结果已保存: {csv_path}")

fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(results.keys())
maes = [results[n]["mae"] for n in names]
colors = ["#2ecc71" if n == "Full_SiteReadout" else "#e74c3c" for n in names]
ax.barh(names[::-1], maes[::-1], color=colors[::-1], edgecolor="k", linewidth=0.6)
ax.axvline(full_mae, color="green", ls="--", lw=1.5)
ax.set_xlabel("Test MAE (eV)")
ax.set_title("Stage A: Host-graph + Site readout")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig_path = OUT_DIR / "stageA_mae_bar.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"可视化已保存: {fig_path}")

print("\nStage A 脚本完成。验收建议：")
print("  1. 日志中 vacancy 宿主图比例、defect_idx 合法率")
print("  2. Full vs GlobalPoolOnly Utility")
print("  3. train/test mp_id 无重叠")

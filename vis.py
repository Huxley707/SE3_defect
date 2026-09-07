#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量注意力可视化（ASE + POV-Ray 出版级版本）
- 直接加载已有权重 defect_aware_gatv2_best.pt
- 原子按注意力热力着色 + 缺陷强高亮
- 每个 pristine 与一个 substitutional 缺陷配对
- 输出：高清 PNG（POV-Ray 渲染）
"""

import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os
# ========== Windows POV-Ray 路径设置 ==========
POVRAY_EXE = r"E:\povray\bin\pvengine.exe"
os.environ["PATH"] = os.path.dirname(POVRAY_EXE) + os.pathsep + os.environ.get("PATH", "")
os.environ["POVRAY"] = POVRAY_EXE
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, LayerNorm, Linear
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase.io import write
from ase.io.pov import get_bondpairs
from ase.data import covalent_radii
import matplotlib.cm as cm

# ============================================================
# 路径
# ============================================================
DATA_PATH = Path("cp2k_gnn_data.json")
CKPT_PATH = Path("artifacts/defect_aware_gatv2_best.pt")
OUT_DIR   = Path("artifacts/attention_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"POV-Ray executable: {POVRAY_EXE}")

# ============================================================
# 1. 数据加载（仅 substitutional + 对应 pristine）
# ============================================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

def parse_dopant(defect_type: str):
    if not defect_type.startswith("substitutional_"):
        return None
    part = defect_type.replace("substitutional_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[0].upper() + part[1:].lower()

systems = {}
for mp_id, sys in raw["systems"].items():
    entry = {"pristine": None, "subs": []}
    if sys.get("pristine") and sys["pristine"].get("has_structure"):
        p = sys["pristine"]
        entry["pristine"] = {
            "id": f"{mp_id}_pristine",
            "energy_diff": 0.0,
            "structure": p["structure"],
            "charge": p.get("inp_charge", 0),
            "multiplicity": p.get("inp_multiplicity", 1),
            "n_elec_s1": p.get("n_electrons_spin1"),
            "n_elec_s2": p.get("n_electrons_spin2"),
            "defect_type": "pristine",
            "dopant": None,
            "mp_id": mp_id,
        }
    for d in sys.get("defects", []):
        if not d.get("has_structure"):
            continue
        dtype = d["defect_type"]
        if not dtype.startswith("substitutional_"):
            continue
        entry["subs"].append({
            "id": f"{mp_id}_{dtype}_q{d.get('charge_from_name', '')}",
            "energy_diff": d.get("energy_diff_ev", d["energy_ev"]),
            "structure": d["structure"],
            "charge": d.get("inp_charge", 0),
            "multiplicity": d.get("inp_multiplicity", 1),
            "n_elec_s1": d.get("n_electrons_spin1"),
            "n_elec_s2": d.get("n_electrons_spin2"),
            "defect_type": dtype,
            "dopant": parse_dopant(dtype),
            "mp_id": mp_id,
        })
    if entry["pristine"] is not None and entry["subs"]:
        systems[mp_id] = entry

print(f"可用配对体系数: {len(systems)}")

samples = []
for entry in systems.values():
    samples.append(entry["pristine"])
    samples.extend(entry["subs"])

# 先读 checkpoint，用训练时的类型列表
_ckpt_preview = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
if "all_defect_types" in _ckpt_preview:
    all_defect_types = list(_ckpt_preview["all_defect_types"])
else:
    all_defect_types = sorted(set(s["defect_type"] for s in samples))
type_to_idx = {t: i for i, t in enumerate(all_defect_types)}
n_types = len(all_defect_types)
print(f"样本数: {len(samples)}, 类型数(与checkpoint对齐): {n_types}")
print(f"  types: {all_defect_types}")

# ============================================================
# 2. 元素特征 / 颜色 / 半径 / 图构建
# ============================================================
element_features = {
    "O":  [8,  3.44, 66, 6, 0], "Ti": [22, 1.54, 160, 4, 1],
    "V":  [23, 1.63, 153, 5, 1], "Cr": [24, 1.66, 139, 6, 1],
    "Fe": [26, 1.83, 132, 8, 1], "Zn": [30, 1.65, 122, 2, 1],
    "Nb": [41, 1.60, 164, 5, 1], "Mo": [42, 2.16, 154, 6, 1],
    "Sn": [50, 1.96, 140, 4, 1], "Hf": [72, 1.30, 175, 4, 1],
    "Ta": [73, 1.50, 170, 5, 1], "W":  [74, 2.36, 162, 6, 1],
}
COVALENT_RADII = {
    "O": 0.66, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Fe": 1.32,
    "Zn": 1.22, "Nb": 1.64, "Mo": 1.54, "Sn": 1.40, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62,
}

def get_atom_feat(el):
    return np.array(element_features.get(el, [0, 0., 0., 0, 0]), dtype=np.float32)

def to_ase(structure):
    syms = [a["element"] for a in structure["atoms"]]
    pos = [[a["x"], a["y"], a["z"]] for a in structure["atoms"]]
    cell = structure.get("lattice_vectors")
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)

def find_defect_idx(structure, dopant):
    if dopant is None:
        return -1
    for i, a in enumerate(structure["atoms"]):
        if a["element"] == dopant:
            return i
    return -1

def build_graph(sample, cutoff=6.0):
    structure = sample["structure"]
    atoms = to_ase(structure)
    n = len(atoms)
    pos = atoms.get_positions()
    scaled = atoms.get_scaled_positions()
    defect_idx = find_defect_idx(structure, sample["dopant"])
    defect_pos = pos[defect_idx] if defect_idx >= 0 else None

    node_feats, node_labels = [], []
    for i, a in enumerate(structure["atoms"]):
        base = get_atom_feat(a["element"])
        frac = scaled[i]
        is_def = 1.0 if i == defect_idx else 0.0
        dist = float(np.linalg.norm(pos[i] - defect_pos)) if defect_pos is not None else 0.0
        node_feats.append(np.concatenate([base, frac, [is_def, dist]]))
        node_labels.append(is_def)
    x = torch.tensor(np.array(node_feats), dtype=torch.float)
    node_label = torch.tensor(node_labels, dtype=torch.float)

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist_t = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist_t - centers)**2) / (2 * 0.5**2))
    edge_attr = torch.cat([edge_attr, dist_t / cutoff], dim=1)

    vol = float(atoms.get_volume()) or 1.0
    n_elem = len(set(a["element"] for a in structure["atoms"]))
    charge = float(sample.get("charge", 0))
    mult = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_e = float(abs(s1 - s2)) if s1 is not None and s2 is not None else 0.0
    type_oh = np.zeros(n_types, dtype=np.float32)
    tname = sample["defect_type"]
    if tname in type_to_idx:
        type_oh[type_to_idx[tname]] = 1.0
    gf = np.concatenate([[n, vol, n/vol, n_elem, charge, mult, delta_e], type_oh]).astype(np.float32)

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        pos=torch.tensor(pos, dtype=torch.float),
        y=torch.tensor([sample["energy_diff"]], dtype=torch.float),
        global_feat=torch.tensor(gf, dtype=torch.float).unsqueeze(0),
        node_label=node_label,
        defect_idx=defect_idx,
        sample_id=sample["id"],
        mp_id=sample["mp_id"],
        symbols=[a["element"] for a in structure["atoms"]],
        cell=torch.tensor(atoms.get_cell().array, dtype=torch.float),
        structure=structure,          # 额外保存原始结构，方便重建 ASE Atoms
    )

graphs = []
for s in samples:
    try:
        graphs.append(build_graph(s))
    except Exception as e:
        print(f"skip {s['id']}: {e}")
print(f"图数量: {len(graphs)}")

# 用 checkpoint 里的统计量做标准化
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
x_mean = torch.tensor(ckpt["x_mean"])
x_std  = torch.tensor(ckpt["x_std"])
gf_mean = torch.tensor(ckpt["gf_mean"])
gf_std  = torch.tensor(ckpt["gf_std"])
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

for g in graphs:
    g.x = (g.x - x_mean) / x_std
    g.global_feat[:, :7] = (g.global_feat[:, :7] - gf_mean) / gf_std
    g.y = (g.y - y_mean) / y_std

# ============================================================
# 3. 模型定义
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

        self.gat1 = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True,
                              edge_dim=hidden, dropout=0.1)
        self.norm1 = LayerNorm(hidden)
        self.gat2 = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True,
                              edge_dim=hidden, dropout=0.1)
        self.norm2 = LayerNorm(hidden)
        self.gat3 = GATv2Conv(hidden, hidden, heads=1, concat=False,
                              edge_dim=hidden, dropout=0.1)
        self.norm3 = LayerNorm(hidden)

        self.global_enc = nn.Sequential(
            Linear(global_in, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden // 2, hidden // 2),
            LayerNorm(hidden // 2)
        )
        self.graph_proj = Linear(hidden * 2, hidden // 2)

        self.fusion_gate = nn.Sequential(
            Linear(hidden, hidden // 2), nn.Sigmoid()
        )
        self.energy_head = nn.Sequential(
            Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden // 2, 1)
        )
        self.node_cls_head = nn.Sequential(
            Linear(hidden, hidden // 2), nn.ReLU(), Linear(hidden // 2, 1)
        )
        self.type_cls_head = nn.Linear(hidden // 2, n_types)

    def forward(self, data, return_attention=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        ea = self.edge_enc(data.edge_attr)
        x = self.node_enc(x)

        if return_attention:
            x1, (ei1, a1) = self.gat1(x, ei, edge_attr=ea, return_attention_weights=True)
            x1 = F.relu(self.norm1(x1))
            x2, (ei2, a2) = self.gat2(x1, ei, edge_attr=ea, return_attention_weights=True)
            x2 = F.relu(self.norm2(x2 + x1))
            x3, (ei3, a3) = self.gat3(x2, ei, edge_attr=ea, return_attention_weights=True)
            x3 = F.relu(self.norm3(x3 + x2))
            att = {"edge_index": ei3, "weights": a3}
        else:
            x1 = F.relu(self.norm1(self.gat1(x, ei, edge_attr=ea)))
            x2 = F.relu(self.norm2(self.gat2(x1, ei, edge_attr=ea) + x1))
            x3 = F.relu(self.norm3(self.gat3(x2, ei, edge_attr=ea) + x2))
            att = None

        g = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], -1)
        g = self.graph_proj(g)
        ge = self.global_enc(data.global_feat)
        gate = self.fusion_gate(torch.cat([g, ge], -1))
        fused = gate * g + (1 - gate) * ge

        energy = self.energy_head(fused).squeeze(-1)
        node_logit = self.node_cls_head(x3).squeeze(-1)
        type_logit = self.type_cls_head(fused)

        if return_attention:
            return energy, node_logit, type_logit, att
        return energy, node_logit, type_logit


_n_types = ckpt.get("n_types", n_types)
_global_in = 7 + _n_types
model = DefectAwareGATv2(
    node_in=10,
    edge_in=21,
    global_in=_global_in,
    hidden=64,
    heads=4,
    n_types=_n_types
).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"已加载权重: {CKPT_PATH}")
print(f"  n_types={_n_types}, global_in={_global_in}, hidden=64")
print(f"  MAE={ckpt.get('mae', 'N/A')}, R²={ckpt.get('r2', 'N/A')}")

# ============================================================
# 4. 注意力提取
# ============================================================
def get_attention(g):
    loader = DataLoader([g], batch_size=1)
    batch = next(iter(loader)).to(DEVICE)
    with torch.no_grad():
        _, _, _, att = model(batch, return_attention=True)
    ei = att["edge_index"].cpu()
    w = att["weights"].cpu()
    if w.dim() > 1:
        w = w.mean(dim=-1)
    n = g.x.shape[0]
    node_att = torch.zeros(n)
    node_att.scatter_add_(0, ei[1], w)
    if node_att.max() > 0:
        node_att = node_att / node_att.max()
    if w.max() > 0:
        w = w / w.max()
    return node_att.numpy(), w.numpy(), ei.numpy()

# ============================================================
# 5. ASE + POV-Ray 高质量渲染
# ============================================================
def attention_to_rgb(node_att, cmap_name="YlOrRd"):
    """0~1 attention → RGB (0~1)"""
    # 使用新的 API
    cmap = plt.get_cmap(cmap_name)  # 或者 matplotlib.colormaps[cmap_name]
    rgba = cmap(np.clip(node_att, 0.0, 1.0))
    return rgba[:, :3]


import subprocess


def render_pov(
        atoms,
        node_att,
        out_prefix,
        defect_idx=-1,
        rotation="15x,25y,5z",
        canvas_width=1200,
        show_bonds=True,
        bond_scale=1.10,
        atom_radius_scale=0.55,
        cmap_name="YlOrRd",
        camera_type="orthographic angle 35",
):
    """
    使用 POV-Ray 生成出版级高清图
    """
    n = len(atoms)

    # 1. 颜色：注意力热力
    colors = attention_to_rgb(node_att, cmap_name)

    # 缺陷原子强制高亮为鲜明蓝色
    if 0 <= defect_idx < n:
        colors[defect_idx] = np.array([0.05, 0.25, 0.95])

    # 2. 半径（共价半径 × 缩放）
    radii = []
    for z in atoms.get_atomic_numbers():
        r = covalent_radii[z] * atom_radius_scale
        radii.append(r)
    radii = np.array(radii)

    if 0 <= defect_idx < n:
        radii[defect_idx] *= 1.30

    # 3. 化学键
    bondatoms = []
    if show_bonds:
        bondatoms = get_bondpairs(atoms, radius=bond_scale)

    # 4. 通用设置
    generic = dict(
        rotation=rotation,
        show_unit_cell=1,
        radii=radii,
        colors=colors,
    )

    # 5. POV-Ray 参数（不包含 povray_executable）
    pov_settings = dict(
        transparent=False,
        background="White",
        canvas_width=canvas_width,
        camera_type=camera_type,
        camera_dist=80.0,
        textures=["jmol"] * n,
        celllinewidth=0.04,
        bondlinewidth=0.08,
        bondatoms=bondatoms,
    )

    # 6. 生成 .pov 文件（但不渲染）
    pov_path = str(out_prefix) + ".pov"
    png_path = str(out_prefix) + ".png"

    try:
        # 只生成 .pov 文件，不自动渲染
        renderer = write(
            pov_path,
            atoms,
            **generic,
            povray_settings=pov_settings,
            run_povray=False,  # 关键：禁止自动渲染
        )

        # 7. 手动调用 POV-Ray 渲染
        print(f"  正在渲染: {Path(pov_path).name} -> {Path(png_path).name}")

        # 使用 pvengine64.exe 命令行参数
        cmd = [
            POVRAY_EXE,
            f"+I{pov_path}",
            f"+O{png_path}",
            f"+W{canvas_width}",
            f"+H{int(canvas_width * 0.9)}",
            "+A",
            "+Q6",
            "-D",
            "/EXIT",

        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 60秒超时
            check=False  # 不自动抛出异常，我们自己处理
        )

        if result.returncode == 0:
            print(f"  ✓ POV-Ray 渲染成功 → {Path(png_path).name}")
            return png_path
        else:
            print(f"  ✗ POV-Ray 渲染失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"    错误信息: {result.stderr[:500]}")
            if result.stdout:
                print(f"    输出信息: {result.stdout[:500]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  ✗ POV-Ray 渲染超时（>60秒）")
        return None
    except Exception as e:
        print(f"  ✗ POV-Ray 渲染失败: {e}")
        return None

# ============================================================
# 6. 生成配对可视化
# ============================================================
print("\n===== 生成配对注意力可视化（POV-Ray） =====")
id2g = {g.sample_id: g for g in graphs}

pairs_done = 0
for mp_id, entry in list(systems.items())[:10]:
    pris_id = entry["pristine"]["id"]
    if pris_id not in id2g:
        continue
    sub = None
    for s in entry["subs"]:
        if s["id"] in id2g and id2g[s["id"]].defect_idx >= 0:
            sub = s
            break
    if sub is None:
        continue

    g_p = id2g[pris_id]
    g_d = id2g[sub["id"]]

    node_att_p, edge_att_p, ei_p = get_attention(g_p)
    node_att_d, edge_att_d, ei_d = get_attention(g_d)

    d_idx = g_d.defect_idx
    att_def = node_att_d[d_idx]
    rank = int((node_att_d >= att_def).sum())
    print(f"\n{mp_id}: defect_idx={d_idx}, att@defect={att_def:.3f}, rank={rank}/{len(node_att_d)}")

    # 从原始 structure 重建 ASE Atoms（保证坐标正确）
    atoms_p = to_ase(g_p.structure)
    atoms_d = to_ase(g_d.structure)

    # 同一视角，方便对比
    rotation = "15x,25y,5z"

    # Pristine
    render_pov(
        atoms_p, node_att_p,
        out_prefix=OUT_DIR / f"pov_pris_{mp_id}",
        defect_idx=-1,
        rotation=rotation,
        canvas_width=1200,
        show_bonds=True,
        cmap_name="YlOrRd",
    )

    # Defect
    render_pov(
        atoms_d, node_att_d,
        out_prefix=OUT_DIR / f"pov_def_{mp_id}_{sub['defect_type']}",
        defect_idx=d_idx,
        rotation=rotation,
        canvas_width=1200,
        show_bonds=True,
        cmap_name="YlOrRd",
    )

    pairs_done += 1
    if pairs_done >= 6:
        break

print(f"\n完成 {pairs_done} 组配对可视化 → {OUT_DIR}")
print("颜色越红注意力越高，蓝色球体为真实缺陷位置。")
print("建议用相同 rotation 的图片左右对比。")
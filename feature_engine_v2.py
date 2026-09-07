#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vs_exp7: 多任务学习 + Mulliken/Hirshfeld 电荷特征增强
====================================================
方案 A 修复版：
  - structure 原子数与 Mulliken/Hirshfeld 原子数不一致时，强制丢弃电荷
  - 避免 72 vs 144 错位特征污染训练
  - 其余与原版相同
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
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
# 0. 路径与全局设置
# ============================================================
DATA_PATH = Path("cp2k_gnn_data.json")
OUT_DIR   = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

HIDDEN_DIM   = 96
NUM_HEADS    = 4
BATCH_SIZE   = 8
MAX_EPOCHS   = 120
LR           = 1e-3
WEIGHT_DECAY = 1e-5
CUTOFF       = 6.0
RANDOM_SEED  = 42
LAMBDA_TYPE  = 0.5
PATIENCE     = 30

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 数据加载 + 方案 A：原子数不一致则丢弃电荷
# ============================================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

def parse_dopant_from_type(defect_type: str):
    if not defect_type.startswith("substitutional_"):
        return None
    part = defect_type.replace("substitutional_", "").split("_")[0]
    return part.capitalize() if len(part) <= 2 else part[:1].upper() + part[1:].lower()

def charge_atom_count(block):
    if not block or "atoms" not in block:
        return 0
    return len(block["atoms"])

def sanitize_charges(sample):
    """
    方案 A 核心：
    structure 原子数与 Mulliken/Hirshfeld 原子数差 > 2 时，丢弃电荷。
    返回 (mulliken, hirshfeld, was_dropped)
    """
    n_struct = len(sample["structure"]["atoms"]) if sample.get("structure") and sample["structure"].get("atoms") else 0
    mull = sample.get("mulliken")
    hirsh = sample.get("hirshfeld")
    n_mull = charge_atom_count(mull)
    n_hirsh = charge_atom_count(hirsh)

    dropped = False
    # 只要有一边对不上，两边都丢掉，避免半套错位特征
    if n_mull > 0 and abs(n_mull - n_struct) > 2:
        mull = None
        dropped = True
    if n_hirsh > 0 and abs(n_hirsh - n_struct) > 2:
        hirsh = None
        dropped = True
    # 一边已经因错位被丢，另一边即使数量碰巧对也一起丢，保持一致
    if dropped:
        mull = None
        hirsh = None
    return mull, hirsh, dropped

samples = []
n_with_charges = 0
n_charge_dropped = 0

for mp_id, system in raw["systems"].items():
    if system.get("pristine") and system["pristine"].get("has_structure"):
        p = system["pristine"]
        sample = {
            "id": f"{mp_id}_pristine",
            "energy_diff": 0.0,
            "structure": p["structure"],
            "charge": p.get("inp_charge", 0),
            "multiplicity": p.get("inp_multiplicity", 1),
            "n_elec_s1": p.get("n_electrons_spin1"),
            "n_elec_s2": p.get("n_electrons_spin2"),
            "defect_type": "pristine",
            "dopant": None,
            "mulliken": p.get("mulliken"),
            "hirshfeld": p.get("hirshfeld"),
        }
        mull, hirsh, dropped = sanitize_charges(sample)
        sample["mulliken"] = mull
        sample["hirshfeld"] = hirsh
        if dropped:
            n_charge_dropped += 1
        if mull or hirsh:
            n_with_charges += 1
        samples.append(sample)

    for d in system.get("defects", []):
        if not d.get("has_structure"):
            continue
        dtype = d["defect_type"]
        sample = {
            "id": f"{mp_id}_{dtype}_q{d.get('charge_from_name', '')}",
            "energy_diff": d.get("energy_diff_ev", d.get("energy_ev", 0.0)),
            "structure": d["structure"],
            "charge": d.get("inp_charge", 0),
            "multiplicity": d.get("inp_multiplicity", 1),
            "n_elec_s1": d.get("n_electrons_spin1"),
            "n_elec_s2": d.get("n_electrons_spin2"),
            "defect_type": dtype,
            "dopant": parse_dopant_from_type(dtype),
            "mulliken": d.get("mulliken"),
            "hirshfeld": d.get("hirshfeld"),
        }
        mull, hirsh, dropped = sanitize_charges(sample)
        sample["mulliken"] = mull
        sample["hirshfeld"] = hirsh
        if dropped:
            n_charge_dropped += 1
        if mull or hirsh:
            n_with_charges += 1
        samples.append(sample)

print(f"总样本数: {len(samples)}")
print(f"因原子数不一致被丢弃电荷的样本: {n_charge_dropped}")
print(f"最终仍保留有效电荷的样本: {n_with_charges}")
all_defect_types = sorted(set(s["defect_type"] for s in samples))
type_to_idx = {t: i for i, t in enumerate(all_defect_types)}
idx_to_type = {i: t for t, i in type_to_idx.items()}
n_types = len(all_defect_types)
print(f"缺陷类型 ({n_types}): {all_defect_types}")

# ============================================================
# 2. 元素特征与图构建
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
    if dopant is None:
        return None
    for i, a in enumerate(structure["atoms"]):
        if a["element"] == dopant:
            return i
    return None

def extract_per_atom_charges(sample, n_atoms):
    """缺失或已被 sanitize 掉时全部填 0"""
    mull_net = np.zeros(n_atoms, dtype=np.float32)
    hirsh_net = np.zeros(n_atoms, dtype=np.float32)
    spin = np.zeros(n_atoms, dtype=np.float32)

    mull = sample.get("mulliken")
    if mull and "atoms" in mull:
        for atom in mull["atoms"]:
            idx = atom["atom_index"] - 1
            if 0 <= idx < n_atoms:
                mull_net[idx] = atom.get("net_charge", 0.0)
                spin[idx] = atom.get("spin_moment", 0.0)

    hirsh = sample.get("hirshfeld")
    if hirsh and "atoms" in hirsh:
        for atom in hirsh["atoms"]:
            idx = atom["atom_index"] - 1
            if 0 <= idx < n_atoms:
                hirsh_net[idx] = atom.get("net_charge", 0.0)

    return mull_net, hirsh_net, spin

def atoms_to_graph(sample, cutoff=CUTOFF):
    structure = sample["structure"]
    energy = sample["energy_diff"]
    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()
    scaled_pos = atoms.get_scaled_positions()

    defect_idx = find_defect_atom_index(structure, sample["dopant"])
    defect_pos = positions[defect_idx] if defect_idx is not None else None

    mull_net, hirsh_net, spin = extract_per_atom_charges(sample, n_atoms)

    node_feats = []
    for i, a in enumerate(structure["atoms"]):
        base = get_atom_features(a["element"])
        frac = scaled_pos[i]
        is_def = 1.0 if (defect_idx is not None and i == defect_idx) else 0.0
        dist = float(np.linalg.norm(positions[i] - defect_pos)) if defect_pos is not None else 0.0
        node_feats.append(np.concatenate([
            base, frac, [is_def, dist],
            [mull_net[i], hirsh_net[i], spin[i]]
        ]))
    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist - rbf_centers)**2) / (2 * 0.5**2))
    edge_attr = torch.cat([edge_attr, dist / cutoff], dim=1)

    vol = float(atoms.get_volume()) if atoms.get_volume() > 0 else 1.0
    density = n_atoms / vol
    n_elements = len(set(a["element"] for a in structure["atoms"]))
    charge = float(sample.get("charge", 0))
    multiplicity = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_elec = float(abs(s1 - s2)) if (s1 is not None and s2 is not None) else 0.0

    abs_mull = np.abs(mull_net)
    mean_abs_q = float(abs_mull.mean()) if n_atoms > 0 else 0.0
    max_abs_q  = float(abs_mull.max()) if n_atoms > 0 else 0.0
    std_q      = float(mull_net.std()) if n_atoms > 1 else 0.0
    defect_q   = float(mull_net[defect_idx]) if defect_idx is not None else 0.0

    global_feat = np.array(
        [n_atoms, vol, density, n_elements, charge, multiplicity, delta_elec,
         mean_abs_q, max_abs_q, std_q, defect_q],
        dtype=np.float32
    )

    type_idx = type_to_idx[sample["defect_type"]]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.tensor([energy], dtype=torch.float),
        y_type=torch.tensor([type_idx], dtype=torch.long),
        global_feat=torch.tensor(global_feat, dtype=torch.float).unsqueeze(0),
    )
    return data

print("\n构建图（方案A：错位电荷已丢弃）...")
graph_list = []
kept_targets = []
kept_types = []
n_graphs_with_charge = 0
for s in samples:
    try:
        g = atoms_to_graph(s)
        if torch.any(g.x[:, 10:13] != 0):
            n_graphs_with_charge += 1
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
        kept_types.append(type_to_idx[s["defect_type"]])
    except Exception as e:
        print(f"  跳过 {s['id']}: {e}")

print(f"成功构建图: {len(graph_list)}")
print(f"其中真正带非零电荷特征的图: {n_graphs_with_charge}")
print(f"节点维: {graph_list[0].x.shape[1]}, 边维: {graph_list[0].edge_attr.shape[1]}, "
      f"全局维: {graph_list[0].global_feat.shape[1]}")

train_idx, test_idx = train_test_split(
    range(len(graph_list)), test_size=0.2, random_state=RANDOM_SEED
)
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

train_targets = np.array([kept_targets[i] for i in train_idx])
y_mean = float(train_targets.mean())
y_std  = float(train_targets.std()) + 1e-8
print(f"能量标准化: mean={y_mean:.3f}, std={y_std:.3f}")

all_x = torch.cat([g.x for g in graph_list], dim=0)
x_mean = all_x.mean(dim=0, keepdim=True)
x_std  = all_x.std(dim=0, keepdim=True) + 1e-8

all_gf = torch.cat([g.global_feat for g in graph_list], dim=0)
gf_mean = all_gf.mean(dim=0, keepdim=True)
gf_std  = all_gf.std(dim=0, keepdim=True) + 1e-8

for g in graph_list:
    g.y = (g.y - y_mean) / y_std
    g.x = (g.x - x_mean) / x_std
    g.global_feat = (g.global_feat - gf_mean) / gf_std

train_loader = DataLoader([graph_list[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader([graph_list[i] for i in test_idx],  batch_size=BATCH_SIZE)

# ============================================================
# 3. 多任务 GATv2
# ============================================================
class MultiTaskGATv2(nn.Module):
    def __init__(self, node_in_dim=13, edge_in_dim=21, global_in_dim=11,
                 hidden_dim=96, num_heads=4, n_types=12):
        super().__init__()
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

        self.energy_head = nn.Sequential(
            Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, 1)
        )

        self.type_head = nn.Sequential(
            Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, n_types)
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
        g_max  = global_max_pool(x3, batch)
        graph_features = torch.cat([g_mean, g_max], dim=-1)

        global_encoded = self.global_encoder(global_feat)
        graph_proj = self.graph_proj(graph_features)

        fusion_input = torch.cat([graph_proj, global_encoded], dim=1)
        gate = self.fusion_gate(fusion_input)
        fused = gate * graph_proj + (1 - gate) * global_encoded

        energy_pred = self.energy_head(fused).squeeze(-1)
        type_logits = self.type_head(fused)
        return energy_pred, type_logits

# ============================================================
# 4. 训练
# ============================================================
model = MultiTaskGATv2(
    node_in_dim=13,
    edge_in_dim=21,
    global_in_dim=11,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    n_types=n_types,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=12, factor=0.5
)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"λ_type = {LAMBDA_TYPE}, MAX_EPOCHS = {MAX_EPOCHS}")
print("\n===== 开始训练（方案A：错位电荷已屏蔽）=====")

best_val_mae = float('inf')
best_state = None
no_improve = 0
history = {"train_loss": [], "val_mae": [], "val_acc": [], "val_r2": []}

t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        energy_pred, type_logits = model(batch)

        loss_e = F.mse_loss(energy_pred, batch.y.view(-1))
        loss_t = F.cross_entropy(type_logits, batch.y_type.view(-1))
        loss = loss_e + LAMBDA_TYPE * loss_t

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    avg_loss = total_loss / len(train_idx)
    history["train_loss"].append(avg_loss)

    model.eval()
    e_preds, e_trues, t_preds, t_trues = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            energy_pred, type_logits = model(batch)
            e_preds.append(energy_pred.cpu().numpy())
            e_trues.append(batch.y.cpu().numpy())
            t_preds.append(type_logits.argmax(dim=-1).cpu().numpy())
            t_trues.append(batch.y_type.cpu().numpy())

    e_preds = np.concatenate(e_preds) * y_std + y_mean
    e_trues = np.concatenate(e_trues) * y_std + y_mean
    t_preds = np.concatenate(t_preds)
    t_trues = np.concatenate(t_trues)

    mae  = mean_absolute_error(e_trues, e_preds)
    r2   = r2_score(e_trues, e_preds)
    acc  = accuracy_score(t_trues, t_preds)

    history["val_mae"].append(mae)
    history["val_acc"].append(acc)
    history["val_r2"].append(r2)

    scheduler.step(mae)

    if mae < best_val_mae:
        best_val_mae = mae
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val MAE {mae:.3f} eV | R² {r2:.4f} | TypeAcc {acc*100:.1f}%")

    if no_improve >= PATIENCE:
        print(f"早停于 epoch {epoch}")
        break

print(f"\n训练耗时: {(time.time()-t0)/60:.1f} min")
print(f"Best Val MAE: {best_val_mae:.3f} eV")

# ============================================================
# 5. 最终评估 + 保存
# ============================================================
model.load_state_dict(best_state)
model.eval()

e_preds, e_trues, t_preds, t_trues = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        energy_pred, type_logits = model(batch)
        e_preds.append(energy_pred.cpu().numpy())
        e_trues.append(batch.y.cpu().numpy())
        t_preds.append(type_logits.argmax(dim=-1).cpu().numpy())
        t_trues.append(batch.y_type.cpu().numpy())

e_preds = np.concatenate(e_preds) * y_std + y_mean
e_trues = np.concatenate(e_trues) * y_std + y_mean
t_preds = np.concatenate(t_preds)
t_trues = np.concatenate(t_trues)

mae  = mean_absolute_error(e_trues, e_preds)
rmse = np.sqrt(mean_squared_error(e_trues, e_preds))
r2   = r2_score(e_trues, e_preds)
acc  = accuracy_score(t_trues, t_preds)

print("\n===== 最终测试结果（方案A）=====")
print(f"Energy  MAE  : {mae:.3f} eV")
print(f"Energy  RMSE : {rmse:.3f} eV")
print(f"Energy  R²   : {r2:.4f}")
print(f"Type    Acc  : {acc*100:.1f}%")
print("\n分类报告:")
print(classification_report(t_trues, t_preds, target_names=all_defect_types, digits=3))

weight_path = OUT_DIR / "charge_augmented_multitask_best.pt"
torch.save({
    "model_state_dict": best_state,
    "y_mean": y_mean,
    "y_std": y_std,
    "x_mean": x_mean,
    "x_std": x_std,
    "gf_mean": gf_mean,
    "gf_std": gf_std,
    "type_to_idx": type_to_idx,
    "idx_to_type": idx_to_type,
    "n_types": n_types,
    "config": {
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "lambda_type": LAMBDA_TYPE,
        "node_in_dim": 13,
        "edge_in_dim": 21,
        "global_in_dim": 11,
        "cutoff": CUTOFF,
        "charge_policy": "drop_on_atom_count_mismatch",
    },
    "metrics": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "type_acc": acc,
        "n_charge_dropped": n_charge_dropped,
        "n_with_valid_charge": n_with_charges,
    }
}, weight_path)
print(f"\n权重已保存: {weight_path}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.scatter(e_trues, e_preds, alpha=0.7, edgecolors='k', linewidths=0.5)
lims = [min(e_trues.min(), e_preds.min()), max(e_trues.max(), e_preds.max())]
ax.plot(lims, lims, 'r--', lw=1.5)
ax.set_xlabel("True Energy (eV)")
ax.set_ylabel("Predicted Energy (eV)")
ax.set_title(f"Energy (Plan A)\nMAE={mae:.2f} eV, R²={r2:.3f}")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(history["val_mae"], label="Val MAE")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE (eV)")
ax.set_title("Energy MAE Curve")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(history["val_r2"], label="Val R²", color="C2")
ax.set_xlabel("Epoch")
ax.set_ylabel("R²")
ax.set_title(f"R² Curve (best={r2:.3f})")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "exp7_planA_result.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"可视化已保存: {fig_path}")

print("\n方案A训练完成。")
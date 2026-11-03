#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FiLM_exp: Feature-wise Linear Modulation of site representations by charge state
================================================================================
Background (decision chain)
---------------------------
Prior Stage A (host-graph + site readout) reached MAE ≈ 66 eV with gated fusion
of the site vector and global electronic features. Stage B (shell histogram +
bond-angle RBF) did not improve upon that baseline. Feature ablations and the
data distribution both indicate that formation energies under the rigid-ion
assumption remain strongly charge-dependent. Parallel / gated fusion treats the
conditioning vector as a competing feature; the present experiment instead lets
the charge-state embedding *modulate* the site representation, following the
FiLM formulation of Perez et al. (AAAI 2018) and its GNN adaptation
(Brockschmidt, ICML 2020):

    h' = γ(c) ⊙ h + β(c),
    γ, β = g(c),

where h is the site (or pooled) representation and c is the global conditioning
vector (charge, spin multiplicity, ΔN_elec, ...).

Configs
-------
  Gate          : Stage-A style soft gate between site and global vectors
  FiLM          : feature-wise affine modulation of the site vector by c
  FiLM_no_charge: same architecture but charge channel zeroed in c
                  (isolates whether modulation utilises charge)

Figures (academic naming, formula rendered)
-------------------------------------------
  FiLM_exp_acceptance_panel.png  – multi-panel parity / residual / metrics
  FiLM_exp_learning_curves.png   – validation MAE trajectories
  FiLM_exp_film_params.png       – γ, β distributions and charge dependence

Usage
-----
  python FiLM_exp.py
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from torch_geometric.utils import scatter
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

# ============================================================
# 0. Paths & hyper-parameters
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
plt.rcParams["figure.dpi"] = 140
plt.rcParams["mathtext.fontset"] = "cm"

HIDDEN_DIM = 96
NUM_HEADS = 4
BATCH_SIZE = 8
MAX_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
CUTOFF = 6.0
RANDOM_SEED = 42
VACANCY_MATCH_DIST = 1.8
PATIENCE = 25
K_NEIGHBOR_POOL = True

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. Element chemical prior (no defect-type one-hot)
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
# 2. Geometry helpers
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
# 3. Data loading
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

print(f"Total samples: {len(samples)}")
n_vac = sum(1 for s in samples if s["defect_type"].startswith("vacancy_"))
n_sub = sum(1 for s in samples if s["defect_type"].startswith("substitutional_"))
n_pris = sum(1 for s in samples if s["defect_type"] == "pristine")
print(f"  pristine={n_pris}, vacancy={n_vac}, substitutional={n_sub}")

# ============================================================
# 4. Graph construction (host graph + site index for vacancies)
# ============================================================
def atoms_to_graph(sample, cutoff=CUTOFF, zero_charge=False):
    use_host = sample.get("use_host_graph", False)
    if use_host and sample.get("pristine_structure") is not None:
        structure = sample["pristine_structure"]
        site_idx = find_vacancy_index_in_pristine(
            sample["pristine_structure"], sample["structure"]
        )
        use_site = site_idx is not None
    else:
        structure = sample["structure"]
        site_idx = None
        use_site = False
        if sample["defect_type"].startswith("substitutional_"):
            site_idx = find_defect_atom_index(structure, sample["dopant"])
            use_site = False

    atoms = structure_to_ase(structure)
    n_atoms = len(atoms)
    positions = atoms.get_positions()

    node_feats = []
    for i, a in enumerate(structure["atoms"]):
        prior = get_prior(a["element"])
        is_def = 1.0 if (site_idx is not None and i == site_idx) else 0.0
        node_feats.append(np.concatenate([prior, [is_def]]))
    x = torch.tensor(np.array(node_feats), dtype=torch.float)

    i_idx, j_idx, d = ase_neighbor_list("ijd", atoms, cutoff)
    if len(i_idx) == 0:
        raise ValueError("no neighbors")
    edge_index = torch.tensor(np.array([i_idx, j_idx]), dtype=torch.long)
    dist = torch.tensor(d, dtype=torch.float).unsqueeze(1)
    rbf_centers = torch.linspace(0, cutoff, 20)
    edge_attr = torch.exp(-((dist - rbf_centers) ** 2) / (2 * 0.5 ** 2))
    edge_attr = torch.cat([edge_attr, dist / cutoff], dim=1)

    vol = float(atoms.get_volume()) if atoms.get_volume() > 0 else 1.0
    density = n_atoms / vol
    charge = 0.0 if zero_charge else float(sample.get("charge", 0))
    multiplicity = float(sample.get("multiplicity", 1))
    s1, s2 = sample.get("n_elec_s1"), sample.get("n_elec_s2")
    delta_elec = float(abs(s1 - s2)) if (s1 is not None and s2 is not None) else 0.0

    # conditioning vector: [charge, multiplicity, ΔN_elec, n_atoms, density]
    cond = np.array([charge, multiplicity, delta_elec, n_atoms, density], dtype=np.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float),
        y=torch.tensor([sample["energy_diff"]], dtype=torch.float),
        global_feat=torch.tensor(cond, dtype=torch.float).unsqueeze(0),
        mp_id=sample["mp_id"],
        sample_id=sample["id"],
        defect_type=sample["defect_type"],
        charge_state=torch.tensor([float(sample.get("charge", 0))], dtype=torch.float),
        defect_idx=torch.tensor(site_idx if site_idx is not None else -1, dtype=torch.long),
        use_site_readout=torch.tensor(1 if use_site else 0, dtype=torch.long),
        n_atoms=n_atoms,
    )
    return data

print("\nBuilding graphs (vacancy → host structure + site index)...")
graph_list = []
kept_targets = []
kept_mp_ids = []
kept_charges = []
vac_ok, vac_fail = 0, 0
for s in samples:
    try:
        g = atoms_to_graph(s, zero_charge=False)
        graph_list.append(g)
        kept_targets.append(s["energy_diff"])
        kept_mp_ids.append(s["mp_id"])
        kept_charges.append(float(s.get("charge", 0)))
        if s["defect_type"].startswith("vacancy_"):
            if int(g.use_site_readout) == 1:
                vac_ok += 1
            else:
                vac_fail += 1
    except Exception as e:
        print(f"  skip {s['id']}: {e}")

print(f"Graphs built: {len(graph_list)}")
print(f"  vacancy host+site success: {vac_ok}, fallback: {vac_fail}")
print(f"  node dim={graph_list[0].x.shape[1]}, edge dim={graph_list[0].edge_attr.shape[1]}, "
      f"cond dim={graph_list[0].global_feat.shape[1]}")

# Grouped split by material id
groups = np.array(kept_mp_ids)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(range(len(graph_list)), groups=groups))
train_idx, test_idx = list(train_idx), list(test_idx)
print(f"\nGrouped split (by material id): Train={len(train_idx)}, Test={len(test_idx)}")
print(f"  train materials={len(set(groups[train_idx]))}, "
      f"test materials={len(set(groups[test_idx]))}, "
      f"overlap={len(set(groups[train_idx]) & set(groups[test_idx]))}")

train_targets = np.array([kept_targets[i] for i in train_idx])
y_mean = float(train_targets.mean())
y_std = float(train_targets.std()) + 1e-8
print(f"Target standardisation: mean={y_mean:.3f}, std={y_std:.3f}")

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
# 5. Model: site readout + fusion variants (Gate / FiLM)
# ============================================================
class SiteReadoutNet(nn.Module):
    """
    Message-passing backbone shared by all configs.
    Fusion is selected by `fusion_mode`:
      - "gate" : soft gate between site and conditioning vectors (Stage-A baseline)
      - "film" : FiLM affine modulation of the site representation
    """
    def __init__(self, node_in_dim=8, edge_in_dim=21, cond_dim=5,
                 hidden_dim=96, num_heads=4, fusion_mode="film",
                 k_neighbor_pool=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.k_neighbor_pool = k_neighbor_pool
        half = hidden_dim // 2

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

        self.cond_encoder = nn.Sequential(
            Linear(cond_dim, half),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(half, half),
            LayerNorm(half),
        )

        site_in = hidden_dim * 2 if k_neighbor_pool else hidden_dim
        self.site_proj = Linear(site_in, half)
        self.graph_proj = Linear(hidden_dim * 2, half)

        if fusion_mode == "gate":
            self.fusion_gate = nn.Sequential(
                Linear(hidden_dim, half),
                nn.Sigmoid(),
            )
        elif fusion_mode == "film":
            # γ, β generators; γ initialised near 1 for stability
            self.film_generator = nn.Sequential(
                Linear(half, half),
                nn.ReLU(),
                Linear(half, half * 2),
            )
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.output_layer = nn.Sequential(
            Linear(half, half),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(half, 1),
        )

    def _site_or_pool(self, h, edge_index, batch, defect_idx, use_site, num_graphs):
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        graph_features = torch.cat([g_mean, g_max], dim=-1)
        graph_vec = self.graph_proj(graph_features)

        counts = scatter(torch.ones_like(batch), batch, dim=0, dim_size=num_graphs, reduce="sum")
        offsets = torch.cumsum(counts, dim=0) - counts
        site_list = []
        for b in range(num_graphs):
            di = int(defect_idx[b].item())
            if use_site[b] == 1 and di >= 0:
                global_i = offsets[b] + di
                v = h[global_i]
                if self.k_neighbor_pool:
                    src, dst = edge_index
                    e_mask = (batch[src] == b) & (dst == global_i)
                    neigh = src[e_mask]
                    v_nb = h[neigh].mean(dim=0) if neigh.numel() > 0 else torch.zeros_like(v)
                    v = torch.cat([v, v_nb], dim=-1)
                site_list.append(v)
            else:
                dim = self.hidden_dim * 2 if self.k_neighbor_pool else self.hidden_dim
                site_list.append(torch.zeros(dim, device=h.device))
        site_feat = torch.stack(site_list, dim=0)
        site_vec = self.site_proj(site_feat)

        use_f = use_site.float().unsqueeze(-1)
        local = use_f * site_vec + (1.0 - use_f) * graph_vec
        return local

    def forward(self, data, return_film=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        cond = data.global_feat
        num_graphs = int(batch.max().item()) + 1

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x1 = F.relu(self.norm1(self.gat1(x, edge_index, edge_attr=edge_attr)))
        x2 = F.relu(self.norm2(self.gat2(x1, edge_index, edge_attr=edge_attr)) + x1)
        x3 = F.relu(self.norm3(self.gat3(x2, edge_index, edge_attr=edge_attr)) + x2)

        cond_enc = self.cond_encoder(cond)
        local = self._site_or_pool(
            x3, edge_index, batch, data.defect_idx, data.use_site_readout, num_graphs
        )

        gamma = beta = None
        if self.fusion_mode == "gate":
            gate_in = torch.cat([local, cond_enc], dim=1)
            gate = self.fusion_gate(gate_in)
            fused = gate * local + (1.0 - gate) * cond_enc
        else:  # film
            params = self.film_generator(cond_enc)
            gamma, beta = params.chunk(2, dim=-1)
            # residual-friendly scale: γ ← 1 + tanh(γ̂)
            gamma = 1.0 + torch.tanh(gamma)
            fused = gamma * local + beta

        out = self.output_layer(fused).squeeze(-1)
        if return_film:
            return out, gamma, beta, local
        return out

# ============================================================
# 6. Training / evaluation
# ============================================================
def train_one(fusion_mode, zero_charge=False, seed=0):
    torch.manual_seed(RANDOM_SEED + seed)
    np.random.seed(RANDOM_SEED + seed)

    # optionally rebuild graphs with charge channel zeroed
    if zero_charge:
        graphs = []
        for s in samples:
            try:
                graphs.append(atoms_to_graph(s, zero_charge=True))
            except Exception:
                continue
        # re-standardise with the same statistics (charge channel becomes constant)
        for g in graphs:
            g.y = (g.y - y_mean) / y_std
            g.x = (g.x - x_mean) / x_std
            g.global_feat = (g.global_feat - gf_mean) / gf_std
        # map original indices carefully
        n = min(len(graphs), len(graph_list))
        tr_loader = DataLoader([graphs[i] for i in train_idx if i < n],
                               batch_size=BATCH_SIZE, shuffle=True)
        te_loader = DataLoader([graphs[i] for i in test_idx if i < n],
                               batch_size=BATCH_SIZE)
        charge_arr = np.array([0.0] * n)  # all zeroed
    else:
        tr_loader = DataLoader([graph_list[i] for i in train_idx],
                               batch_size=BATCH_SIZE, shuffle=True)
        te_loader = DataLoader([graph_list[i] for i in test_idx],
                               batch_size=BATCH_SIZE)
        charge_arr = np.array(kept_charges)

    model = SiteReadoutNet(
        node_in_dim=8, edge_in_dim=21, cond_dim=5,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
        fusion_mode=fusion_mode, k_neighbor_pool=K_NEIGHBOR_POOL,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=12, factor=0.5)

    best_mae = float("inf")
    best_state = None
    history = []
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_g = 0.0, 0
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            n_g += batch.num_graphs

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in te_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                trues.append(batch.y.cpu().numpy())
        preds = np.concatenate(preds) * y_std + y_mean
        trues = np.concatenate(trues) * y_std + y_mean
        mae = mean_absolute_error(trues, preds)
        history.append(mae)
        sched.step(mae)

        if mae < best_mae - 1e-6:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{fusion_mode}{'|noQ' if zero_charge else ''}] "
                  f"Epoch {epoch:3d} | loss {total_loss/max(n_g,1):.4f} | val MAE {mae:.2f} eV")
        if wait >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # final predictions + optional FiLM parameters
    preds, trues, charges, gammas, betas = [], [], [], [], []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(DEVICE)
            if fusion_mode == "film":
                pred, g, b, _ = model(batch, return_film=True)
                gammas.append(g.cpu().numpy())
                betas.append(b.cpu().numpy())
            else:
                pred = model(batch)
            preds.append(pred.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
            charges.append(batch.charge_state.cpu().numpy().reshape(-1))
    preds = np.concatenate(preds) * y_std + y_mean
    trues = np.concatenate(trues) * y_std + y_mean
    charges = np.concatenate(charges)
    gamma_all = np.concatenate(gammas, axis=0) if gammas else None
    beta_all = np.concatenate(betas, axis=0) if betas else None

    metrics = {
        "mae": float(mean_absolute_error(trues, preds)),
        "rmse": float(np.sqrt(mean_squared_error(trues, preds))),
        "r2": float(r2_score(trues, preds)),
        "history": history,
        "preds": preds,
        "trues": trues,
        "charges": charges,
        "gamma": gamma_all,
        "beta": beta_all,
    }
    return metrics

# ============================================================
# 7. Run configs
# ============================================================
CONFIGS = OrderedDict([
    ("Gate",          {"fusion_mode": "gate", "zero_charge": False,
                       "label": "Gated fusion (baseline)"}),
    ("FiLM",          {"fusion_mode": "film", "zero_charge": False,
                       "label": "FiLM modulation"}),
    ("FiLM_no_charge",{"fusion_mode": "film", "zero_charge": True,
                       "label": "FiLM without charge"}),
])

results = OrderedDict()
t0 = time.time()
for name, cfg in CONFIGS.items():
    print(f"\n>>> {name}: {cfg['label']}")
    m = train_one(cfg["fusion_mode"], zero_charge=cfg["zero_charge"], seed=0)
    results[name] = {**m, "label": cfg["label"]}
    print(f"  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R²={m['r2']:.4f}")

print(f"\nAll configs finished in {(time.time()-t0)/60:.1f} min")

# ============================================================
# 8. Figures (academic naming + modulation formula)
# ============================================================
COLORS = {
    "Gate": "#e74c3c",
    "FiLM": "#27ae60",
    "FiLM_no_charge": "#3498db",
}
CHARGE_CMAP = plt.cm.coolwarm

def _charge_norm(c):
    c = np.asarray(c, dtype=float)
    lo, hi = c.min(), c.max()
    if hi - lo < 1e-8:
        return np.zeros_like(c)
    return (c - lo) / (hi - lo)

# ---------- Figure 1: acceptance panel ----------
fig = plt.figure(figsize=(12.5, 9.2))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

# (a) Parity plot – FiLM vs Gate, coloured by charge state
ax = fig.add_subplot(gs[0, 0])
for name in ["Gate", "FiLM"]:
    r = results[name]
    sc = ax.scatter(
        r["trues"], r["preds"],
        c=r["charges"], cmap=CHARGE_CMAP, s=28, alpha=0.75,
        edgecolors="k", linewidths=0.3, label=results[name]["label"],
        marker="o" if name == "FiLM" else "^",
    )
lims = [
    min(results["Gate"]["trues"].min(), results["FiLM"]["trues"].min()),
    max(results["Gate"]["trues"].max(), results["FiLM"]["trues"].max()),
]
pad = 0.05 * (lims[1] - lims[0] + 1)
ax.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad],
        "k--", lw=1.0, alpha=0.6)
ax.set_xlabel(r"DFT formation energy (eV)")
ax.set_ylabel(r"Predicted formation energy (eV)")
ax.set_title("(a) Parity plot (colour = charge state)")
ax.legend(loc="upper left", fontsize=9)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"Charge state $q$")
ax.grid(alpha=0.25)

# (b) Residual versus charge state
ax = fig.add_subplot(gs[0, 1])
width = 0.35
charge_levels = sorted(set(np.round(results["FiLM"]["charges"]).astype(int)))
for i, name in enumerate(["Gate", "FiLM"]):
    r = results[name]
    residuals = r["preds"] - r["trues"]
    data_by_q = []
    positions = []
    for q in charge_levels:
        mask = np.round(r["charges"]).astype(int) == q
        if mask.sum() == 0:
            continue
        data_by_q.append(residuals[mask])
        positions.append(q + (i - 0.5) * width)
    bp = ax.boxplot(
        data_by_q, positions=positions, widths=width * 0.9,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=COLORS[name], alpha=0.55),
        medianprops=dict(color="k", lw=1.2),
    )
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel(r"Charge state $q$")
ax.set_ylabel(r"Residual (pred $-$ DFT) / eV")
ax.set_title("(b) Residual distribution by charge state")
ax.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=COLORS["Gate"], alpha=0.55),
     plt.Rectangle((0, 0), 1, 1, fc=COLORS["FiLM"], alpha=0.55)],
    ["Gated fusion", "FiLM modulation"], fontsize=9, loc="upper right",
)
ax.grid(axis="y", alpha=0.25)

# (c) Metrics comparison (grouped bars, not a lone bar chart)
ax = fig.add_subplot(gs[1, 0])
names = list(results.keys())
x = np.arange(len(names))
w = 0.25
mae_v = [results[n]["mae"] for n in names]
rmse_v = [results[n]["rmse"] for n in names]
# normalise R² to a comparable visual scale (1-R²)*100 for display only
r2_gap = [(1 - results[n]["r2"]) * 100 for n in names]
bars1 = ax.bar(x - w, mae_v, w, label="MAE (eV)", color="#2c3e50", edgecolor="k", lw=0.5)
bars2 = ax.bar(x, rmse_v, w, label="RMSE (eV)", color="#7f8c8d", edgecolor="k", lw=0.5)
ax2 = ax.twinx()
bars3 = ax2.bar(x + w, r2_gap, w, label=r"$(1-R^2)\times 100$", color="#e67e22",
                edgecolor="k", lw=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([results[n]["label"] for n in names], rotation=12, ha="right")
ax.set_ylabel("Error (eV)")
ax2.set_ylabel(r"$(1-R^2)\times 100$")
ax.set_title("(c) Test-set metrics")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.25)
for b, v in zip(bars1, mae_v):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.1f}",
            ha="center", va="bottom", fontsize=8)

# (d) Formula panel + γ mean vs charge
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
formula_text = (
    r"Feature-wise Linear Modulation (FiLM)" "\n\n"
    r"$\mathbf{h}^{\prime}=\boldsymbol{\gamma}(\mathbf{c})\odot\mathbf{h}"
    r"+\boldsymbol{\beta}(\mathbf{c})$" "\n\n"
    r"$\boldsymbol{\gamma},\,\boldsymbol{\beta}=g(\mathbf{c})$" "\n\n"
    r"$\mathbf{h}$: site representation" "\n"
    r"$\mathbf{c}$: conditioning vector" "\n"
    r"(charge state, multiplicity, $\Delta N_{\mathrm{e}}$, …)" "\n\n"
    r"Scale initialisation: $\boldsymbol{\gamma}=1+\tanh(\hat{\boldsymbol{\gamma}})$"
)
ax.text(0.05, 0.95, formula_text, transform=ax.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="serif",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7"))

# small inset: mean |γ| vs charge for FiLM
if results["FiLM"]["gamma"] is not None:
    ax_in = ax.inset_axes([0.55, 0.08, 0.42, 0.38])
    g = results["FiLM"]["gamma"]
    ch = results["FiLM"]["charges"]
    q_unique = sorted(set(np.round(ch).astype(int)))
    mean_abs_g = []
    for q in q_unique:
        mask = np.round(ch).astype(int) == q
        if mask.sum() > 0:
            mean_abs_g.append(np.abs(g[mask]).mean())
        else:
            mean_abs_g.append(np.nan)
    ax_in.plot(q_unique, mean_abs_g, "o-", color="#27ae60", ms=5, lw=1.5)
    ax_in.set_xlabel(r"$q$", fontsize=8)
    ax_in.set_ylabel(r"mean $|\gamma|$", fontsize=8)
    ax_in.tick_params(labelsize=7)
    ax_in.set_title(r"Modulation strength", fontsize=8)
    ax_in.grid(alpha=0.3)

fig.suptitle(
    "Charge-state conditioning via feature-wise linear modulation of the site representation",
    fontsize=13, fontweight="bold", y=0.98,
)
panel_path = OUT_DIR / "FiLM_exp_acceptance_panel.png"
fig.savefig(panel_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {panel_path}")

# ---------- Figure 2: learning curves ----------
fig, ax = plt.subplots(figsize=(8.0, 4.8))
for name, r in results.items():
    ax.plot(r["history"], label=r["label"], color=COLORS[name], lw=2.0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MAE (eV)")
ax.set_title("Learning curves under different fusion schemes")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
curve_path = OUT_DIR / "FiLM_exp_learning_curves.png"
fig.savefig(curve_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {curve_path}")

# ---------- Figure 3: FiLM parameter analysis ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
fig.suptitle(
    r"Analysis of FiLM parameters  "
    r"$\mathbf{h}^{\prime}=\boldsymbol{\gamma}(\mathbf{c})\odot\mathbf{h}+\boldsymbol{\beta}(\mathbf{c})$",
    fontsize=12, fontweight="bold",
)

if results["FiLM"]["gamma"] is not None:
    g = results["FiLM"]["gamma"].ravel()
    b = results["FiLM"]["beta"].ravel()
    ch = results["FiLM"]["charges"]

    # (a) γ histogram
    ax = axes[0]
    ax.hist(g, bins=40, color="#27ae60", alpha=0.75, edgecolor="k", lw=0.4)
    ax.axvline(1.0, color="k", ls="--", lw=1.0, label=r"$\gamma=1$ (identity)")
    ax.set_xlabel(r"Scale parameter $\gamma$")
    ax.set_ylabel("Count")
    ax.set_title(r"(a) Distribution of $\gamma$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (b) β histogram
    ax = axes[1]
    ax.hist(b, bins=40, color="#2980b9", alpha=0.75, edgecolor="k", lw=0.4)
    ax.axvline(0.0, color="k", ls="--", lw=1.0, label=r"$\beta=0$")
    ax.set_xlabel(r"Shift parameter $\beta$")
    ax.set_ylabel("Count")
    ax.set_title(r"(b) Distribution of $\beta$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (c) mean γ per charge state (feature-averaged)
    ax = axes[2]
    g_mat = results["FiLM"]["gamma"]  # (N, F)
    q_unique = sorted(set(np.round(ch).astype(int)))
    means, stds = [], []
    for q in q_unique:
        mask = np.round(ch).astype(int) == q
        if mask.sum() > 0:
            vals = g_mat[mask].mean(axis=1)  # mean over features per sample
            means.append(vals.mean())
            stds.append(vals.std())
        else:
            means.append(np.nan)
            stds.append(0)
    ax.errorbar(q_unique, means, yerr=stds, fmt="o-", color="#c0392b",
                capsize=3, lw=1.5, ms=6)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"Charge state $q$")
    ax.set_ylabel(r"Mean $\gamma$ (feature-averaged)")
    ax.set_title(r"(c) Charge dependence of $\gamma$")
    ax.grid(alpha=0.25)
else:
    for ax in axes:
        ax.text(0.5, 0.5, "No FiLM parameters recorded", ha="center", va="center")

plt.tight_layout()
param_path = OUT_DIR / "FiLM_exp_film_params.png"
fig.savefig(param_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {param_path}")

# ---------- summary table ----------
print("\n" + "=" * 64)
print(f"{'Config':<22} {'MAE (eV)':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 64)
for name, r in results.items():
    print(f"{r['label']:<22} {r['mae']:10.3f} {r['rmse']:10.3f} {r['r2']:8.4f}")
print("=" * 64)
print("FiLM_exp finished.")

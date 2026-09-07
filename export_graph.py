#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 cp2k_gnn_data.json 的 systems 条目中提取原子结构，
生成可供 Gephi 可视化的图数据。

输出目录（默认 artifacts/gephi/）:
  - all_structures.gexf   : 所有结构合并的多连通分量图
  - nodes.csv / edges.csv : 表格形式（Gephi 用 Import Spreadsheet 导入）
  - examples/             : 若干单独结构的 .gexf 示例
  - README_Gephi.txt      : 使用说明

节点 = 原子，边 = 键长 ≤ CUTOFF 的近邻对（已考虑周期性边界条件）
"""

import json
import csv
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

# ============================================================
# 配置
# ============================================================
JSON_PATH = Path("cp2k_gnn_data.json")
OUT_DIR = Path("artifacts/gephi")
EXAMPLE_DIR = OUT_DIR / "examples"

# 键长截断 (Å)。氧化物中金属-O 典型 1.8–2.4 Å，3.0 Å 可覆盖第一近邻壳层
CUTOFF = 3.0

# 单独写出示例 GEXF 的数量
N_EXAMPLES = 6


# ============================================================
# 工具函数
# ============================================================
def lattice_matrix(structure):
    """返回 3×3 晶格矩阵（行向量为晶格矢量）"""
    lv = structure.get("lattice_vectors")
    if lv is None:
        lv = structure["cell"]["lattice_vectors"]
    return np.array(lv, dtype=float)


def minimum_image_distance(pos_i, pos_j, cell, inv_cell):
    """周期性边界条件下的最小镜像距离"""
    d = pos_j - pos_i
    frac = d @ inv_cell
    frac -= np.round(frac)
    d_mic = frac @ cell
    return np.linalg.norm(d_mic)


def build_graph(structure, structure_id, mp_id, defect_type, charge,
                energy_ev=None, energy_diff_ev=None):
    """
    从单个 structure 构建 NetworkX 图。
    节点 = 原子，边 = 距离 < CUTOFF 的原子对（PBC 感知）。
    """
    atoms = structure["atoms"]
    n = len(atoms)
    if n == 0:
        return None

    cell = lattice_matrix(structure)
    inv_cell = np.linalg.inv(cell)
    pos = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)

    G = nx.Graph()
    G.graph["structure_id"] = structure_id
    G.graph["mp_id"] = mp_id
    G.graph["defect_type"] = defect_type
    G.graph["charge"] = charge
    G.graph["n_atoms"] = n
    if energy_ev is not None:
        G.graph["energy_ev"] = energy_ev
    if energy_diff_ev is not None:
        G.graph["energy_diff_ev"] = energy_diff_ev

    # ---- 节点 ----
    for i, a in enumerate(atoms):
        node_id = f"{structure_id}__{i}"
        G.add_node(
            node_id,
            label=f"{a.get('element', 'X')}{i}",
            element=a.get("element", "X"),
            kind=a.get("kind", a.get("element", "X")),
            x=float(a["x"]),
            y=float(a["y"]),
            z=float(a["z"]),
            atom_index=i,
            structure_id=structure_id,
            mp_id=mp_id,
            defect_type=defect_type,
            charge=charge,
            energy_ev=energy_ev if energy_ev is not None else "",
            energy_diff_ev=energy_diff_ev if energy_diff_ev is not None else "",
        )

    # ---- 边（展开一层周期像，用 KDTree 找候选对）----
    images = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                images.append(ix * cell[0] + iy * cell[1] + iz * cell[2])
    images = np.array(images)

    all_pos, all_owner = [], []
    for i in range(n):
        for img in images:
            all_pos.append(pos[i] + img)
            all_owner.append(i)
    all_pos = np.array(all_pos)

    tree = cKDTree(all_pos)
    pairs = tree.query_pairs(r=CUTOFF)

    edge_set = set()
    for p, q in pairs:
        i, j = all_owner[p], all_owner[q]
        if i >= j:
            continue
        dist = minimum_image_distance(pos[i], pos[j], cell, inv_cell)
        if dist < 1e-4 or dist > CUTOFF:
            continue
        key = (i, j)
        if key not in edge_set:
            edge_set.add(key)
            u = f"{structure_id}__{i}"
            v = f"{structure_id}__{j}"
            G.add_edge(u, v, weight=round(dist, 4), distance=round(dist, 4))

    return G


def write_gexf(G, path):
    nx.write_gexf(G, path)


def graph_to_csv_rows(G, nodes_rows, edges_rows):
    for n, attr in G.nodes(data=True):
        nodes_rows.append({
            "Id": n,
            "Label": attr.get("label", n),
            "element": attr.get("element"),
            "kind": attr.get("kind"),
            "x": attr.get("x"),
            "y": attr.get("y"),
            "z": attr.get("z"),
            "atom_index": attr.get("atom_index"),
            "structure_id": attr.get("structure_id"),
            "mp_id": attr.get("mp_id"),
            "defect_type": attr.get("defect_type"),
            "charge": attr.get("charge"),
            "energy_ev": attr.get("energy_ev"),
            "energy_diff_ev": attr.get("energy_diff_ev"),
        })
    for u, v, attr in G.edges(data=True):
        edges_rows.append({
            "Source": u,
            "Target": v,
            "Type": "Undirected",
            "Weight": attr.get("weight", 1.0),
            "distance": attr.get("distance"),
            "structure_id": G.nodes[u].get("structure_id"),
        })


# ============================================================
# 主流程
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading JSON ...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    systems = data["systems"]
    print(f"Systems: {len(systems)}")

    G_all = nx.Graph()
    G_all.graph["description"] = (
        "All CP2K defect structures (nodes=atoms, edges=bonds < {:.1f} Å)".format(CUTOFF)
    )

    nodes_rows = []
    edges_rows = []
    example_count = 0
    n_ok = 0
    n_fail = 0

    for mp_id, system in systems.items():
        # ---------- pristine ----------
        if system.get("pristine") and system["pristine"].get("has_structure"):
            p = system["pristine"]
            sid = f"{mp_id}_pristine_q{p.get('charge_from_name', '0')}"
            try:
                G = build_graph(
                    p["structure"], sid, mp_id,
                    defect_type="pristine",
                    charge=p.get("inp_charge", 0),
                    energy_ev=p.get("energy_ev"),
                    energy_diff_ev=0.0,
                )
                if G is not None:
                    G_all = nx.compose(G_all, G)
                    graph_to_csv_rows(G, nodes_rows, edges_rows)
                    n_ok += 1
                    if example_count < N_EXAMPLES:
                        try:
                            write_gexf(G, EXAMPLE_DIR / f"{sid}.gexf")
                            example_count += 1
                            print(f"  example: {sid}  (n={G.number_of_nodes()}, e={G.number_of_edges()})")
                        except OSError as oe:
                            print(f"  (skip example write {sid}: {oe})")
            except Exception as e:
                print(f"  FAIL {sid}: {e}")
                n_fail += 1

        # ---------- defects ----------
        for d in system.get("defects", []):
            if not d.get("has_structure"):
                continue
            dtype = d.get("defect_type", "unknown")
            q = d.get("charge_from_name", str(d.get("inp_charge", "")))
            # 避免 defect_type 已以 _q 结尾时出现双 _q
            if dtype.endswith("_q"):
                sid = f"{mp_id}_{dtype}{q}"
            else:
                sid = f"{mp_id}_{dtype}_q{q}"
            try:
                G = build_graph(
                    d["structure"], sid, mp_id,
                    defect_type=dtype,
                    charge=d.get("inp_charge", 0),
                    energy_ev=d.get("energy_ev"),
                    energy_diff_ev=d.get("energy_diff_ev"),
                )
                if G is not None:
                    G_all = nx.compose(G_all, G)
                    graph_to_csv_rows(G, nodes_rows, edges_rows)
                    n_ok += 1
                    if example_count < N_EXAMPLES:
                        try:
                            write_gexf(G, EXAMPLE_DIR / f"{sid}.gexf")
                            example_count += 1
                            print(f"  example: {sid}  (n={G.number_of_nodes()}, e={G.number_of_edges()})")
                        except OSError as oe:
                            print(f"  (skip example write {sid}: {oe})")
            except Exception as e:
                print(f"  FAIL {sid}: {e}")
                n_fail += 1

    print(f"\nSuccessfully built: {n_ok} structures  (failed: {n_fail})")
    print(f"Combined graph: {G_all.number_of_nodes()} nodes, {G_all.number_of_edges()} edges")

    # ---- 写出合并 GEXF ----
    gexf_path = OUT_DIR / "all_structures.gexf"
    print(f"Writing {gexf_path} ...")
    write_gexf(G_all, gexf_path)

    # ---- 写出 CSV ----
    nodes_path = OUT_DIR / "nodes.csv"
    edges_path = OUT_DIR / "edges.csv"

    if nodes_rows:
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(nodes_rows[0].keys()))
            w.writeheader()
            w.writerows(nodes_rows)
        print(f"Wrote {nodes_path} ({len(nodes_rows)} nodes)")

    if edges_rows:
        with open(edges_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(edges_rows[0].keys()))
            w.writeheader()
            w.writerows(edges_rows)
        print(f"Wrote {edges_path} ({len(edges_rows)} edges)")

    # ---- 说明文件 ----
    readme = OUT_DIR / "README_Gephi.txt"
    readme.write_text(
        f"""Gephi 可视化说明
================

推荐导入方式（最简单）:
1. 打开 Gephi
2. File → Import Spreadsheet → 选择 nodes.csv
   - 选择 "Nodes table"
   - 确认 Id 列作为节点 ID
3. 再次 Import Spreadsheet → 选择 edges.csv
   - 选择 "Edges table"
   - Source / Target 对应边的两端

或者直接打开 all_structures.gexf（File → Open）

节点属性（可用来过滤 / 上色 / 调整大小）:
  element, kind, x, y, z, structure_id, mp_id, defect_type, charge,
  energy_ev, energy_diff_ev

边属性:
  distance (Å), weight (= distance)

建议操作:
- Appearance → Nodes → Color → Partition → element   （按元素着色）
- Appearance → Nodes → Size → Ranking → degree       （按配位数调整大小）
- Filters → Attributes → Equal → structure_id        （只看某一个结构）
- Layout → ForceAtlas2 或 Yifan Hu                    （力导向布局）

examples/ 目录下有几个单独结构的 .gexf，可直接打开快速查看。

键长截断: {CUTOFF} Å（已考虑周期性边界条件）
""",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")
    print("Done.")


if __name__ == "__main__":
    main()

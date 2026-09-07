#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CP2K output + input (.inp) parser
for pristine + defect calculations → Graph Neural Network training data

支持：
- 普通能量计算
- Mulliken / Hirshfeld 电荷分析（max_scf=1 restart 轻量计算）
- 自动把电荷信息 merge 进已有 cp2k_gnn_data.json
"""

import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np

# 1 Hartree = 27.211386245988 eV
HARTREE_TO_EV = 27.211386245988


def abc_angles_to_matrix(a: float, b: float, c: float,
                         alpha: float, beta: float, gamma: float) -> np.ndarray:
    """把 abc + 角度 转换成 3x3 晶胞矩阵（行向量形式），角度单位：度"""
    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    bx = b * np.cos(gamma)
    by = b * np.sin(gamma)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    matrix = np.array([
        [a,   0.0, 0.0],
        [bx,  by,  0.0],
        [cx,  cy,  cz ]
    ])
    return matrix


def parse_cell(content: str) -> Dict:
    """健壮的 CELL 解析，支持 A/B/C 向量 和 ABC+角度 两种写法"""
    result = {
        "lattice_vectors": None,
        "abc": None,
        "angles": None,
        "periodic": True
    }

    match = re.search(r'&CELL\b(.*?)&END\s*CELL', content, re.DOTALL | re.IGNORECASE)
    if not match:
        return result

    block = match.group(1)

    # 1. 尝试 A B C 向量
    vectors = {}
    for name in ['A', 'B', 'C']:
        pattern = rf'{name}\s+([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s+([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s+([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)'
        m = re.search(pattern, block, re.IGNORECASE)
        if m:
            vectors[name] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

    if len(vectors) == 3:
        result["lattice_vectors"] = [vectors['A'], vectors['B'], vectors['C']]
        return result

    # 2. 尝试 ABC + ALPHA_BETA_GAMMA
    m = re.search(
        r'ABC\s+([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s+([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)\s+([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)',
        block, re.IGNORECASE
    )
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        result["abc"] = [a, b, c]

    m = re.search(
        r'ALPHA_BETA_GAMMA\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)',
        block, re.IGNORECASE
    )
    if m:
        alpha, beta, gamma = float(m.group(1)), float(m.group(2)), float(m.group(3))
        result["angles"] = [alpha, beta, gamma]

    if result["abc"] is not None and result["angles"] is not None:
        matrix = abc_angles_to_matrix(
            result["abc"][0], result["abc"][1], result["abc"][2],
            result["angles"][0], result["angles"][1], result["angles"][2]
        )
        result["lattice_vectors"] = matrix.tolist()

    m = re.search(r'PERIODIC\s+(\S+)', block, re.IGNORECASE)
    if m:
        result["periodic"] = m.group(1).upper() != "NONE"

    return result


# ============================================================
# 1. 解析 CP2K .inp 文件
# ============================================================
def parse_cp2k_inp(filepath: str) -> Dict[str, Any]:
    """解析 CP2K 输入文件，提取结构与关键设置"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    info: Dict[str, Any] = {
        "inp_filepath": str(filepath),
        "inp_filename": Path(filepath).name,
        "cell": None,
        "lattice_vectors": None,
        "atoms": [],
        "n_atoms_inp": 0,
        "kinds": {},
        "charge": None,
        "multiplicity": None,
        "coord_unit": "angstrom",
        "periodic": True,
        "run_type": None,
        "project_name": None,
    }

    # GLOBAL
    m = re.search(r'&GLOBAL.*?PROJECT(?:_NAME)?\s+(\S+)', content, re.DOTALL | re.IGNORECASE)
    if m:
        info["project_name"] = m.group(1)

    m = re.search(r'&GLOBAL.*?RUN_TYPE\s+(\S+)', content, re.DOTALL | re.IGNORECASE)
    if m:
        info["run_type"] = m.group(1)

    # 电荷与多重度
    m = re.search(r'&DFT.*?CHARGE\s+([-+]?\d+)', content, re.DOTALL | re.IGNORECASE)
    if m:
        info["charge"] = int(m.group(1))
    m = re.search(r'&DFT.*?MULTIPLICITY\s+(\d+)', content, re.DOTALL | re.IGNORECASE)
    if m:
        info["multiplicity"] = int(m.group(1))

    # CELL
    cell_info = parse_cell(content)
    info["cell"] = cell_info
    info["lattice_vectors"] = cell_info["lattice_vectors"]

    # COORD
    coord_match = re.search(r'&COORD\b(.*?)&END\s*COORD', content, re.DOTALL | re.IGNORECASE)
    if coord_match:
        coord_block = coord_match.group(1)

        m = re.search(r'UNIT\s+(\S+)', coord_block, re.IGNORECASE)
        if m:
            info["coord_unit"] = m.group(1).lower()

        atom_pattern = re.compile(
            r'^[ \t]*([A-Z][a-z]?)[ \t]+'
            r'([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)[ \t]+'
            r'([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)[ \t]+'
            r'([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)'
            r'(?:[ \t]+(\S+))?[ \t]*$'
        )
        for raw_line in coord_block.splitlines():
            line = raw_line.rstrip('\r')
            m = atom_pattern.match(line)
            if not m:
                continue
            atom = {
                "element": m.group(1),
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "z": float(m.group(4)),
            }
            if m.group(5):
                atom["kind"] = m.group(5)
            info["atoms"].append(atom)

        info["n_atoms_inp"] = len(info["atoms"])

    # KIND
    kind_blocks = re.finditer(r'&KIND\s+(\S+)(.*?)&END\s*KIND', content, re.DOTALL | re.IGNORECASE)
    for km in kind_blocks:
        kind_name = km.group(1)
        block = km.group(2)
        kind_info = {}

        m = re.search(r'BASIS_SET\s+(\S+)', block, re.IGNORECASE)
        if m:
            kind_info["basis_set"] = m.group(1)
        m = re.search(r'POTENTIAL\s+(\S+)', block, re.IGNORECASE)
        if m:
            kind_info["potential"] = m.group(1)
        m = re.search(r'ELEMENT\s+(\S+)', block, re.IGNORECASE)
        if m:
            kind_info["element"] = m.group(1)
        else:
            kind_info["element"] = kind_name

        info["kinds"][kind_name] = kind_info

    return info


# ============================================================
# 2. 解析 Mulliken / Hirshfeld
# ============================================================
def parse_mulliken(content: str) -> Optional[Dict]:
    """解析 Mulliken Population Analysis"""
    # 找到 Mulliken 段落
    m = re.search(
        r'Mulliken Population Analysis\s*\n\s*\n'
        r'#\s*Atom\s+Element\s+Kind\s+Atomic population \(alpha,beta\)\s+Net charge\s+Spin moment\s*\n'
        r'(.*?)'
        r'#\s*Total charge and spin\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
        content, re.DOTALL | re.IGNORECASE
    )
    if not m:
        # 兼容更宽松的匹配
        m = re.search(
            r'Mulliken Population Analysis.*?Net charge\s+Spin moment\s*\n(.*?)#\s*Total charge and spin\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
            content, re.DOTALL | re.IGNORECASE
        )
    if not m:
        return None

    atoms_block = m.group(1)
    total_alpha = float(m.group(2))
    total_beta  = float(m.group(3))
    total_net   = float(m.group(4))
    total_spin  = float(m.group(5))

    atoms = []
    # 每行:   1     V        1         6.113842     6.113842     0.772317    -0.000000
    line_pat = re.compile(
        r'^\s*(\d+)\s+([A-Z][a-z]?)\s+(\d+)\s+'
        r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+'
        r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
        re.MULTILINE
    )
    for lm in line_pat.finditer(atoms_block):
        atoms.append({
            "atom_index": int(lm.group(1)),
            "element": lm.group(2),
            "kind": int(lm.group(3)),
            "population_alpha": float(lm.group(4)),
            "population_beta": float(lm.group(5)),
            "net_charge": float(lm.group(6)),
            "spin_moment": float(lm.group(7)),
        })

    return {
        "atoms": atoms,
        "total_alpha": total_alpha,
        "total_beta": total_beta,
        "total_net_charge": total_net,
        "total_spin": total_spin,
        "n_atoms": len(atoms)
    }


def parse_hirshfeld(content: str) -> Optional[Dict]:
    """解析 Hirshfeld Charges"""
    m = re.search(
        r'Hirshfeld Charges\s*\n\s*\n'
        r'\s*#Atom\s+Element\s+Kind\s+Ref Charge\s+Population\s+Spin moment\s+Net charge\s*\n'
        r'(.*?)'
        r'Total Charge\s+([-+]?\d+\.\d+)',
        content, re.DOTALL | re.IGNORECASE
    )
    if not m:
        # 宽松匹配
        m = re.search(
            r'Hirshfeld Charges.*?Net charge\s*\n(.*?)Total Charge\s+([-+]?\d+\.\d+)',
            content, re.DOTALL | re.IGNORECASE
        )
    if not m:
        return None

    atoms_block = m.group(1)
    total_charge = float(m.group(2))

    atoms = []
    # 每行:      1       V      1      13.000    6.888   6.888           -0.000     -0.776
    # 注意 Population 可能是两个数（α β）或一个数
    line_pat = re.compile(
        r'^\s*(\d+)\s+([A-Z][a-z]?)\s+(\d+)\s+'
        r'([-+]?\d+\.\d+)\s+'
        r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+'
        r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
        re.MULTILINE
    )
    for lm in line_pat.finditer(atoms_block):
        atoms.append({
            "atom_index": int(lm.group(1)),
            "element": lm.group(2),
            "kind": int(lm.group(3)),
            "ref_charge": float(lm.group(4)),
            "population_alpha": float(lm.group(5)),
            "population_beta": float(lm.group(6)),
            "spin_moment": float(lm.group(7)),
            "net_charge": float(lm.group(8)),
        })

    return {
        "atoms": atoms,
        "total_charge": total_charge,
        "n_atoms": len(atoms)
    }


# ============================================================
# 3. 解析 CP2K .output 文件
# ============================================================
def parse_cp2k_output(filepath: str) -> Dict[str, Any]:
    """解析单个 CP2K output 文件（支持普通能量计算 + Mulliken/Hirshfeld）"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    info: Dict[str, Any] = {
        "filepath": str(filepath),
        "filename": Path(filepath).name,
        "success": False,
        "energy_au": None,
        "energy_ev": None,
        "n_atoms": None,
        "n_kinds": None,
        "n_electrons_spin1": None,
        "n_electrons_spin2": None,
        "scf_steps": None,
        "scf_converged": False,
        "final_convergence": None,
        "project_name": None,
        "mp_id": None,
        "defect_type": None,
        "charge_from_name": None,
        "run_time_s": None,
        "mpi_processes": None,
        "omp_threads": None,
        "warnings": 0,
        "mulliken": None,
        "hirshfeld": None,
        "has_charges": False,
    }

    # Project name
    m = re.search(r'GLOBAL\| Project name\s+(\S+)', content)
    if m:
        info["project_name"] = m.group(1)

    # mp_id
    m = re.search(r'(mp-\d+)', Path(filepath).name)
    if m:
        info["mp_id"] = m.group(1)
    else:
        m = re.search(r'(mp-\d+)', info.get("project_name") or "")
        if m:
            info["mp_id"] = m.group(1)

    # defect_type & charge
    fname = Path(filepath).name.lower()
    if "pristine" in fname:
        info["defect_type"] = "pristine"
    else:
        m = re.search(r'(substitutional_\w+|vacancy_\w+|interstitial_\w+|defect_\w+)', fname)
        info["defect_type"] = m.group(1) if m else "defect"

    m = re.search(r'q([+-]?\d+)', fname)
    info["charge_from_name"] = m.group(1) if m else "0"

    # 原子数 / kind
    m = re.search(r'-\s*Atoms:\s+(\d+)', content)
    if m:
        info["n_atoms"] = int(m.group(1))
    m = re.search(r'-\s*Atomic kinds:\s+(\d+)', content)
    if m:
        info["n_kinds"] = int(m.group(1))

    # 电子数
    m = re.search(r'Spin 1\s+Number of electrons:\s+(\d+)', content)
    if m:
        info["n_electrons_spin1"] = int(m.group(1))
    m = re.search(r'Spin 2\s+Number of electrons:\s+(\d+)', content)
    if m:
        info["n_electrons_spin2"] = int(m.group(1))

    # 能量（优先 FORCE_EVAL，其次 Total energy）
    m = re.search(
        r'ENERGY\|\s+Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+([-+]?\d+\.\d+)',
        content
    )
    if m:
        energy_au = float(m.group(1))
        info["energy_au"] = energy_au
        info["energy_ev"] = energy_au * HARTREE_TO_EV
        info["success"] = True
    else:
        m = re.search(r'Total energy:\s+([-+]?\d+\.\d+)', content)
        if m:
            energy_au = float(m.group(1))
            info["energy_au"] = energy_au
            info["energy_ev"] = energy_au * HARTREE_TO_EV
            info["success"] = True

    # SCF
    m = re.search(r'\*\*\* SCF run converged in\s+(\d+) steps \*\*\*', content)
    if m:
        info["scf_steps"] = int(m.group(1))
        info["scf_converged"] = True

    steps = re.findall(
        r'^\s*\d+\s+OT\s+\w+\s+[\d.E+-]+\s+[\d.]+\s+([\d.E+-]+)\s+([-+]?\d+\.\d+)',
        content, re.MULTILINE
    )
    if steps:
        info["final_convergence"] = float(steps[-1][0])

    # 运行时间
    start = re.search(r'PROGRAM STARTED AT\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', content)
    end = re.search(r'PROGRAM ENDED AT\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', content)
    if start and end:
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        t0 = datetime.strptime(start.group(1), fmt)
        t1 = datetime.strptime(end.group(1), fmt)
        info["run_time_s"] = (t1 - t0).total_seconds()

    # MPI / OMP
    m = re.search(r'GLOBAL\| Total number of message passing processes\s+(\d+)', content)
    if m:
        info["mpi_processes"] = int(m.group(1))
    m = re.search(r'GLOBAL\| Number of threads for this process\s+(\d+)', content)
    if m:
        info["omp_threads"] = int(m.group(1))

    # warnings
    m = re.search(r'The number of warnings for this run is :\s+(\d+)', content)
    if m:
        info["warnings"] = int(m.group(1))

    # ========== 新增：Mulliken / Hirshfeld ==========
    mull = parse_mulliken(content)
    if mull:
        info["mulliken"] = mull
        info["has_charges"] = True

    hirsh = parse_hirshfeld(content)
    if hirsh:
        info["hirshfeld"] = hirsh
        info["has_charges"] = True

    return info


# ============================================================
# 4. 文件名匹配工具
# ============================================================
def get_base_name(filename: str) -> str:
    """
    从 output 或 inp 文件名中提取核心 basename，用于匹配。
    自动去掉 _mull、.inp.output、.output、.out、.inp 等后缀。
    """
    name = Path(filename).name
    # 先去掉常见输出后缀
    for suffix in ['.inp.output', '.output', '.out', '.inp', '.in']:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    # 再去掉 _mull（支持 _mull 和 _mull.xxx）
    if name.lower().endswith('_mull'):
        name = name[:-5]
    return name


def find_matching_inp(output_path: Path, inp_dir: Path) -> Optional[Path]:
    """在指定的 inp_dir 中寻找与 output 匹配的 .inp 文件"""
    base = get_base_name(output_path.name)

    candidates = [
        inp_dir / f"{base}.inp",
        inp_dir / f"{base}.in",
    ]

    if not any(c.exists() for c in candidates):
        for ext in ['.inp', '.in']:
            found = list(inp_dir.rglob(f"{base}{ext}"))
            if found:
                return found[0]

    for c in candidates:
        if c.exists():
            return c

    return None


# ============================================================
# 5. 批量处理主函数
# ============================================================
def process_directory(
    output_dir: str,
    inp_dir: str,
    output_csv: str = "cp2k_gnn_summary.csv",
    output_json: str = "cp2k_gnn_data.json",
    recursive: bool = True
) -> Tuple[List[Dict], Dict]:
    """批量处理普通 output 或 mull output"""
    output_path = Path(output_dir)
    inp_path = Path(inp_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"output 目录不存在: {output_dir}")
    if not inp_path.exists():
        raise FileNotFoundError(f"inp 目录不存在: {inp_dir}")

    pattern = "**/*" if recursive else "*"
    files = list(output_path.glob(f"{pattern}.output")) + list(output_path.glob(f"{pattern}.out"))

    results = []
    for f in sorted(files):
        print(f"Parsing output: {f.name}")
        try:
            out_info = parse_cp2k_output(str(f))

            matched_inp = find_matching_inp(f, inp_path)
            if matched_inp:
                print(f"  → matched inp: {matched_inp}")
                try:
                    inp_info = parse_cp2k_inp(str(matched_inp))
                    out_info["structure"] = {
                        "lattice_vectors": inp_info.get("lattice_vectors"),
                        "cell": inp_info.get("cell"),
                        "atoms": inp_info.get("atoms"),
                        "n_atoms_inp": inp_info.get("n_atoms_inp"),
                        "kinds": inp_info.get("kinds"),
                        "coord_unit": inp_info.get("coord_unit"),
                        "periodic": inp_info.get("periodic"),
                    }
                    out_info["inp_charge"] = inp_info.get("charge")
                    out_info["inp_multiplicity"] = inp_info.get("multiplicity")
                    out_info["inp_run_type"] = inp_info.get("run_type")
                    out_info["inp_filepath"] = str(matched_inp)

                    # ---- 硬校验：inp 里解析出的原子数必须和 output 里
                    # "Total number of - Atoms:" 报告的真实原子数一致 ----
                    n_inp = inp_info.get("n_atoms_inp")
                    n_out = out_info.get("n_atoms")
                    if n_out is not None and n_inp is not None and n_inp != n_out:
                        print(f"  [WARN] 原子数不一致: inp解析={n_inp}, output真实={n_out}，"
                              f"structure 标记为不可信")
                        out_info["has_structure"] = False
                        out_info["structure_atom_count_mismatch"] = {
                            "n_atoms_from_inp_parse": n_inp,
                            "n_atoms_from_output": n_out,
                        }
                    else:
                        out_info["has_structure"] = True
                except Exception as e:
                    print(f"  [WARN] 解析 inp 失败: {e}")
                    out_info["has_structure"] = False
                    out_info["structure_error"] = str(e)
            else:
                print(f"  → 未找到对应 inp 文件（basename: {get_base_name(f.name)}）")
                out_info["has_structure"] = False

            results.append(out_info)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            results.append({
                "filepath": str(f),
                "filename": f.name,
                "success": False,
                "error": str(e)
            })

    # 按 mp_id 分组
    grouped = defaultdict(lambda: {"pristine": None, "defects": []})
    for r in results:
        if not r.get("success"):
            continue
        mp = r.get("mp_id") or "unknown"
        if r["defect_type"] == "pristine":
            grouped[mp]["pristine"] = r
        else:
            grouped[mp]["defects"].append(r)

    # 计算相对能量
    for mp, group in grouped.items():
        prist = group["pristine"]
        if prist is None:
            continue
        E_prist = prist["energy_au"]
        for d in group["defects"]:
            d["energy_diff_au"] = d["energy_au"] - E_prist
            d["energy_diff_ev"] = d["energy_diff_au"] * HARTREE_TO_EV
            if d.get("n_atoms") == prist.get("n_atoms") and d.get("n_atoms"):
                d["energy_diff_per_atom_ev"] = d["energy_diff_ev"] / d["n_atoms"]

    # 写 CSV
    fieldnames = [
        "mp_id", "defect_type", "charge_from_name", "filename",
        "energy_au", "energy_ev", "energy_diff_au", "energy_diff_ev", "energy_diff_per_atom_ev",
        "n_atoms", "n_kinds", "n_electrons_spin1", "n_electrons_spin2",
        "scf_steps", "scf_converged", "final_convergence",
        "run_time_s", "mpi_processes", "omp_threads", "warnings",
        "has_structure", "has_charges", "inp_charge", "inp_multiplicity",
        "project_name", "success"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 写 JSON
    json_data = {
        "meta": {
            "n_files": len(results),
            "n_success": sum(1 for r in results if r.get("success")),
            "n_with_structure": sum(1 for r in results if r.get("has_structure")),
            "n_with_charges": sum(1 for r in results if r.get("has_charges")),
            "hartree_to_ev": HARTREE_TO_EV,
            "output_dir": str(output_path),
            "inp_dir": str(inp_path),
        },
        "systems": {}
    }
    for mp, group in grouped.items():
        json_data["systems"][mp] = {
            "pristine": group["pristine"],
            "defects": group["defects"]
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] CSV  → {output_csv}")
    print(f"[Done] JSON → {output_json}")
    print(f"成功解析 output   : {json_data['meta']['n_success']} / {json_data['meta']['n_files']}")
    print(f"成功解析 structure: {json_data['meta']['n_with_structure']} / {json_data['meta']['n_files']}")
    print(f"成功解析 charges  : {json_data['meta']['n_with_charges']} / {json_data['meta']['n_files']}")

    return results, dict(grouped)


# ============================================================
# 6. 把 Mulliken / Hirshfeld 合并进已有 JSON
# ============================================================
def merge_mull_charges(
    mull_output_dir: str,
    existing_json: str = "cp2k_gnn_data.json",
    out_json: str = "cp2k_gnn_data.json",
    recursive: bool = True
) -> Dict:
    """
    读取 mull 目录下所有 output，提取电荷，并 merge 进已有 JSON。
    匹配键：mp_id + defect_type + charge_from_name
    只更新 mulliken / hirshfeld / has_charges 字段，绝不覆盖能量和结构。
    """
    print(f"Loading existing JSON: {existing_json}")
    with open(existing_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    systems = data.get("systems", {})
    output_path = Path(mull_output_dir)
    pattern = "**/*" if recursive else "*"
    files = list(output_path.glob(f"{pattern}.output")) + list(output_path.glob(f"{pattern}.out"))

    print(f"Found {len(files)} mull output files")

    n_matched = 0
    n_updated = 0
    n_not_found = 0

    for f in sorted(files):
        try:
            out_info = parse_cp2k_output(str(f))
            if not out_info.get("has_charges"):
                print(f"  [SKIP] {f.name}: 没有电荷分析")
                continue

            mp = out_info.get("mp_id")
            dtype = out_info.get("defect_type")
            charge = out_info.get("charge_from_name")

            if mp not in systems:
                print(f"  [MISS] {f.name}: mp_id={mp} 在 JSON 中不存在")
                n_not_found += 1
                continue

            target = None
            if dtype == "pristine":
                target = systems[mp].get("pristine")
            else:
                for d in systems[mp].get("defects", []):
                    if d.get("defect_type") == dtype and str(d.get("charge_from_name")) == str(charge):
                        target = d
                        break
                # 宽松一点：只比 defect_type
                if target is None:
                    for d in systems[mp].get("defects", []):
                        if d.get("defect_type") == dtype:
                            target = d
                            break

            if target is None:
                print(f"  [MISS] {f.name}: 找不到匹配记录 (mp={mp}, type={dtype}, q={charge})")
                n_not_found += 1
                continue

            # 只写入电荷相关字段
            if out_info.get("mulliken"):
                target["mulliken"] = out_info["mulliken"]
            if out_info.get("hirshfeld"):
                target["hirshfeld"] = out_info["hirshfeld"]
            target["has_charges"] = True
            # 可选：记录来源
            target["mull_source"] = out_info["filename"]

            n_matched += 1
            n_updated += 1
            print(f"  [OK]   {f.name} → {mp} / {dtype} / q={charge}")

        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    # 更新 meta
    if "meta" not in data:
        data["meta"] = {}
    data["meta"]["n_with_charges"] = sum(
        1 for sys in systems.values()
        for rec in ([sys.get("pristine")] if sys.get("pristine") else []) + sys.get("defects", [])
        if rec and rec.get("has_charges")
    )
    data["meta"]["last_mull_merge"] = datetime.now().isoformat(timespec="seconds")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n========== Merge 完成 ==========")
    print(f"匹配成功 : {n_matched}")
    print(f"更新记录 : {n_updated}")
    print(f"未找到   : {n_not_found}")
    print(f"输出文件 : {out_json}")
    print(f"当前带电荷的记录数: {data['meta'].get('n_with_charges', 0)}")

    return data


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CP2K output parser + Mulliken/Hirshfeld merger")
    parser.add_argument("--mode", choices=["normal", "mull"], default="mull",
                        help="normal: 普通批量解析; mull: 只提取电荷并 merge 进已有 JSON")
    parser.add_argument("--output_dir", default="./mull",
                        help="output 文件所在目录（mull 模式默认 ./mull）")
    parser.add_argument("--inp_dir", default="./cp2k_inputs",
                        help="inp 文件所在目录（normal 模式需要）")
    parser.add_argument("--existing_json", default="cp2k_gnn_data.json",
                        help="已有的 JSON 文件（mull 模式用）")
    parser.add_argument("--out_json", default="cp2k_gnn_data.json",
                        help="输出 JSON 路径")
    parser.add_argument("--out_csv", default="cp2k_gnn_summary.csv",
                        help="输出 CSV 路径（normal 模式）")
    args = parser.parse_args()

    if args.mode == "mull":
        # 专门处理 mull 目录，只提取电荷并 merge
        merge_mull_charges(
            mull_output_dir=args.output_dir,
            existing_json=args.existing_json,
            out_json=args.out_json,
            recursive=True
        )
    else:
        # 普通模式（兼容原来用法）
        process_directory(
            output_dir=args.output_dir,
            inp_dir=args.inp_dir,
            output_csv=args.out_csv,
            output_json=args.out_json,
            recursive=True
        )
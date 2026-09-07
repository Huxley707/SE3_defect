#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据完整性审计：核对 cp2k_gnn_data.json 里每条记录的 structure
是否真的对应它自己报告的 n_electrons_spin1/2。

原理：
  理论价电子数 = sum(每种元素价电子数 * 该元素原子数) - inp_charge
  价电子数从 structure["kinds"][element]["potential"] 里的 "...-qN" 解析（N 即价电子数）
  实际电子数 = n_electrons_spin1 + n_electrons_spin2

如果两者差距超过 1（留一点取整误差空间），说明这条记录的
structure 字段很可能不是真正用于该次 DFT 计算的完整结构
（例如只导出了对称不等价的一半原子），不应直接用于训练。

用法：
  python audit_structure_integrity.py cp2k_gnn_data.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def valence_from_potential(pot_str: str):
    m = re.search(r"q(\d+)", pot_str)
    return int(m.group(1)) if m else None


def expected_electrons(structure: dict, charge: int):
    kinds = structure.get("kinds", {})
    comp = Counter(a["element"] for a in structure["atoms"])
    total = 0
    for el, cnt in comp.items():
        pot = kinds.get(el, {}).get("potential")
        if not pot:
            return None
        v = valence_from_potential(pot)
        if v is None:
            return None
        total += v * cnt
    return total - charge


def audit(data: dict):
    rows = []
    for mp_id, system in data["systems"].items():
        entries = []
        if system.get("pristine") and system["pristine"].get("has_structure"):
            entries.append(("pristine", system["pristine"]))
        for d in system.get("defects", []):
            if d.get("has_structure"):
                entries.append((d.get("defect_type"), d))

        for label, e in entries:
            structure = e["structure"]
            charge = e.get("inp_charge", 0) or 0
            exp = expected_electrons(structure, charge)
            s1, s2 = e.get("n_electrons_spin1"), e.get("n_electrons_spin2")
            actual = (s1 + s2) if (s1 is not None and s2 is not None) else None

            status = "unknown"
            ratio = None
            if exp is not None and actual is not None:
                diff = abs(actual - exp)
                ratio = actual / exp if exp else None
                if diff <= 1:
                    status = "ok"
                elif ratio and abs(ratio - 2.0) < 0.05:
                    status = "likely_half_structure"  # 实际电子数≈理论值的2倍
                else:
                    status = "mismatch_other"

            rows.append({
                "mp_id": mp_id,
                "defect_type": label,
                "charge_from_name": e.get("charge_from_name"),
                "project_name": e.get("project_name"),
                "n_atoms_recorded": len(structure["atoms"]),
                "expected_electrons": exp,
                "actual_electrons": actual,
                "ratio_actual_over_expected": round(ratio, 3) if ratio else None,
                "status": status,
            })
    return rows


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cp2k_gnn_data.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = audit(data)

    n_total = len(rows)
    status_counter = Counter(r["status"] for r in rows)
    print(f"总记录数: {n_total}")
    for status, cnt in status_counter.most_common():
        print(f"  {status:24s}: {cnt:4d}  ({cnt/n_total*100:.1f}%)")

    out_path = path.with_name("structure_integrity_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n详细报告已保存: {out_path}")

    ok_ids = [r["project_name"] for r in rows if r["status"] == "ok"]
    print(f"\n可信（结构与电子数吻合）的记录数: {len(ok_ids)} / {n_total}")
    if ok_ids:
        print("示例:", ok_ids[:10])


if __name__ == "__main__":
    main()
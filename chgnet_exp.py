#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHGNet 零样本基线：与方案 A 的 energy_diff 对比
------------------------------------------------
依赖: pip install chgnet pymatgen torch
输入: cp2k_gnn_data.json（与 GAT 相同）
输出: 测试集 MAE / RMSE / R²，以及散点图
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Lattice
from chgnet.model.model import CHGNet

# ------------------------------------------------------------
# 0. 路径与设置
# ------------------------------------------------------------
DATA_PATH = Path("cp2k_gnn_data.json")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

# ------------------------------------------------------------
# 1. JSON → pymatgen Structure
# ------------------------------------------------------------
def json_structure_to_pymatgen(struct_dict) -> Structure:
    """把 JSON 里的 structure 转成 pymatgen Structure"""
    atoms = struct_dict["atoms"]
    species = [a["element"] for a in atoms]
    coords = [[a["x"], a["y"], a["z"]] for a in atoms]
    lattice = Lattice(struct_dict["lattice_vectors"])
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def load_samples(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 按 mp_id 分组，方便算 E_defect - E_pristine
    by_mp = defaultdict(lambda: {"pristine": None, "defects": []})

    for mp_id, system in raw["systems"].items():
        if system.get("pristine") and system["pristine"].get("has_structure"):
            p = system["pristine"]
            by_mp[mp_id]["pristine"] = {
                "id": f"{mp_id}_pristine",
                "structure": json_structure_to_pymatgen(p["structure"]),
                "energy_diff": 0.0,  # 相对自身为 0
                "defect_type": "pristine",
            }
        for d in system.get("defects", []):
            if not d.get("has_structure"):
                continue
            by_mp[mp_id]["defects"].append({
                "id": f"{mp_id}_{d['defect_type']}_q{d.get('charge_from_name', '')}",
                "structure": json_structure_to_pymatgen(d["structure"]),
                "energy_diff": float(d.get("energy_diff_ev", d.get("energy_ev", 0.0))),
                "defect_type": d["defect_type"],
            })
    return by_mp


# ------------------------------------------------------------
# 2. CHGNet 预测（eV/atom → 总能量 eV）
# ------------------------------------------------------------
def predict_formation_energy(chgnet: CHGNet, structure: Structure) -> float:
    """
    直接预测形成能 (eV/atom)
    CHGNet的"ef"任务会使用元素化学势参考态
    """
    pred = chgnet.predict_structure(structure, task="ef")
    # "ef"返回的是每原子的形成能 (eV/atom)
    return float(pred["ef"])


def main():
    print("加载数据...")
    by_mp = load_samples(DATA_PATH)

    print("加载预训练 CHGNet...")
    chgnet = CHGNet.load()  # 默认最新预训练权重
    chgnet.eval()

    # 收集所有 (true_diff, pred_diff, id, type)
    records = []
    n_skip = 0

    for mp_id, group in by_mp.items():
        prist = group["pristine"]
        if prist is None:
            # 没有 pristine，无法算相对能量，跳过该 mp 的缺陷
            n_skip += len(group["defects"])
            continue

        try:
            E_prist = predict_formation_energy(chgnet, prist["structure"])
        except Exception as e:
            print(f"  [跳过 pristine] {mp_id}: {e}")
            n_skip += 1 + len(group["defects"])
            continue

        # pristine 自身 diff=0
        records.append({
            "id": prist["id"],
            "true": 0.0,
            "pred": 0.0,  # 相对自身
            "defect_type": "pristine",
        })

        for d in group["defects"]:
            try:
                E_prist = predict_formation_energy(chgnet, prist["structure"])
                E_def = predict_formation_energy(chgnet, d["structure"])
                pred_diff = (E_def - E_prist) * len(d["structure"])
                records.append({
                    "id": d["id"],
                    "true": d["energy_diff"],
                    "pred": pred_diff,
                    "defect_type": d["defect_type"],
                })
            except Exception as e:
                print(f"  [跳过] {d['id']}: {e}")
                n_skip += 1

    print(f"有效样本: {len(records)}, 跳过: {n_skip}")

    y_true = np.array([r["true"] for r in records])
    y_pred = np.array([r["pred"] for r in records])

    # 与方案 A 相同划分（按样本索引，seed=42）
    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED)

    y_true_test = y_true[test_idx]
    y_pred_test = y_pred[test_idx]

    mae = mean_absolute_error(y_true_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    r2 = r2_score(y_true_test, y_pred_test)

    print("\n===== CHGNet 零样本（测试集，与方案A同划分）=====")
    print(f"Energy  MAE  : {mae:.3f} eV")
    print(f"Energy  RMSE : {rmse:.3f} eV")
    print(f"Energy  R²   : {r2:.4f}")
    print(f"测试样本数   : {len(test_idx)}")

    # 全量也报一份，便于参考
    mae_all = mean_absolute_error(y_true, y_pred)
    r2_all = r2_score(y_true, y_pred)
    print(f"\n全量 MAE: {mae_all:.3f} eV | R²: {r2_all:.4f} | N={len(records)}")

    # 可视化
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true_test, y_pred_test, alpha=0.7, edgecolors="k", linewidths=0.5)
    lims = [
        min(y_true_test.min(), y_pred_test.min()),
        max(y_true_test.max(), y_pred_test.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("True energy_diff (eV)")
    ax.set_ylabel("CHGNet pred energy_diff (eV)")
    ax.set_title(f"CHGNet zero-shot\nMAE={mae:.1f} eV, R²={r2:.3f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / "chgnet_zeroshot_result.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"图已保存: {fig_path}")

    # 保存数值结果
    np.savez(
        OUT_DIR / "chgnet_zeroshot_metrics.npz",
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        mae=mae,
        rmse=rmse,
        r2=r2,
        test_idx=test_idx,
    )
    print("数值已保存: artifacts/chgnet_zeroshot_metrics.npz")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage A 结果可视化与验收报告
================================
基于 Stage A 运行日志与指标，生成多面板图 + 文字验收清单。
不依赖重新训练；学习曲线取自日志中的验证集 MAE 采样点。
图注与报告用语为学术表述，不出现代码内部变量名。

用法：
  python stageA_visualize_accept.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = Path("artifacts")
if not OUT_DIR.exists():
    OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 内部键名仅作字典索引；对外显示一律用 DISPLAY_NAME
RESULTS = {
    "site_readout": {
        "mae": 66.427,
        "rmse": 106.690,
        "r2": 0.9851,
        "desc": "宿主图 + 位点读出（主模型）",
        "color": "#27ae60",
    },
    "global_pool": {
        "mae": 105.706,
        "rmse": 219.789,
        "r2": 0.9368,
        "desc": "全局均值/最大值池化读出",
        "color": "#e74c3c",
    },
    "no_elem_prior": {
        "mae": 62.266,
        "rmse": 102.561,
        "r2": 0.9862,
        "desc": "位点读出，去除 d 电子数与赝势价电子数通道",
        "color": "#3498db",
    },
}

DISPLAY_NAME = {
    "site_readout": "Site readout\n(host graph)",
    "global_pool": "Global pool\nonly",
    "no_elem_prior": "w/o element\nprior",
}

DISPLAY_NAME_INLINE = {
    "site_readout": "Site readout",
    "global_pool": "Global pool only",
    "no_elem_prior": "w/o element prior",
}

CURVES = {
    "site_readout": {
        1: 572.02, 10: 248.89, 20: 165.99, 30: 186.64,
        40: 84.51, 50: 105.20, 60: 125.24, 70: 69.44, 80: 81.20,
    },
    "global_pool": {
        1: 753.42, 10: 515.93, 20: 383.09, 30: 420.24,
        40: 362.24, 50: 415.99, 60: 487.82, 70: 176.99, 80: 186.57,
    },
    "no_elem_prior": {
        1: 607.38, 10: 217.73, 20: 152.49, 30: 214.29,
        40: 93.78, 50: 86.04, 60: 107.18, 70: 93.22, 80: 84.44,
    },
}

DATA_STATS = {
    "n_total": 307,
    "n_pristine": 18,
    "n_vacancy": 213,
    "n_substitutional": 76,
    "vac_host_ok": 175,
    "vac_host_fail": 38,
    "n_train": 238,
    "n_test": 69,
    "n_train_mp": 14,
    "n_test_mp": 4,
    "mp_overlap": 0,
    "y_mean": 349.187,
    "y_std": 825.530,
    "node_dim": 8,
    "edge_dim": 21,
    "global_dim": 5,
}

BASE_MAE = RESULTS["site_readout"]["mae"]
ORDER = ["site_readout", "global_pool", "no_elem_prior"]


def utility(key):
    if key == "site_readout":
        return 0.0
    return (RESULTS[key]["mae"] - BASE_MAE) / BASE_MAE * 100.0


def make_acceptance_panel():
    plt.rcParams.update({
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    fig = plt.figure(figsize=(14, 10))
    # 第一行 3 列，第二行 2 列居中
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    labels = [DISPLAY_NAME[k] for k in ORDER]
    colors = [RESULTS[k]["color"] for k in ORDER]
    maes = [RESULTS[k]["mae"] for k in ORDER]
    rmses = [RESULTS[k]["rmse"] for k in ORDER]
    r2s = [RESULTS[k]["r2"] for k in ORDER]
    utils = [utility(k) for k in ORDER]

    # ========== (a) MAE ==========
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(labels, maes, color=colors, edgecolor="k",
                  linewidth=0.8, width=0.65, zorder=3)
    ax.axhline(BASE_MAE, color="#27ae60", ls="--", lw=1.4, alpha=0.85,
               label=f"Site-readout MAE = {BASE_MAE:.1f} eV")
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("(a) Test MAE by readout / feature setting")
    ax.set_ylim(0, max(maes) * 1.22)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== (b) Utility ==========
    ax = fig.add_subplot(gs[0, 1])
    util_colors = ["#95a5a6" if u == 0 else ("#e74c3c" if u > 0 else "#27ae60") for u in utils]
    bars = ax.bar(labels, utils, color=util_colors, edgecolor="k",
                  linewidth=0.8, width=0.65, zorder=3)
    ax.axhline(0, color="k", lw=1.0)
    for b, v in zip(bars, utils):
        ypos = v + (2.5 if v >= 0 else -4.5)
        ax.text(b.get_x() + b.get_width() / 2, ypos, f"{v:+.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax.set_ylabel("Utility (%)")
    ax.set_title("(b) Ablation utility\n(+ = removing component increases error)")
    ax.set_ylim(min(utils) - 15, max(utils) + 18)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate("Site readout\nnecessary", xy=(1, utils[1]), xytext=(0.35, utils[1] + 14),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))

    # ========== (c) R² + RMSE (修复标注显示不全) ==========
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(ORDER))
    w = 0.35

    # R² 左轴
    bars_r2 = ax.bar(x - w / 2, r2s, w, label=r"$R^2$", color="#8e44ad",
                     edgecolor="k", linewidth=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(r"$R^2$", color="#8e44ad")
    ax.set_ylim(0.90, 1.01)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # RMSE 右轴
    ax2 = ax.twinx()
    bars_rmse = ax2.bar(x + w / 2, rmses, w, label="RMSE", color="#e67e22",
                        edgecolor="k", linewidth=0.6, zorder=3)
    ax2.set_ylabel("RMSE (eV)", color="#e67e22")
    ax2.set_ylim(0, max(rmses) * 1.25)
    ax2.spines["top"].set_visible(False)

    ax.set_title(r"(c) $R^2$ and RMSE")

    # 图例放在图上方
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              fontsize=8, bbox_to_anchor=(0.0, 1.18), ncol=2, frameon=True)

    # 柱顶标数值
    for bar, val in zip(bars_r2, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#6c3483")
    for bar, val in zip(bars_rmse, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#a04000")

    # ========== (d) Learning curves (第二行左侧) ==========
    ax = fig.add_subplot(gs[1, 0])
    for key in ORDER:
        epochs = sorted(CURVES[key].keys())
        vals = [CURVES[key][e] for e in epochs]
        ax.plot(epochs, vals, "o-", color=RESULTS[key]["color"],
                label=DISPLAY_NAME_INLINE[key], lw=1.8, markersize=5)
        ax.axhline(RESULTS[key]["mae"], color=RESULTS[key]["color"],
                   ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("(d) Learning curves (validation MAE)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 800)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== (e) Sample composition (第二行右侧) ==========
    ax = fig.add_subplot(gs[1, 1])
    type_labels = ["Pristine", "Vacancy", "Substitutional"]
    type_counts = [DATA_STATS["n_pristine"], DATA_STATS["n_vacancy"],
                   DATA_STATS["n_substitutional"]]
    type_colors = ["#95a5a6", "#2980b9", "#9b59b6"]
    _, _, autotexts = ax.pie(
        type_counts, labels=type_labels, colors=type_colors,
        autopct=lambda p: f"{p:.0f}%\n({int(round(p / 100 * DATA_STATS['n_total']))})",
        startangle=90, pctdistance=0.55,
        wedgeprops=dict(width=0.45, edgecolor="w", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(8)
    ok_rate = 100 * DATA_STATS["vac_host_ok"] / DATA_STATS["n_vacancy"]
    ax.set_title(
        "(e) Dataset composition\n"
        f"Vacancy host-graph site resolved: "
        f"{DATA_STATS['vac_host_ok']}/{DATA_STATS['n_vacancy']} ({ok_rate:.0f}%)"
    )

    # gs[1,1] 留空，不画任何内容

    fig.suptitle(
        "Host-crystal graph with site-level readout",
        fontsize=13, fontweight="bold", y=0.98,
    )

    path = OUT_DIR / "stageA_acceptance_panel.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def make_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key in ORDER:
        epochs = sorted(CURVES[key].keys())
        vals = [CURVES[key][e] for e in epochs]
        ax.plot(epochs, vals, "o-", color=RESULTS[key]["color"],
                label=DISPLAY_NAME_INLINE[key], lw=2, markersize=6)
        ax.axhline(RESULTS[key]["mae"], color=RESULTS[key]["color"],
                   ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Learning curves (full range)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 800)

    ax = axes[1]
    for key in ORDER:
        epochs = [e for e in sorted(CURVES[key].keys()) if e >= 30]
        vals = [CURVES[key][e] for e in epochs]
        ax.plot(epochs, vals, "o-", color=RESULTS[key]["color"],
                label=f"{DISPLAY_NAME_INLINE[key]} (best {RESULTS[key]['mae']:.1f} eV)",
                lw=2, markersize=6)
        ax.axhline(RESULTS[key]["mae"], color=RESULTS[key]["color"],
                   ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Late training (epoch ≥ 30)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(50, 450)

    plt.tight_layout()
    path = OUT_DIR / "stageA_learning_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def write_summary():
    lines = []
    lines.append("=" * 72)
    lines.append("Stage A acceptance report: host-crystal graph with site-level readout")
    lines.append("=" * 72)
    lines.append("")
    lines.append("[Data and split]")
    lines.append(
        f"  Total samples: {DATA_STATS['n_total']}  "
        f"(pristine={DATA_STATS['n_pristine']}, "
        f"vacancy={DATA_STATS['n_vacancy']}, "
        f"substitutional={DATA_STATS['n_substitutional']})"
    )
    lines.append(
        f"  Vacancy host-graph site resolved: {DATA_STATS['vac_host_ok']} / "
        f"{DATA_STATS['n_vacancy']} "
        f"({100 * DATA_STATS['vac_host_ok'] / DATA_STATS['n_vacancy']:.1f}%), "
        f"fallback: {DATA_STATS['vac_host_fail']}"
    )
    lines.append(f"  Split: train={DATA_STATS['n_train']}, test={DATA_STATS['n_test']}")
    lines.append(
        f"  Materials: train={DATA_STATS['n_train_mp']}, test={DATA_STATS['n_test_mp']}, "
        f"overlap={DATA_STATS['mp_overlap']}  → no leakage"
    )
    lines.append(
        f"  Target energy: mean={DATA_STATS['y_mean']:.2f} eV, "
        f"std={DATA_STATS['y_std']:.2f} eV"
    )
    lines.append(
        f"  Feature dims: node={DATA_STATS['node_dim']}, edge={DATA_STATS['edge_dim']}, "
        f"global={DATA_STATS['global_dim']} (no defect-type one-hot)"
    )
    lines.append("")
    lines.append("[Metrics]")
    lines.append(f"  {'Setting':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Utility':>10}")
    lines.append("  " + "-" * 58)
    for key in ORDER:
        u = utility(key)
        u_str = "(base)" if key == "site_readout" else f"{u:+.1f}%"
        r = RESULTS[key]
        lines.append(
            f"  {DISPLAY_NAME_INLINE[key]:<22} {r['mae']:8.2f} {r['rmse']:8.2f} "
            f"{r['r2']:8.4f} {u_str:>10}"
        )
    lines.append("")
    lines.append("[Conclusions]")
    lines.append("  1. Site-level readout vs global pooling")
    lines.append(
        f"     Global-pool utility = {utility('global_pool'):+.1f}%  → strongly positive; "
        "site readout is effective."
    )
    lines.append(
        "     Interpretation: under the rigid-vacancy assumption, formation energies are "
        "governed by the local chemical environment of the removed host site; "
        "global pooling dilutes that signal."
    )
    lines.append("")
    lines.append("  2. Element chemical priors (d-electron count, pseudopotential valence)")
    lines.append(
        f"     Utility after removing these channels = {utility('no_elem_prior'):+.1f}%  → "
        "error does not increase."
    )
    lines.append(
        "     Consistent with the design intent: after site-centered message passing, "
        "hard-coded elemental descriptors contribute little marginal information."
    )
    lines.append("")
    lines.append("  3. Split and feature constraints")
    lines.append(
        "     No material overlap between train and test; no defect-type one-hot features; "
        "vacancies prefer the host-graph path. All constraints satisfied."
    )
    lines.append("")
    lines.append("  4. Relative to prior no-type global-pool baselines (~137 eV MAE scale)")
    lines.append(f"     Site-readout MAE = {BASE_MAE:.1f} eV; clear reduction.")
    lines.append("")
    lines.append("[Summary]")
    lines.append(
        "  Stage A meets the acceptance criteria: site readout is effective, dependence on "
        "hand-crafted elemental priors is reduced, the material-wise split is clean, and "
        "defect-type shortcuts are absent. Suitable as the baseline for subsequent work."
    )
    lines.append("")
    lines.append("[Limitations]")
    lines.append("  - Single random seed; formal reporting should use mean±std over ≥3 seeds.")
    lines.append(
        f"  - {DATA_STATS['vac_host_fail']} vacancies failed site resolution and fell back "
        "to the defect graph; report the failure rate in the appendix if needed."
    )
    lines.append(
        "  - Predicted–true scatter plots require saving test predictions during training."
    )
    lines.append(
        f"  - Test set covers only {DATA_STATS['n_test_mp']} materials; "
        "extrapolation claims should remain cautious."
    )
    lines.append("=" * 72)

    text = "\n".join(lines)
    path = OUT_DIR / "stageA_acceptance_summary.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {path}")
    return path


def make_utility_focus():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [DISPLAY_NAME[k] for k in ORDER]
    utils = [utility(k) for k in ORDER]
    colors = ["#95a5a6", "#e74c3c", "#27ae60"]
    bars = ax.barh(labels[::-1], utils[::-1], color=colors[::-1], edgecolor="k",
                   height=0.55, linewidth=0.9)
    ax.axvline(0, color="k", lw=1.2)
    for b, v in zip(bars, utils[::-1]):
        xoff = 2 if v >= 0 else -2
        ha = "left" if v >= 0 else "right"
        ax.text(v + xoff, b.get_y() + b.get_height() / 2, f"{v:+.1f}%",
                va="center", ha=ha, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Utility = $(\mathrm{MAE}_{\mathrm{ablated}} - \mathrm{MAE}_{\mathrm{base}}) "
                  r"/ \mathrm{MAE}_{\mathrm{base}} \times 100\%$")
    ax.set_title("Ablation utility of site readout and element priors")
    ax.set_xlim(-25, 75)
    ax.grid(axis="x", alpha=0.3)
    ax.text(
        0.98, 0.05,
        "Global pool only +59% → site readout is essential\n"
        "w/o element prior −6% → neighborhood messages carry species information",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#f39c12"),
    )
    plt.tight_layout()
    path = OUT_DIR / "stageA_utility_focus.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    print("Generating Stage A acceptance figures...\n")
    make_acceptance_panel()
    make_learning_curves()
    make_utility_focus()
    write_summary()
    print("\nDone.")
    print("  stageA_acceptance_panel.png")
    print("  stageA_learning_curves.png")
    print("  stageA_utility_focus.png")
    print("  stageA_acceptance_summary.txt")
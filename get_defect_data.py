"""
Defect-structure + local-environment feature dataset builder
for photocatalytic oxide screening (TiO2, WO3, BiVO4, SrTiO3, ...).

This replaces the earlier "index-only" pipeline. Key differences:

1. Defects are REAL pymatgen-analysis-defects Defect objects (Vacancy /
   Substitution), generated on the SYMMETRIZED PRIMITIVE cell, not a
   fixed 2x2x2 supercell of the conventional cell. Symmetry-distinct
   sites are found once, cheaply, on the small cell.
2. Supercells are built per-defect via `Defect.get_supercell_structure()`,
   which searches for a (near-)cubic cell under an atom-count cap, instead
   of a single hardcoded scaling matrix for every material.
3. Charge states are no longer hardcoded to 0. They are estimated from a
   bond-valence (BVAnalyzer) oxidation-state guess of the host, which is a
   defensible but still approximate heuristic -- see `charge_states_for_*`.
   VERIFY these against known formal charges for your materials (e.g.
   Ti4+/O2- in TiO2, W6+/O2- in WO3, Bi3+/V5+/O2- in BiVO4) before treating
   them as ground truth; oxidation-state guessing can fail silently on
   materials with mixed/unusual valence.
4. Local-environment features are computed with the correct CrystalNN API
   (integer site index into a structure, not fractional coordinates), on
   the PRISTINE cell at the defect's site index -- i.e. "what used to
   surround this site" -- which is well-defined for vacancies and
   substitutions alike. For precision, CrystalNN is run on the
   OXIDATION-STATE-DECORATED structure with `cation_anion=True`, which
   uses ionic radii and restricts search to cation-anion contacts --
   noticeably more accurate for these ionic oxides than the
   undecorated/covalent-radius fallback (which is what triggers pymatgen's
   "No oxidation states specified" warnings). This is the DFT-free feature
   set that can seed a few-shot / physics-prior defect-level model.
5. Storage is de-duplicated: each unique defect (structure + site) is
   stored once; charge states are stored as lightweight references to it,
   instead of copying the full supercell JSON for every charge state.
6. Materials can come from two sources, merged into one run: (a) fetched
   from Materials Project via `TARGET_MATERIALS` (mp-id lookup), and/or
   (b) a local folder of CIF files via `CIF_DIR`, for e.g. a scraped
   set of experimental/computed structures. This is the main lever for
   growing dataset size beyond a handful of MP entries.

IMPORTANT VERSION CAVEAT:
pymatgen-analysis-defects' public API (class names, `generate()` vs
`get_defects()`, keyword names for SubstitutionGenerator) has changed
across releases. The calls below match the current documented API
(materialsproject.github.io/pymatgen-analysis-defects) as of writing.
If you hit an AttributeError/TypeError on `VacancyGenerator`/
`SubstitutionGenerator`, run `help(VacancyGenerator)` /
`help(SubstitutionGenerator)` in your environment and adjust the two
generator calls accordingly -- the rest of the pipeline does not depend
on which variant is used.

This script does NOT compute formation energies or defect levels -- it
does not attempt to sidestep the need for DFT. It produces (a) physically
constructed defect structures at multiple charge states and (b) DFT-free
structural/coordination descriptors, which is the sensible role for a
low-cost pre-processing step feeding a few-shot model that will still need
some DFT- (or literature-) labeled formation energies / transition levels
to actually train on.
"""

import glob
import json
import os
import uuid
from typing import Optional

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.defects.core import Substitution, Vacancy
from pymatgen.analysis.defects.generators import (
    SubstitutionGenerator,
    VacancyGenerator,
)
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Species
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ====================== configuration ======================

MP_API_KEY = "J1uYn8CaW0o9oUo7S60H2mjJDVPDs0Qm"  # https://materialsproject.org
mpr = MPRester(MP_API_KEY)

TARGET_MATERIALS = {
    "TiO2_anatase": "mp-390",
    "TiO2_rutile": "mp-2657",
    "WO3": "mp-19342",
    "BiVO4": "mp-23506",   # monoclinic scheelite (C2/c) -- the photocatalytically active phase.
                           # mp-613338 does not resolve via MPRester (empty response -> IndexError).
                           # Other BiVO4 polymorphs in MP if you need them instead:
                           #   mp-23044 (tetragonal I4_1/a), mp-540864 (orthorhombic Pbcn, "pucherite").
    "SrTiO3": "mp-4651",
}

# Cation site -> candidate substitutional dopants. Extend/edit per material;
# the previous version only ever triggered for 'Ti', silently producing zero
# substitutional defects for WO3 / BiVO4 / SrTiO3.
CATION_DOPANTS = {
    "Ti": ["V", "Cr", "Fe", "Nb", "Ta"],
    "W": ["Mo", "Nb", "Ta"],
    "Bi": ["La", "Y"],
    "V": ["Nb", "Ta", "Mo"],  # the V-site in BiVO4, not the V dopant above
    "Sr": ["Ca", "Ba", "La"],
}

# Point this at a folder of .cif files to pull in your own scraped
# structures alongside (or instead of) TARGET_MATERIALS. Set to None to
# disable. Every ordered CIF found here becomes an additional material.
CIF_DIR = "data/cif_files"  # e.g. r"E:\GAT-eht\my_cifs"

CIF_GLOB = "*.cif"

# Set False if you only want your local CIFs and not the MP-fetched set.
USE_TARGET_MATERIALS = True

MAX_SC_ATOMS = 300          # cap on generated supercell size
CHARGE_PADDING = 1          # extra charge state on each side of the formal bound
SYMPREC = 0.01

OUTPUT_DIR = "data"

# ====================== structure / symmetry helpers ======================


def _to_primitive(structure: Structure) -> Structure:
    """Reduce any ordered structure to its symmetrized primitive cell.
    Defects are generated on this cell; supercells are built later,
    per-defect, at whatever size that defect actually needs."""
    sga = SpacegroupAnalyzer(structure, symprec=SYMPREC)
    return sga.get_primitive_standard_structure()


def get_primitive_structure_from_mp(mp_id: str) -> Structure:
    """Fetch a structure from Materials Project by id and reduce it."""
    try:
        structure = mpr.get_structure_by_material_id(mp_id)
    except IndexError as exc:
        raise RuntimeError(
            f"MPRester returned no structure for '{mp_id}'. Either the ID does not "
            f"exist / was retired, or your MP_API_KEY doesn't have access to it. "
            f"Double check the ID at https://next-gen.materialsproject.org/materials/{mp_id}"
        ) from exc
    return _to_primitive(structure)


def load_materials_from_cifs(cif_dir: str, pattern: str = "*.cif"):
    """Yield (material_name, Structure, cif_path) for every readable,
    fully-ordered CIF in cif_dir.

    Scraped/experimental CIFs frequently carry partial or fractional site
    occupancies from Rietveld refinement (mixed-occupancy sites, split
    positions). VacancyGenerator / SubstitutionGenerator require an
    ORDERED structure -- feeding them a disordered site produces either a
    crash or a silently wrong defect. Rather than guess how to resolve the
    occupancy, such CIFs are skipped with a printed warning; if you need
    one of them, decide explicitly how to order it first (e.g.
    `structure.replace(i, dominant_species)` for the majority occupant per
    site), then re-run.
    """
    entries = []
    paths = sorted(glob.glob(os.path.join(cif_dir, pattern)))
    if not paths:
        print(f"  [warn] no files matched {pattern} in {cif_dir}")
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            structure = Structure.from_file(path)
        except Exception as exc:
            print(f"  [skip] {name}: failed to parse CIF ({exc})")
            continue
        if not structure.is_ordered:
            print(f"  [skip] {name}: has fractional/partial site occupancies; "
                  f"defect generators require an ordered structure.")
            continue
        entries.append((name, structure, path))
    print(f"  loaded {len(entries)} ordered structures from {cif_dir} "
          f"({len(paths) - len(entries)} skipped)")
    return entries


def find_site_index(structure: Structure, target_frac_coords, tol: float = 1e-3) -> int:
    """Match a defect's site back to its index in a reference structure by
    fractional coordinates (robust to the small numerical differences that
    can appear between a site produced by a generator and the original
    structure's site list)."""
    for i, site in enumerate(structure):
        if np.allclose(site.frac_coords, target_frac_coords, atol=tol):
            return i
    raise ValueError("Could not locate defect site in the reference structure "
                      "(tolerance may need loosening).")


def guess_oxidation_states(structure: Structure) -> Structure:
    """Best-effort oxidation-state decoration via bond-valence analysis,
    falling back to composition-based guessing. This is a heuristic --
    spot-check the assigned states against known chemistry for each
    material before trusting downstream charge-state ranges."""
    try:
        bva = BVAnalyzer()
        return bva.get_oxi_state_decorated_structure(structure)
    except Exception as exc:
        print(f"    [warn] BVAnalyzer failed ({exc}); falling back to "
              f"composition-based oxidation-state guess.")
        s = structure.copy()
        s.add_oxidation_state_by_guess()
        return s


# ====================== charge-state heuristics ======================


def charge_states_for_vacancy(oxi_structure: Structure, site_index: int) -> list:
    """A vacancy of a formally n-charged ion can accommodate charge states
    between 0 and -n (the fully ionic limit), e.g. removing O2- -> q in
    {0, +1, +2} range in the usual defect-charge convention (opposite sign
    to the removed ion), removing Ti4+ -> q in {0,-1,-2,-3,-4}. We pad by
    CHARGE_PADDING on either side to avoid missing shallow states."""
    oxi = oxi_structure[site_index].specie.oxi_state
    q_bound = -int(round(oxi))
    lo, hi = sorted([0, q_bound])
    return list(range(lo - CHARGE_PADDING, hi + CHARGE_PADDING + 1))


def charge_states_for_substitution(oxi_structure: Structure, site_index: int,
                                    dopant_symbol: str) -> list:
    """Charge states from the difference between the dopant's common
    oxidation states and the host ion's formal charge, e.g. Cr3+ on Ti4+
    is nominally a -1 (acceptor-like) defect; Nb5+ on Ti4+ is nominally +1
    (donor-like). Uses pymatgen's tabulated common oxidation states for the
    dopant element, which is coarse -- verify against literature for the
    specific host lattice."""
    host_oxi = oxi_structure[site_index].specie.oxi_state
    common = Species(dopant_symbol).element.common_oxidation_states
    if not common:
        common = [0]
    diffs = sorted({int(round(d - host_oxi)) for d in common})
    lo, hi = min(diffs), max(diffs)
    return list(range(lo - CHARGE_PADDING, hi + CHARGE_PADDING + 1))


# ====================== local environment (DFT-free features) ======================


def local_environment_features(oxi_structure: Structure, site_index: int) -> dict:
    """Coordination-shell descriptor for the defect site, computed on the
    PRISTINE, OXIDATION-STATE-DECORATED structure (i.e. what surrounded
    this site before the defect was introduced). Works uniformly for
    vacancies and substitutions since both reference an existing,
    occupied site.

    `cation_anion=True` tells CrystalNN to use ionic radii and only
    consider cation<->anion contacts, which is the physically correct
    bonding picture for these oxides and measurably more accurate than
    the covalent/atomic-radius fallback CrystalNN uses when no oxidation
    states are set (that fallback is what produces the "No oxidation
    states specified" warnings from the earlier version). If oxidation
    decoration failed upstream (all sites default to 0), cation_anion
    mode itself will fail (no anions to find) and we retry once without
    it rather than lose the whole record.
    """
    try:
        nn_finder = CrystalNN(weighted_cn=True, cation_anion=True)
        nn_info = nn_finder.get_nn_info(oxi_structure, site_index)
    except Exception:
        try:
            nn_finder = CrystalNN(weighted_cn=True)
            nn_info = nn_finder.get_nn_info(oxi_structure, site_index)
        except Exception as exc:
            return {
                "coordination_number": 0,
                "neighbors": [],
                "avg_bond_length": None,
                "element_diversity": 0,
                "error": str(exc),
            }
    structure = oxi_structure

    center = structure[site_index]
    neighbors = []
    for nn in nn_info:
        neighbors.append({
            "element": str(nn["site"].specie),
            "distance": round(center.distance(nn["site"]), 4),
            "weight": round(float(nn.get("weight", 1.0)), 4),
        })

    distances = [n["distance"] for n in neighbors]
    return {
        "coordination_number": len(neighbors),
        "neighbors": neighbors,
        "avg_bond_length": float(np.mean(distances)) if distances else None,
        "element_diversity": len({n["element"] for n in neighbors}),
    }


# ====================== main defect generation ======================


def generate_defect_dataset(material_name: str, source: dict):
    """Returns (primitive_structure, list_of_defect_records,
    list_of_charge_state_records) for one material.

    `source` is either {"kind": "mp", "mp_id": "mp-390"} or
    {"kind": "cif", "structure": Structure, "path": "/path/to/file.cif"}.
    """
    mp_id = source.get("mp_id")
    label = mp_id if source["kind"] == "mp" else source.get("path", "cif")
    print(f"\n=== {material_name} ({label}) ===")

    if source["kind"] == "mp":
        prim = get_primitive_structure_from_mp(mp_id)
    elif source["kind"] == "cif":
        prim = _to_primitive(source["structure"])
    else:
        raise ValueError(f"unknown source kind: {source['kind']!r}")

    oxi_struct = guess_oxidation_states(prim)
    print(f"  primitive cell: {len(prim)} atoms, formula {prim.formula}")

    defect_records = []
    charge_records = []

    # ---------- vacancies on every symmetry-distinct site ----------
    for vac in VacancyGenerator(symprec=SYMPREC).generate(prim):
        site_idx = find_site_index(prim, vac.site.frac_coords)
        local_env = local_environment_features(oxi_struct, site_idx)
        sc_struct = vac.get_supercell_structure(max_atoms=MAX_SC_ATOMS)

        defect_id = str(uuid.uuid4())
        defect_records.append({
            "defect_id": defect_id,
            "material": material_name,
            "mp_id": mp_id,
            "defect_type": "vacancy",
            "defect_name": vac.name,
            "defect_species": str(vac.site.specie),
            "multiplicity": vac.multiplicity,
            "defect_frac_coords": vac.site.frac_coords.tolist(),
            "defect_supercell": sc_struct.as_dict(),
            "local_environment": local_env,
        })

        for q in charge_states_for_vacancy(oxi_struct, site_idx):
            charge_records.append({
                "defect_id": defect_id,
                "material": material_name,
                "defect_type": "vacancy",
                "charge_state": q,
            })

    # ---------- substitutions, per configured cation dopant table ----------
    host_elements_present = {str(site.specie) for site in prim}
    for host_species, dopants in CATION_DOPANTS.items():
        if host_species not in host_elements_present:
            continue
        try:
            substitutions = SubstitutionGenerator(symprec=SYMPREC).generate(
                prim, {host_species: dopants}
            )
        except TypeError:
            # API variant fallback -- see version caveat in module docstring.
            print(f"    [warn] SubstitutionGenerator signature mismatch for "
                  f"{host_species}; skipping. Check installed API.")
            continue

        for sub in substitutions:
            site_idx = find_site_index(prim, sub.site.frac_coords)
            dopant_symbol = str(sub.site.specie.element) if hasattr(
                sub.site.specie, "element") else str(sub.site.specie)
            local_env = local_environment_features(oxi_struct, site_idx)
            sc_struct = sub.get_supercell_structure(max_atoms=MAX_SC_ATOMS)

            defect_id = str(uuid.uuid4())
            defect_records.append({
                "defect_id": defect_id,
                "material": material_name,
                "mp_id": mp_id,
                "defect_type": "substitutional",
                "defect_name": sub.name,
                "original_species": host_species,
                "dopant_species": dopant_symbol,
                "multiplicity": sub.multiplicity,
                "defect_frac_coords": sub.site.frac_coords.tolist(),
                "defect_supercell": sc_struct.as_dict(),
                "local_environment": local_env,
            })

            for q in charge_states_for_substitution(oxi_struct, site_idx, dopant_symbol):
                charge_records.append({
                    "defect_id": defect_id,
                    "material": material_name,
                    "defect_type": "substitutional",
                    "charge_state": q,
                })

    print(f"  {len(defect_records)} unique defect structures, "
          f"{len(charge_records)} (defect, charge) entries")
    return prim, defect_records, charge_records


# ====================== driver ======================

def _save_outputs(host_structures, all_defect_records, all_charge_records):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "host_structures.json"), "w") as f:
        json.dump(host_structures, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "defect_structures.json"), "w") as f:
        json.dump(all_defect_records, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "charge_states.json"), "w") as f:
        json.dump(all_charge_records, f, indent=2)


def _build_sources() -> dict:
    """Merge TARGET_MATERIALS (MP-fetched) and CIF_DIR (local files) into
    one {material_name: source_dict} mapping. Name collisions between the
    two sources get a '_cif' suffix rather than silently overwriting."""
    sources = {}

    if USE_TARGET_MATERIALS:
        for mat_name, mp_id in TARGET_MATERIALS.items():
            sources[mat_name] = {"kind": "mp", "mp_id": mp_id}

    if CIF_DIR:
        for name, structure, path in load_materials_from_cifs(CIF_DIR, CIF_GLOB):
            key = name if name not in sources else f"{name}_cif"
            sources[key] = {"kind": "cif", "structure": structure, "path": path}

    return sources


import random
from collections import defaultdict


def sample_defect_data(
        defect_json_path: str = "data/defect_structures.json",
        charge_json_path: str = "data/charge_states.json",
        target_count: int = 200,
        strategy: str = "stratified",  # "stratified", "random", "diverse"
        random_seed: int = 42,
        target_charge_count: Optional[int] = None,
) -> tuple:
    """
    从完整的缺陷数据中采样一个子集，尽量保持原始 (material, defect_type) 分布。

    采样在 unique defect 层面进行（保留该 defect 的全部 charge states），
    从而不破坏同一结构不同电荷态之间的一致性。最终 .inp 文件数量 =
    采样后的 charge_records 数量。

    Args:
        defect_json_path: defect_structures.json 路径
        charge_json_path: charge_states.json 路径
        target_count: 目标 unique defect 数量（约）。若提供 target_charge_count
                      则此值会被覆盖为估算值。
        strategy: 采样策略
            - "stratified": 按 (material, defect_type) 分层比例采样（推荐，最保分布）
            - "random": 完全随机采样 unique defects
            - "diverse": 在 stratified 基础上进一步按代表性电荷态分组，增加覆盖
        random_seed: 随机种子
        target_charge_count: 若给定，则优先使最终 charge_records ≈ 此值。
                             内部用平均每 defect 的电荷态数估算需要的 defect 数。

    Returns:
        (sampled_defect_records, sampled_charge_records)
    """
    random.seed(random_seed)

    with open(defect_json_path, "r") as f:
        defect_records = json.load(f)
    with open(charge_json_path, "r") as f:
        charge_records = json.load(f)

    if not defect_records:
        return [], []

    # 建立 defect_id -> charge_states 的映射
    defect_charges = defaultdict(list)
    for charge_rec in charge_records:
        defect_charges[charge_rec["defect_id"]].append(charge_rec["charge_state"])

    # 为每个defect记录添加电荷态信息（浅拷贝，不污染原数据）
    enriched_defects = []
    for d in defect_records:
        d_copy = d.copy()
        d_copy["charge_states"] = sorted(defect_charges.get(d["defect_id"], []))
        enriched_defects.append(d_copy)

    # 若用户指定了目标 charge 数量，则根据平均电荷态数反推需要的 defect 数
    if target_charge_count is not None and target_charge_count > 0:
        total_charges = len(charge_records)
        avg_charges_per_defect = total_charges / max(len(enriched_defects), 1)
        # 保守一点，略多采一些 defect，后面不会截断 charge
        estimated = max(1, int(round(target_charge_count / max(avg_charges_per_defect, 1.0))))
        target_count = min(estimated, len(enriched_defects))
        print(f"  [sample] target_charge_count={target_charge_count} → "
              f"estimated target defects ≈ {target_count} "
              f"(avg {avg_charges_per_defect:.1f} charges/defect)")

    target_count = min(target_count, len(enriched_defects))

    if strategy == "random":
        sampled_defects = random.sample(enriched_defects, target_count)

    elif strategy == "stratified":
        # 按 (material, defect_type) 分层，比例尽量接近原始分布
        layers = defaultdict(list)
        for d in enriched_defects:
            key = (d["material"], d["defect_type"])
            layers[key].append(d)

        total = len(enriched_defects)
        n_layers = len(layers)
        # 使用最大余数法（Hare-Niemeyer）分配整数名额，严格保持比例
        # 当层数远大于 target_count 时（大量 CIF 材料），不强行给每层 1 个，
        # 否则会严重破坏目标规模与主要类别的占比。
        quotas = []
        for key, layer_defects in layers.items():
            exact = target_count * len(layer_defects) / total
            quotas.append((key, layer_defects, exact, int(exact)))

        sampled_defects = []
        assigned = 0
        remainders = []
        for key, layer_defects, exact, floor_n in quotas:
            n = min(floor_n, len(layer_defects))
            # 仅当层数不太多时，给非空层一个保底名额，避免完全丢掉小类
            if n == 0 and layer_defects and n_layers <= target_count:
                n = 1
            if n > 0:
                sampled = random.sample(layer_defects, n)
                sampled_defects.extend(sampled)
                assigned += n
            remainders.append((exact - floor_n, key, layer_defects, n))

        # 用最大余数补齐到 target_count（不超采）
        remainders.sort(reverse=True)
        need = target_count - len(sampled_defects)
        for _, key, layer_defects, already in remainders:
            if need <= 0:
                break
            remaining_in_layer = [
                d for d in layer_defects
                if d["defect_id"] not in {x["defect_id"] for x in sampled_defects}
            ]
            if remaining_in_layer:
                sampled_defects.append(random.choice(remaining_in_layer))
                need -= 1

        # 最终兜底：仍不足则从全局剩余补；若因保底略超则随机下调到 target
        sampled_ids = {d["defect_id"] for d in sampled_defects}
        if len(sampled_defects) < target_count:
            remaining = [d for d in enriched_defects if d["defect_id"] not in sampled_ids]
            extra_n = min(target_count - len(sampled_defects), len(remaining))
            if extra_n > 0:
                sampled_defects.extend(random.sample(remaining, extra_n))
        elif len(sampled_defects) > target_count:
            sampled_defects = random.sample(sampled_defects, target_count)

    elif strategy == "diverse":
        # 在 stratified 框架上，进一步按代表性电荷态分组，提高电荷覆盖
        layers = defaultdict(list)
        for d in enriched_defects:
            key = (d["material"], d["defect_type"])
            layers[key].append(d)

        total = len(enriched_defects)
        sampled_defects = []
        for key, layer_defects in layers.items():
            charge_groups = defaultdict(list)
            for d in layer_defects:
                if d["charge_states"]:
                    rep_charge = d["charge_states"][len(d["charge_states"]) // 2]
                    charge_groups[rep_charge].append(d)
                else:
                    charge_groups[None].append(d)

            layer_target = max(1, int(round(target_count * len(layer_defects) / total)))
            for group in charge_groups.values():
                n = max(1, int(round(layer_target * len(group) / len(layer_defects))))
                n = min(n, len(group))
                sampled_defects.extend(random.sample(group, n))

        # 去重并截断/补齐到 target_count
        seen_ids = set()
        unique_sampled = []
        for d in sampled_defects:
            if d["defect_id"] not in seen_ids:
                seen_ids.add(d["defect_id"])
                unique_sampled.append(d)
        sampled_defects = unique_sampled

        if len(sampled_defects) > target_count:
            sampled_defects = random.sample(sampled_defects, target_count)
        elif len(sampled_defects) < target_count:
            remaining = [d for d in enriched_defects
                         if d["defect_id"] not in {x["defect_id"] for x in sampled_defects}]
            extra_n = min(target_count - len(sampled_defects), len(remaining))
            if extra_n > 0:
                sampled_defects.extend(random.sample(remaining, extra_n))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 提取对应的全部 charge 记录（保持同一 defect 的所有电荷态）
    sampled_ids = {d["defect_id"] for d in sampled_defects}
    sampled_charges = [c for c in charge_records if c["defect_id"] in sampled_ids]

    print(f"Sampled {len(sampled_defects)} unique defects → "
          f"{len(sampled_charges)} charge-state entries (≈ .inp files)")
    print(f"Coverage: {len({d['material'] for d in sampled_defects})} materials, "
          f"{len({d['defect_type'] for d in sampled_defects})} defect types")

    return sampled_defects, sampled_charges


def create_sampled_datasets(
        defect_json_path: str = "data/defect_structures.json",
        charge_json_path: str = "data/charge_states.json",
        target_count: int = 200,
        strategy: str = "stratified",
        output_suffix: str = "_sampled",
        target_charge_count: Optional[int] = 1000,
        random_seed: int = 42,
):
    """
    创建采样后的数据集 JSON 文件，供 write_inputs_from_json 使用。

    默认目标约 1000 条 charge-state（即约 1000 个 .inp 文件）。
    使用 stratified 策略，尽量不破坏原始 (material, defect_type) 分布。

    Args:
        defect_json_path: 原始 defect JSON 路径
        charge_json_path: 原始 charge JSON 路径
        target_count: 备用的 unique-defect 目标数（当 target_charge_count=None 时使用）
        strategy: "stratified"（推荐） / "random" / "diverse"
        output_suffix: 输出文件后缀
        target_charge_count: 期望的最终 charge_records / .inp 数量（优先）
        random_seed: 随机种子

    Returns:
        (defect_out_path, charge_out_path)
    """
    sampled_defects, sampled_charges = sample_defect_data(
        defect_json_path=defect_json_path,
        charge_json_path=charge_json_path,
        target_count=target_count,
        strategy=strategy,
        random_seed=random_seed,
        target_charge_count=target_charge_count,
    )

    out_dir = os.path.dirname(defect_json_path) or "data"
    os.makedirs(out_dir, exist_ok=True)

    defect_out = os.path.join(out_dir, f"defect_structures{output_suffix}.json")
    charge_out = os.path.join(out_dir, f"charge_states{output_suffix}.json")

    with open(defect_out, "w") as f:
        json.dump(sampled_defects, f, indent=2)
    with open(charge_out, "w") as f:
        json.dump(sampled_charges, f, indent=2)

    print(f"Saved sampled data to:")
    print(f"  {defect_out}")
    print(f"  {charge_out}")

    return defect_out, charge_out


if __name__ == "__main__":
    host_structures = {}
    all_defect_records = []
    all_charge_records = []
    failed_materials = {}

    materials = _build_sources()
    print(f"\n{len(materials)} materials queued "
          f"({sum(1 for s in materials.values() if s['kind'] == 'mp')} from MP, "
          f"{sum(1 for s in materials.values() if s['kind'] == 'cif')} from local CIFs)")

    for mat_name, source in materials.items():
        try:
            prim, defects, charges = generate_defect_dataset(mat_name, source)
        except Exception as exc:
            # Don't let one bad material_id / bad CIF / generator error wipe
            # out everything already computed for the other materials --
            # log it, save what we have so far, and move on.
            print(f"  [ERROR] {mat_name} failed: {exc}")
            failed_materials[mat_name] = {
                "mp_id": source.get("mp_id"),
                "path": source.get("path"),
                "error": str(exc),
            }
            _save_outputs(host_structures, all_defect_records, all_charge_records)
            continue

        host_structures[mat_name] = {
            "mp_id": source.get("mp_id"),
            "source_cif": source.get("path"),
            "primitive_structure": prim.as_dict(),
        }
        all_defect_records.extend(defects)
        all_charge_records.extend(charges)
        # Save after every successful material too, so a crash mid-run still
        # leaves usable partial output on disk.
        _save_outputs(host_structures, all_defect_records, all_charge_records)

    if failed_materials:
        with open(os.path.join(OUTPUT_DIR, "failed_materials.json"), "w") as f:
            json.dump(failed_materials, f, indent=2)

    print(f"\n====== totals ======")
    print(f"materials succeeded: {len(host_structures)}")
    print(f"materials failed:    {len(failed_materials)}  {list(failed_materials) or ''}")
    print(f"unique defects:      {len(all_defect_records)}")
    print(f"(defect, charge):    {len(all_charge_records)}")
    print(f"saved to {OUTPUT_DIR}/host_structures.json, "
          f"defect_structures.json, charge_states.json"
          + (", failed_materials.json" if failed_materials else ""))
"""
cp2k_input_writer.py
JSON → CP2K input file converter for defect superc
ells.

Reads your existing defect_structures.json + charge_states.json,
produces one or many .inp files with proper CP2K keywords tailored
to photocatalytic oxides.

Expected integration point: after dataset_builder.py has produced
the JSON files, call write_inputs_from_json().
"""

import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

from pymatgen.core.structure import Structure

# ====================== 材料特定的CP2K参数预设 ======================

# 每个化学元素默认的基组和赝势（TZVP精度，适合力/能量训练数据）
DEFAULT_KIND_SETTINGS = {
    "Ti": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q12", "dft_plus_u": {"l": 2, "u": 3.5}},
    "O": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q6"},
    "W": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q14", "dft_plus_u": {"l": 2, "u": 4.0}},
    "Bi": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q15"},
    "V": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q13"},
    "Sr": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q10"},
    "Nb": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q13"},
    "Ta": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q13"},
    "Mo": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q14"},
    "Cr": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q14"},
    "Fe": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q16"},
    "La": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q11"},
    "Ca": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q10"},
    "Ba": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q10"},
    "Y": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q11"},
    "C": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q4"},
    "N": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q5"},
    "S": {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q6"},
    "H": {"basis": "DZVP-MOLOPT-GTH", "pot": "GTH-PBE-q1"},
    "Zn":  {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q12"},
    "Hf":  {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q12"},
    "Sn":  {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q4"},
    "Pb":  {"basis": "DZVP-MOLOPT-SR-GTH", "pot": "GTH-PBE-q4"},
}

# 材料级预设：CUTOFF、是否用+U、是否自旋极化等。
# 全部固定纯 Gamma 点 + OT，不再生成任何 &KPOINTS 段。
MATERIAL_PRESETS = {
    # TiO2：+U修正常用，PBE即可，需要时开HSE06
    "TiO2_anatase": {"cutoff": 350, "use_u": True, "spin_polarized": False},
    "TiO2_rutile": {"cutoff": 350, "use_u": True, "spin_polarized": False},
    # WO3：可能需要+U，带隙类似
    "WO3": {"cutoff": 350, "use_u": True, "spin_polarized": False},
    # BiVO4：无+U需求，但要注意C2/c对称性
    "BiVO4": {"cutoff": 350, "use_u": False, "spin_polarized": False},
    # SrTiO3：经典立方钙钛矿
    "SrTiO3": {"cutoff": 350, "use_u": False, "spin_polarized": False},
    # 默认回落
    "DEFAULT": {"cutoff": 350, "use_u": False, "spin_polarized": False},
}


def _get_element_string(site) -> str:
    """稳健地获取元素符号，无论specie是Element还是Species对象"""
    specie = site.specie
    # 如果是Species（带氧化态），有.element属性
    if hasattr(specie, 'element'):
        return str(specie.element)
    # 如果是纯Element，直接转字符串
    else:
        return str(specie)


def _get_material_preset(material_name: str) -> dict:
    """模糊匹配材料名到预设（如 'TiO2_anatase' 和 'TiO2_rutile_Sc2x2x2' 都匹配到TiO2预设）"""
    for key in MATERIAL_PRESETS:
        if material_name.startswith(key):
            return MATERIAL_PRESETS[key]
    return MATERIAL_PRESETS["DEFAULT"]


def _parse_valence_electrons(pot_string: str) -> int:
    """从赝势字符串（如 'GTH-PBE-q12'）中解析出价电子数"""
    m = re.search(r'-q(\d+)$', pot_string)
    if not m:
        raise ValueError(f"无法从赝势字符串 '{pot_string}' 解析价电子数"
                         f"（应形如 ...-q<N>），请检查 DEFAULT_KIND_SETTINGS")
    return int(m.group(1))


def _total_valence_electrons(structure: Structure) -> int:
    """按 DEFAULT_KIND_SETTINGS 里各元素赝势的 q 值，累加整个超胞的总价电子数"""
    total = 0
    for site in structure:
        elem = _get_element_string(site)
        settings = DEFAULT_KIND_SETTINGS.get(elem)
        if settings is None:
            raise ValueError(f"No CP2K kind settings for element '{elem}'. "
                             f"Add it to DEFAULT_KIND_SETTINGS in cp2k_input_writer.py")
        total += _parse_valence_electrons(settings["pot"])
    return total


def _determine_spin_multiplicity(structure: Structure, defect_type: str,
                                  charge_state: int) -> tuple:
    """
    确定自旋多重度，并强制满足CP2K的硬性奇偶校验规则。

    CP2K要求：总电子数（= 各原子赝势价电子数之和 - CHARGE）的奇偶性，
    必须和 MULTIPLICITY 的奇偶性匹配：
        电子数为偶数 → MULTIPLICITY 必须是奇数（闭壳层最简单取1）
        电子数为奇数 → MULTIPLICITY 必须是偶数（至少取2，双重态），
                      且此时必须打开 UNRESTRICTED_KOHN_SHAM，否则
                      CP2K会直接abort，报 "LSD: try to use a different
                      multiplicity"。
    这里之前的实现完全没有考虑 CHARGE 的奇偶性，恒定返回1，
    是该报错的根本原因；现在按实际价电子数计算后强制修正。

    缺陷体系真实的基态自旋很难单靠公式猜准，这里仍然只给出满足奇偶
    校验的、自旋最低的合理起点（单重态或双重态），后续如需更高自旋
    态，建议做单点能扫描比较能量。

    Returns:
        (multiplicity, force_uks): force_uks=True 表示无论材料预设
        是否要求自旋极化，电子数为奇数时都必须强制打开UKS。
    """
    n_electrons = _total_valence_electrons(structure) - charge_state
    if n_electrons % 2 == 0:
        # 偶数电子：闭壳层，奇数多重度，最简单取单重态
        multiplicity = 1
        force_uks = False
    else:
        # 奇数电子：必然有未配对电子，偶数多重度，最简单取双重态，
        # 且必须开UKS，否则CHARGE的奇偶性和MULTIPLICITY/UKS设置不
        # 自洽，CP2K会直接报错
        multiplicity = 2
        force_uks = True
    return multiplicity, force_uks


def _build_scf_section() -> str:
    """生成 &SCF 块。固定纯 Gamma 点，始终使用 OT。"""
    lines = [
        "    &SCF",
        "      SCF_GUESS ATOMIC",
        "      MAX_SCF 300",
        "      EPS_SCF 1.0E-5",
        "      IGNORE_CONVERGENCE_FAILURE TRUE",
        "      &OT",
        "        MINIMIZER DIIS",
        "        PRECONDITIONER FULL_SINGLE_INVERSE",
        "        ENERGY_GAP 0.2",
        "      &END OT",
        "    &END SCF",
    ]
    return "\n".join(lines)


def _build_kind_section(element: str, use_u: bool,
                        kind_settings: Optional[dict] = None) -> str:
    """为一个化学元素生成 &KIND 块"""
    if kind_settings is None:
        settings = DEFAULT_KIND_SETTINGS.get(element)
        if settings is None:
            raise ValueError(f"No CP2K kind settings for element '{element}'. "
                             f"Add it to DEFAULT_KIND_SETTINGS in cp2k_input_writer.py")
    else:
        settings = kind_settings

    lines = []
    lines.append(f"    &KIND {element}")
    lines.append(f"      BASIS_SET {settings['basis']}")
    lines.append(f"      POTENTIAL {settings['pot']}")

    # DFT+U（如果材料预设要求且此元素定义了U参数）
    if use_u and "dft_plus_u" in settings:
        u_params = settings["dft_plus_u"]
        lines.append("      &DFT_PLUS_U")
        lines.append(f"        L {u_params['l']}")
        lines.append(f"        U_MINUS_J {u_params['u']}")
        lines.append("      &END DFT_PLUS_U")

    lines.append("    &END KIND")
    return "\n".join(lines)


def _build_coord_section(structure: Structure) -> str:
    """从pymatgen Structure生成 &COORD 块（笛卡尔坐标）"""
    lines = ["    &COORD"]
    for site in structure:
        # 稳健提取纯元素符号
        elem = _get_element_string(site)
        x, y, z = site.coords
        lines.append(f"      {elem}  {x:.8f}  {y:.8f}  {z:.8f}")
    lines.append("    &END COORD")
    return "\n".join(lines)


def _build_cell_section(structure: Structure) -> str:
    """从pymatgen Structure生成 &CELL 块"""
    # 使用晶格矩阵的行向量（CP2K的ABC输入方式）
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    lines = []
    lines.append("    &CELL")
    lines.append(f"      ABC  {a:.6f}  {b:.6f}  {c:.6f}")
    lines.append(f"      ALPHA_BETA_GAMMA  {alpha:.4f}  {beta:.4f}  {gamma:.4f}")
    lines.append("      PERIODIC XYZ")
    lines.append("    &END CELL")
    return "\n".join(lines)


def _build_print_section(write_density_cube: bool = False,
                          cube_stride: str = "4 4 4") -> str:
    """
    生成 &PRINT 块。

    MULLIKEN/HIRSHFELD/LOWDIN/MOMENTS：形成能计算不需要这些人口分析，
    额外占内存，现在整段注释掉（含闭合的 &END，避免CP2K解析报错）。
    注意：parse_cp2k_output() 里读取 hirshfeld_charges 等字段的正则
    会因此匹配不到内容，如果下游还依赖这些字段，需要去掉本函数里的
    注释符 "!" 重新打开对应块。

    E_DENSITY_CUBE 默认关闭：parse_cp2k_output() 里并没有读取cube文件
    的逻辑，之前每个任务都以 STRIDE 1 1 1（不降采样）全网格输出，是纯粹的
    内存+磁盘开销，对所有材料都成立。如果确实需要电荷密度立方体做后续可视化
    /分析，显式传 write_density_cube=True，并优先用更粗的STRIDE（如4 4 4）
    降低内存占用。
    """
    lines = [
        "    &PRINT",
        # 形成能计算不需要这些人口分析，额外占内存，整段注释掉
        # （连闭合的 &END 也必须一起注释，否则 CP2K 解析器会报
        # "not opened / unexpected &END" 之类的错误）
        "      !&MULLIKEN ON",
        "      !&END MULLIKEN",
        "      !&HIRSHFELD ON",
        "      !&END HIRSHFELD",
        "      !&LOWDIN ON",
        "      !&END LOWDIN",
        "      !&MOMENTS ON",
        "      !  PERIODIC .FALSE.",
        "      !&END MOMENTS",
    ]
    if write_density_cube:
        lines.append("      &E_DENSITY_CUBE")
        lines.append(f"        STRIDE {cube_stride}")
        lines.append("      &END E_DENSITY_CUBE")
    lines.append("    &END PRINT")
    return "\n".join(lines)


def _build_cp2k_input(defect_record: dict,
                      charge_state: int,
                      preset: dict,
                      write_density_cube: bool = False) -> str:
    """
    为单个 (缺陷结构, 电荷态) 组合生成完整的CP2K输入文件内容。

    固定纯 Gamma 点 + OT，不写任何 &KPOINTS 段。

    Args:
        defect_record: 来自 defect_structures.json 的一条记录，
                      必须包含 'defect_supercell' 字段。
        charge_state: 电荷态整数值（来自 charge_states.json）。
        preset: 材料预设参数字典。
        write_density_cube: 是否输出电荷密度立方体文件，默认关闭
                            （parse_cp2k_output() 不使用它，见 _build_print_section）。

    Returns:
        CP2K输入文件内容（字符串）。
    """
    # 从JSON恢复Structure
    sc_struct = Structure.from_dict(defect_record["defect_supercell"])
    material = defect_record["material"]
    defect_type = defect_record["defect_type"]

    # 确定所有唯一化学元素
    unique_elements = sorted({_get_element_string(site) for site in sc_struct})

    # 自旋多重度：按实际价电子数奇偶性强制校验（见函数注释），
    # 同时返回是否必须强制开UKS（电子数为奇数时）
    multiplicity, force_uks = _determine_spin_multiplicity(sc_struct, defect_type, charge_state)

    # 是否开自旋极化（有过渡金属时保守开启；电子数为奇数时强制开，
    # 否则 MULTIPLICITY 和 UNRESTRICTED_KOHN_SHAM 不自洽会被CP2K拒绝）
    has_transition_metal = any(
        elem in {"Ti", "V", "Cr", "Fe", "W", "Mo", "Nb", "Ta"}
        for elem in unique_elements
    )
    spin_polarized = preset["spin_polarized"] or has_transition_metal or force_uks

    # 构建输入文件各个部分（始终纯 Gamma + OT + ASPC）
    input_str = f"""\
&GLOBAL
  PROJECT {material}_{defect_record['defect_id'][:8]}_q{charge_state:+d}
  RUN_TYPE ENERGY
  PRINT_LEVEL LOW
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  !STRESS_TENSOR ANALYTICAL

  &DFT
    CHARGE {charge_state}
    MULTIPLICITY {multiplicity}
    UNRESTRICTED_KOHN_SHAM {'TRUE' if spin_polarized else 'FALSE'}
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS

    &MGRID
      CUTOFF {preset['cutoff']}
      REL_CUTOFF 40
    &END MGRID

    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-10
      EXTRAPOLATION ASPC
    &END QS

{_build_scf_section()}

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC

{_build_print_section(write_density_cube)}
  &END DFT

  &SUBSYS
{_build_cell_section(sc_struct)}

{_build_coord_section(sc_struct)}

"""
    # 为每个元素添加 &KIND
    for elem in unique_elements:
        input_str += _build_kind_section(elem, preset["use_u"])
        input_str += "\n\n"

    input_str += "  &END SUBSYS\n&END FORCE_EVAL\n"

    return textwrap.dedent(input_str)


# ====================== 母体（无缺陷）结构支持 ======================
#
# 缺陷结构目前没有做任何弛豫，所以完美晶体可以直接从缺陷记录自身的描述
# "回填"重构出来，不需要额外读取/存储单独的host/bulk/pristine结构：
#   - vacancy（空位）      : 把被移走的原子按记录的位置放回去
#   - substitutional（替位）: 把替换进来的原子改回原来的元素
#   - interstitial（间隙）  : 把插入进来的原子删掉
# 电荷统一在写入时显式配平为0。
#
# 与 sampled JSON 字段对齐：
#   vacancy        → defect_species (被移走的物种) + defect_frac_coords
#   substitutional → original_species (宿主) + dopant_species (替位原子)
#                   + defect_frac_coords（定位优先用唯一 dopant 匹配）

def _locate_defect_site_index(struct: Structure, defect_record: dict,
                               fallback_species: Optional[str]) -> int:
    """
    定位缺陷在 defect_supercell 里对应的位点下标。
    优先使用JSON里显式给出的下标字段，没有的话退化为按元素种类唯一匹配。
    """
    for key in ("defect_site_index", "site_index"):
        if defect_record.get(key) is not None:
            return int(defect_record[key])

    if fallback_species is None:
        raise ValueError("记录里既没有 defect_site_index/site_index，也没有"
                          "可用于定位的物种字段，无法定位缺陷位点")

    matches = [i for i, s in enumerate(struct) if _get_element_string(s) == fallback_species]
    if len(matches) == 0:
        raise ValueError(f"结构中找不到元素 '{fallback_species}' 对应的位点，"
                          f"请在JSON里提供 defect_site_index")
    if len(matches) > 1:
        raise ValueError(f"元素 '{fallback_species}' 在结构里出现了 {len(matches)} 次，"
                          f"无法唯一定位缺陷位点，请在JSON里提供 defect_site_index")
    return matches[0]


def _reconstruct_pristine_structure(defect_record: dict, defect_struct: Structure) -> Structure:
    """对单条缺陷记录的 defect_supercell 做回填，得到未弛豫的完美晶体。"""
    struct = defect_struct.copy()
    defect_type = (defect_record.get("defect_type") or "").lower()

    # 原来（缺陷引入前）该位点上的元素
    # vacancy 在 sampled JSON 里用 defect_species 表示被移走的物种；
    # substitutional 用 original_species。
    original_species = (defect_record.get("original_species")
                        or defect_record.get("host_species")
                        or defect_record.get("defect_species"))

    # 缺陷引入的元素（vacancy 没有；substitutional 用 dopant_species）
    defect_species = (defect_record.get("dopant_species")
                      or defect_record.get("defect_species"))

    if defect_type == "vacancy":
        if original_species is None:
            raise ValueError("vacancy记录缺少 defect_species / original_species / host_species，"
                              "不知道要把什么元素放回去")
        frac_coords = (defect_record.get("defect_frac_coords")
                       or defect_record.get("defect_site_frac_coords")
                       or defect_record.get("vacancy_frac_coords"))
        if frac_coords is None:
            raise ValueError("vacancy记录缺少缺陷位置坐标"
                              "（defect_frac_coords / vacancy_frac_coords），无法回填")
        struct.append(original_species, frac_coords,
                      coords_are_cartesian=False, validate_proximity=False)

    elif defect_type in ("substitutional", "substitution"):
        if original_species is None:
            raise ValueError("substitutional记录缺少 original_species/host_species，"
                              "不知道要把该位点改回什么元素")
        # 用 dopant_species 唯一定位（sampled 数据中 dopant 总是唯一的）
        idx = _locate_defect_site_index(struct, defect_record, defect_species)
        struct.replace(idx, original_species)

    elif defect_type == "interstitial":
        idx = _locate_defect_site_index(struct, defect_record, defect_species)
        struct.remove_sites([idx])

    else:
        raise ValueError(f"未知的 defect_type '{defect_type}'，不知道怎么回填成完美晶体")

    return struct


def _make_pristine_record(defect_record: dict) -> dict:
    """
    直接对该material任取的一条缺陷记录做回填，重构出对应的完美晶体，
    包装成可以直接复用 _build_cp2k_input() 的"伪缺陷记录"。

    defect_id 用 "pristine_<material>" 的md5哈希代替（母体没有真正的
    defect_id，但保持32位十六进制字符串形式，其他地方到处出现的
    defect_id[:8] 切片写法不需要特殊处理，天然对齐命名风格）。
    """
    material = defect_record["material"]
    defect_struct = Structure.from_dict(defect_record["defect_supercell"])
    pristine_struct = _reconstruct_pristine_structure(defect_record, defect_struct)

    full_id = hashlib.md5(f"pristine_{material}".encode("utf-8")).hexdigest()
    return {
        "defect_supercell": pristine_struct.as_dict(),
        "material": material,
        "defect_type": "pristine",
        "defect_id": full_id,
        "defect_species": "host",
    }


def _write_single_input(defect_record: dict, charge_state: int,
                         preset: dict, output_dir: str,
                         write_density_cube: bool = False) -> str:
    """
    按统一命名风格生成并写入单个 .inp 文件，返回文件路径。
    缺陷态和母体（pristine）都走这一个函数，保证命名规则完全一致：
        {material}_{defect_type}_{defect_species}_q{charge:+d}_{id[:8]}.inp
    例如缺陷: TiO2_vacancy_O_q+2_abc12345.inp
          母体: TiO2_pristine_host_q+0_9f1c2a3b.inp
    """
    material = defect_record["material"]
    defect_id = defect_record["defect_id"]
    # 命名用物种优先级：dopant_species（替位）> defect_species（空位/母体）> original_species
    defect_species = (defect_record.get("dopant_species")
                      or defect_record.get("defect_species")
                      or defect_record.get("original_species", "X"))
    filename = (
        f"{material}_{defect_record['defect_type']}_"
        f"{defect_species}_q{charge_state:+d}_"
        f"{defect_id[:8]}.inp"
    )
    filepath = os.path.join(output_dir, filename)

    cp2k_input = _build_cp2k_input(defect_record, charge_state, preset,
                                    write_density_cube=write_density_cube)
    with open(filepath, "w") as f:
        f.write(cp2k_input)
    return filepath


def write_inputs_from_json(
        defect_json_path: str = "data/defect_structures.json",
        charge_json_path: str = "data/charge_states.json",
        output_dir: str = "cp2k_inputs",
        max_files: Optional[int] = None,
        include_pristine: bool = True,
        write_density_cube: bool = False,
) -> List[str]:
    """
    从你的JSON文件批量生成CP2K输入文件。

    推荐工作流（约 1k 条 .inp，保持材料/缺陷类型分布）:
        from get_defect_data import create_sampled_datasets
        defect_json, charge_json = create_sampled_datasets(
            defect_json_path="data/defect_structures.json",
            charge_json_path="data/charge_states.json",
            target_charge_count=1000,   # 期望最终 .inp 数量
            strategy="stratified",      # 按 (material, defect_type) 分层，最保分布
        )
        input_files = write_inputs_from_json(
            defect_json_path=defect_json,
            charge_json_path=charge_json,
            output_dir="cp2k_inputs",
            max_files=None,
        )

    注意: max_files 只是简单截断 charge_records 前 N 条，会破坏分布；
    大规模缩减请务必先用 create_sampled_datasets / sample_defect_data。

    母体（无缺陷）参考结构:
        每个在本批次中实际出现过的 material，会额外生成一份 CHARGE 0、
        不含缺陷的母体超胞 .inp（文件名里 defect_type=pristine，
        defect_species=host），用于后续计算缺陷形成能等需要母体总能量
        做参照的场景。

        由于缺陷结构目前没有做任何弛豫，母体结构直接从该material任取一条
        缺陷记录"回填"重构得到，不需要额外读取/存储单独的host结构：
          - defect_type=vacancy       : 按 defect_species（或 original_species /
            host_species）+ defect_frac_coords（或 defect_site_frac_coords /
            vacancy_frac_coords）把被移走的原子放回去
          - defect_type=substitutional: 把 dopant_species 对应的位点改回
            original_species/host_species；位点优先用 defect_site_index /
            site_index 定位，缺失时退化为按元素种类唯一匹配
          - defect_type=interstitial  : 把 defect_species 对应的位点删掉，
            定位方式同上
        若某条记录缺少必要字段（如 vacancy 缺坐标、substitutional 缺
        original_species，或元素种类不唯一又没给下标）导致无法回填，会
        跳过该材料的母体文件生成并打印提示，不会中断整个批次。
        设 include_pristine=False 可以关闭这个行为，只生成缺陷态。

    Args:
        defect_json_path: defect_structures.json 路径
        charge_json_path: charge_states.json 路径
        output_dir: 输出目录
        max_files: 最大生成文件数（仅用于快速测试；正式缩减请用上游采样），None=全部
        include_pristine: 是否为本批次中出现的每个material额外生成一份
                           母体（无缺陷）参考 .inp，默认开启
        write_density_cube: 是否输出E_DENSITY_CUBE，默认关闭（下游
                             parse_cp2k_output()不使用它，开启会显著增加
                             内存/磁盘占用，仅在确实需要电荷密度可视化时打开）

    Returns:
        生成的文件路径列表（缺陷态在前，母体参考在后）
    """
    with open(defect_json_path, "r") as f:
        defect_records = json.load(f)
    with open(charge_json_path, "r") as f:
        charge_records = json.load(f)

    # 建立 defect_id -> defect_record 的查找表
    defect_lookup = {d["defect_id"]: d for d in defect_records}

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # 记录本批次实际用到的 material -> 任意一条对应的 defect_record
    # （用来去该material的host_supercell等字段，同一material只需生成一份母体）
    materials_seen: Dict[str, dict] = {}

    for i, charge_rec in enumerate(charge_records):
        if max_files and i >= max_files:
            break

        defect_id = charge_rec["defect_id"]
        charge_state = charge_rec["charge_state"]
        defect_record = defect_lookup[defect_id]

        # 获取材料预设
        material = defect_record["material"]
        preset = _get_material_preset(material)
        materials_seen.setdefault(material, defect_record)

        filepath = _write_single_input(defect_record, charge_state, preset, output_dir,
                                        write_density_cube=write_density_cube)
        generated_files.append(filepath)

    n_defect_files = len(generated_files)

    # 顺带为本批次涉及的每个material生成一份母体（无缺陷）参考 .inp
    # —— 直接对该material任取的一条缺陷记录做"回填"重构，不需要额外的
    # host/bulk结构字段（缺陷本身没做弛豫，所以这个回填是精确的）。
    n_pristine_files = 0
    if include_pristine:
        for material, sample_record in materials_seen.items():
            try:
                pristine_record = _make_pristine_record(sample_record)
            except (ValueError, KeyError) as e:
                print(f"[WARN] material '{material}' 无法从缺陷记录回填重构母体结构"
                      f"（{e}），跳过母体 .inp 生成。")
                continue

            preset = _get_material_preset(material)
            filepath = _write_single_input(pristine_record, 0, preset, output_dir,
                                            write_density_cube=write_density_cube)
            generated_files.append(filepath)
            n_pristine_files += 1

    print(f"Generated {n_defect_files} defect + {n_pristine_files} pristine "
          f"CP2K input files ({len(generated_files)} total) in {output_dir}/")
    return generated_files


# ====================== 输出解析器 ======================

def parse_cp2k_output(out_path: str) -> dict:
    """
    解析CP2K输出文件，提取训练SE3网络所需的核心物理量。

    Returns:
        dict with keys:
        - total_energy (Hartree)
        - forces (Nx3 array, Hartree/Bohr)
        - stress (6-component array, bar)
        - mulliken_charges
        - hirshfeld_charges
        - lowdin_charges
        - homo_energy, lumo_energy, gap (eV)
    """
    # 这里我给出框架和关键正则表达式，你可以封装成解析类
    with open(out_path, "r") as f:
        text = f.read()

    result = {}

    # 总能量
    m = re.search(r'ENERGY\| Total FORCE_EVAL.*?(\S+)$', text, re.MULTILINE)
    if m:
        result["total_energy_Ha"] = float(m.group(1))

    # 原子力（在 "ATOMIC FORCES" 后的表格中）
    # 格式通常是每行：元素 idx X Y Z
    force_section = re.search(
        r'ATOMIC FORCES.*?\n(.*?)(?:\n\s*\n|SUM OF ATOMIC FORCES)',
        text, re.DOTALL
    )
    if force_section:
        lines = force_section.group(1).strip().split('\n')
        forces = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                forces.append([float(x) for x in parts[-3:]])
        result["forces_Ha_per_Bohr"] = forces

    # 应力张量
    m = re.search(r'STRESS TENSOR.*?\[bar\].*?\n(.*?)(?:\n\s*\n)', text, re.DOTALL)
    if m:
        # 提取 x,y,z,x, y,z,x,y,z 分量
        pass

    # Hirshfeld电荷
    hirshfeld_section = re.search(
        r'Hirshfeld Charges.*?\n.*?\n(.*?)(?:\n\s*\n|Total)',
        text, re.DOTALL
    )
    if hirshfeld_section:
        lines = hirshfeld_section.group(1).strip().split('\n')
        charges = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                charges.append(float(parts[-1]))
        result["hirshfeld_charges"] = charges

    # HOMO/LUMO
    m = re.search(r'HOMO - LUMO.*?(\S+)\s+(\S+)\s+(\S+)', text)
    if m:
        result["homo_eV"] = float(m.group(1))
        result["lumo_eV"] = float(m.group(2))
        result["gap_eV"] = float(m.group(3))

    return result


# ====================== 命令行入口 ======================

if __name__ == "__main__":
    import sys

    # 用法: python cp2k_input_writer.py [max_files]
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else None

    files = write_inputs_from_json(
        defect_json_path="data/defect_structures.json",
        charge_json_path="data/charge_states.json",
        output_dir="scripts/cp2k_inputs",
        max_files=max_files,
    )

    print(f"\nFirst file preview ({files[0]}):")
    with open(files[0], "r") as f:
        print(f.read()[:1500])
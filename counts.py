#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIF数据统计脚本 - 材料科学全面统计分析
作者: AI Assistant
功能: 统计CIF文件中的晶体结构、化学组成、对称性等材料相关特征
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class CIFStatistics:
    def __init__(self, data_path='data/cif_files'):
        """初始化统计类"""
        self.data_path = data_path
        self.cif_files = []
        self.stats_data = []
        self.summary_stats = {}

    def parse_cif_file(self, filepath):
        """解析单个CIF文件，提取关键信息"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.readlines()

            # 初始化字典存储数据
            cif_data = {
                'filename': os.path.basename(filepath),
                'formula': None,
                'formula_sum': None,
                'space_group': None,
                'space_group_number': None,
                'cell_a': None,
                'cell_b': None,
                'cell_c': None,
                'cell_alpha': None,
                'cell_beta': None,
                'cell_gamma': None,
                'cell_volume': None,
                'cell_units': None,
                'atoms': [],
                'atom_types': [],
                'site_multiplicities': [],
                'z_values': [],
                'fract_x': [],
                'fract_y': [],
                'fract_z': [],
                'occupancies': [],
                'density': None,
                'num_atoms': 0,
                'num_elements': 0,
                'elements': [],
                'element_counts': {},
                'max_occupancy': 0,
                'min_occupancy': 1,
                'avg_occupancy': 0,
                'symmetry_equiv_pos': [],
                'has_disorder': False,
                'has_vacancy': False,
            }

            # 解析CIF内容
            i = 0
            while i < len(content):
                line = content[i].strip()

                # 获取化学式
                if line.startswith('_chemical_formula_structural'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['formula'] = ' '.join(parts[1:])

                elif line.startswith('_chemical_formula_sum'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['formula_sum'] = ' '.join(parts[1:]).strip("'")

                # 获取空间群信息
                elif line.startswith('_symmetry_space_group_name_H-M'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['space_group'] = ' '.join(parts[1:]).strip("'")

                elif line.startswith('_symmetry_Int_Tables_number'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['space_group_number'] = int(parts[1])

                # 获取晶胞参数
                elif line.startswith('_cell_length_a'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_a'] = float(parts[1])
                elif line.startswith('_cell_length_b'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_b'] = float(parts[1])
                elif line.startswith('_cell_length_c'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_c'] = float(parts[1])
                elif line.startswith('_cell_angle_alpha'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_alpha'] = float(parts[1])
                elif line.startswith('_cell_angle_beta'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_beta'] = float(parts[1])
                elif line.startswith('_cell_angle_gamma'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_gamma'] = float(parts[1])

                # 获取晶胞体积和Z值
                elif line.startswith('_cell_volume'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_volume'] = float(parts[1])
                elif line.startswith('_cell_formula_units_Z'):
                    parts = line.split()
                    if len(parts) >= 2:
                        cif_data['cell_units'] = int(parts[1])

                # 获取对称操作
                elif line.startswith('_symmetry_equiv_pos_as_xyz'):
                    # 跳过loop_行
                    i += 1
                    while i < len(content):
                        line_sym = content[i].strip()
                        if not line_sym.startswith("'") and not line_sym.startswith('"'):
                            break
                        cif_data['symmetry_equiv_pos'].append(line_sym)
                        i += 1
                    continue

                # 获取原子位点
                elif line.startswith('_atom_site_type_symbol') or line.startswith('loop_'):
                    # 跳过表头直到数据行
                    i += 1
                    while i < len(content):
                        line_atom = content[i].strip()
                        if not line_atom or line_atom.startswith('_'):
                            i += 1
                            continue

                        # 解析原子数据行
                        parts = line_atom.split()
                        if len(parts) >= 7:
                            try:
                                atom_type = parts[0]
                                label = parts[1]
                                multiplicity = int(parts[2])
                                x = float(parts[3])
                                y = float(parts[4])
                                z = float(parts[5])
                                occupancy = float(parts[6])

                                cif_data['atoms'].append({
                                    'type': atom_type,
                                    'label': label,
                                    'multiplicity': multiplicity,
                                    'x': x, 'y': y, 'z': z,
                                    'occupancy': occupancy
                                })
                                cif_data['atom_types'].append(atom_type)
                                cif_data['site_multiplicities'].append(multiplicity)
                                cif_data['z_values'].append(z)
                                cif_data['fract_x'].append(x)
                                cif_data['fract_y'].append(y)
                                cif_data['fract_z'].append(z)
                                cif_data['occupancies'].append(occupancy)

                                # 检查无序和空位
                                if occupancy < 1.0:
                                    cif_data['has_vacancy'] = True
                                if occupancy != 1.0 and occupancy != 0:
                                    cif_data['has_disorder'] = True

                            except (ValueError, IndexError) as e:
                                pass
                        i += 1
                    break

                i += 1

            # 计算原子统计信息
            if cif_data['atoms']:
                cif_data['num_atoms'] = len(cif_data['atoms'])
                elements = [atom['type'] for atom in cif_data['atoms']]
                cif_data['elements'] = list(set(elements))
                cif_data['num_elements'] = len(set(elements))
                cif_data['element_counts'] = dict(Counter(elements))

                # 计算占用率统计
                occ = cif_data['occupancies']
                cif_data['max_occupancy'] = max(occ) if occ else 0
                cif_data['min_occupancy'] = min(occ) if occ else 0
                cif_data['avg_occupancy'] = np.mean(occ) if occ else 0

                # 计算密度（粗略估计）
                if cif_data['cell_volume'] and cif_data['atoms']:
                    # 使用原子质量近似（仅常见元素）
                    atomic_masses = {
                        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012,
                        'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                        'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
                        'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065,
                        'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
                        'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
                        'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
                        'Cu': 63.546, 'Zn': 65.380, 'Ga': 69.723, 'Ge': 72.630,
                        'As': 74.922, 'Se': 78.960, 'Br': 79.904, 'Kr': 83.798,
                        'Rb': 85.468, 'Sr': 87.620, 'Y': 88.906, 'Zr': 91.224,
                        'Nb': 92.906, 'Mo': 95.950, 'Tc': 98, 'Ru': 101.07,
                        'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
                        'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60,
                        'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33,
                        'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
                        'Pm': 145, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25,
                        'Tb': 158.93, 'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26,
                        'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
                        'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23,
                        'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
                        'Tl': 204.38, 'Pb': 207.20, 'Bi': 208.98, 'Po': 209,
                        'At': 210, 'Rn': 222, 'Fr': 223, 'Ra': 226,
                        'Ac': 227, 'Th': 232.04, 'Pa': 231.04, 'U': 238.03,
                        'Np': 237, 'Pu': 244, 'Am': 243, 'Cm': 247,
                        'Bk': 247, 'Cf': 251, 'Es': 252, 'Fm': 257,
                        'Md': 258, 'No': 259, 'Lr': 262, 'Rf': 267,
                        'Db': 268, 'Sg': 269, 'Bh': 270, 'Hs': 270,
                        'Mt': 278, 'Ds': 281, 'Rg': 282, 'Cn': 285
                    }
                    mass_sum = 0
                    for atom in cif_data['atoms']:
                        mass = atomic_masses.get(atom['type'], 0)
                        mass_sum += mass * atom['occupancy']

                    # 密度 = 质量 / 体积 (g/cm³)
                    # 1 Å³ = 1e-24 cm³, 原子质量单位 g/mol * 1.66054e-24 g
                    avogadro = 6.02214e23
                    if mass_sum > 0 and cif_data['cell_volume'] > 0:
                        mass_g = mass_sum / avogadro
                        volume_cm3 = cif_data['cell_volume'] * 1e-24
                        cif_data['density'] = mass_g / volume_cm3

            return cif_data

        except Exception as e:
            print(f"Error parsing {filepath}: {str(e)}")
            return None

    def load_all_cif_files(self):
        """加载所有CIF文件"""
        print(f"Loading CIF files from {self.data_path}...")
        cif_pattern = os.path.join(self.data_path, '*.cif')
        self.cif_files = glob.glob(cif_pattern)
        print(f"Found {len(self.cif_files)} CIF files")

        for filepath in self.cif_files:
            print(f"Processing: {os.path.basename(filepath)}")
            cif_data = self.parse_cif_file(filepath)
            if cif_data and cif_data['num_atoms'] > 0:
                self.stats_data.append(cif_data)
            else:
                print(f"Warning: {os.path.basename(filepath)} could not be parsed or contains no atoms")

        print(f"Successfully parsed {len(self.stats_data)} CIF files")
        return self.stats_data

    def calculate_statistics(self):
        """计算各种统计指标"""
        if not self.stats_data:
            print("No data loaded. Please run load_all_cif_files() first.")
            return {}

        # 基础统计
        total_files = len(self.stats_data)

        # 晶胞参数统计
        cell_params = ['cell_a', 'cell_b', 'cell_c', 'cell_alpha',
                       'cell_beta', 'cell_gamma', 'cell_volume']
        cell_stats = {}
        for param in cell_params:
            values = [d[param] for d in self.stats_data if d[param] is not None]
            if values:
                cell_stats[param] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'count': len(values)
                }

        # 原子统计
        atom_counts = [d['num_atoms'] for d in self.stats_data]
        element_counts_all = []
        for d in self.stats_data:
            element_counts_all.extend(d['elements'])

        # 化学式统计
        formulas = [d['formula'] for d in self.stats_data if d['formula']]
        formula_sums = [d['formula_sum'] for d in self.stats_data if d['formula_sum']]

        # 空间群统计
        space_groups = [d['space_group'] for d in self.stats_data if d['space_group']]
        space_group_numbers = [d['space_group_number'] for d in self.stats_data
                               if d['space_group_number'] is not None]

        # 占用率统计
        occupancies_all = []
        for d in self.stats_data:
            occupancies_all.extend(d['occupancies'])

        # 密度统计
        densities = [d['density'] for d in self.stats_data if d['density'] is not None]

        # 元素组成统计
        element_composition = defaultdict(list)
        for d in self.stats_data:
            for element, count in d['element_counts'].items():
                element_composition[element].append(count)

        # 对称性统计
        symmetry_pos_counts = [len(d['symmetry_equiv_pos']) for d in self.stats_data]

        # 总结统计
        self.summary_stats = {
            'total_files': total_files,
            'cell_parameters': cell_stats,
            'atom_statistics': {
                'num_atoms': {
                    'mean': np.mean(atom_counts) if atom_counts else 0,
                    'median': np.median(atom_counts) if atom_counts else 0,
                    'std': np.std(atom_counts) if atom_counts else 0,
                    'min': np.min(atom_counts) if atom_counts else 0,
                    'max': np.max(atom_counts) if atom_counts else 0,
                    'total': sum(atom_counts)
                },
                'num_elements': {
                    'mean': np.mean([d['num_elements'] for d in self.stats_data]),
                    'median': np.median([d['num_elements'] for d in self.stats_data]),
                    'min': np.min([d['num_elements'] for d in self.stats_data]),
                    'max': np.max([d['num_elements'] for d in self.stats_data])
                },
                'element_counts': dict(Counter(element_counts_all)),
                'occupancy': {
                    'mean': np.mean(occupancies_all) if occupancies_all else 0,
                    'std': np.std(occupancies_all) if occupancies_all else 0,
                    'min': np.min(occupancies_all) if occupancies_all else 0,
                    'max': np.max(occupancies_all) if occupancies_all else 0
                }
            },
            'chemical_formulas': {
                'unique_formulas': len(set(formulas)),
                'unique_formula_sums': len(set(formula_sums)),
                'most_common_formulas': Counter(formulas).most_common(10),
                'most_common_formula_sums': Counter(formula_sums).most_common(10)
            },
            'space_groups': {
                'unique': len(set(space_groups)),
                'most_common': Counter(space_groups).most_common(10),
                'numbers_range': {
                    'min': np.min(space_group_numbers) if space_group_numbers else 0,
                    'max': np.max(space_group_numbers) if space_group_numbers else 0,
                    'mean': np.mean(space_group_numbers) if space_group_numbers else 0
                }
            },
            'density': {
                'mean': np.mean(densities) if densities else 0,
                'std': np.std(densities) if densities else 0,
                'min': np.min(densities) if densities else 0,
                'max': np.max(densities) if densities else 0,
                'count': len(densities)
            } if densities else None,
            'disorder_statistics': {
                'has_disorder': sum(1 for d in self.stats_data if d['has_disorder']),
                'has_vacancy': sum(1 for d in self.stats_data if d['has_vacancy']),
                'total_structures': total_files,
                'disorder_fraction': sum(1 for d in self.stats_data if d['has_disorder']) / total_files,
                'vacancy_fraction': sum(1 for d in self.stats_data if d['has_vacancy']) / total_files
            },
            'symmetry_operations': {
                'min': np.min(symmetry_pos_counts) if symmetry_pos_counts else 0,
                'max': np.max(symmetry_pos_counts) if symmetry_pos_counts else 0,
                'mean': np.mean(symmetry_pos_counts) if symmetry_pos_counts else 0
            },
            'element_composition': {k: {'mean': np.mean(v), 'total': sum(v), 'count': len(v)}
                                    for k, v in element_composition.items()},
            'crystal_system_estimate': self._estimate_crystal_systems()
        }

        return self.summary_stats

    def _estimate_crystal_systems(self):
        """根据晶胞参数估计晶体系统"""
        systems = []
        for d in self.stats_data:
            if all(d[param] is not None for param in ['cell_a', 'cell_b', 'cell_c',
                                                      'cell_alpha', 'cell_beta', 'cell_gamma']):
                a, b, c = d['cell_a'], d['cell_b'], d['cell_c']
                alpha, beta, gamma = d['cell_alpha'], d['cell_beta'], d['cell_gamma']

                # 判断晶体系统（近似）
                if abs(a - b) < 0.01 and abs(b - c) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(
                        gamma - 90) < 0.1:
                    systems.append('Cubic')
                elif abs(a - b) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
                    systems.append('Tetragonal')
                elif abs(a - b) < 0.01 and abs(a - c) < 0.01 and abs(alpha - 90) < 0.1:
                    systems.append('Rhombohedral/Trigonal')
                elif abs(a - b) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
                    systems.append('Hexagonal')
                elif abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
                    systems.append('Orthorhombic')
                elif abs(alpha - 90) < 0.1 and abs(gamma - 90) < 0.1:
                    systems.append('Monoclinic')
                elif abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1:
                    systems.append('Triclinic')
                else:
                    systems.append('Triclinic')

        return dict(Counter(systems))

    def generate_report(self):
        """生成统计报告"""
        if not self.summary_stats:
            self.calculate_statistics()

        report = []
        report.append("=" * 80)
        report.append("CIF数据统计分析报告")
        report.append("=" * 80)
        report.append(f"\n总文件数: {self.summary_stats['total_files']}")

        # 晶胞参数统计
        report.append("\n【晶胞参数统计】")
        report.append("-" * 60)
        for param, stats in self.summary_stats['cell_parameters'].items():
            report.append(f"\n{param}:")
            report.append(f"  平均值: {stats['mean']:.6f}")
            report.append(f"  中位数: {stats['median']:.6f}")
            report.append(f"  标准差: {stats['std']:.6f}")
            report.append(f"  最小值: {stats['min']:.6f}")
            report.append(f"  最大值: {stats['max']:.6f}")
            report.append(f"  25%分位: {stats['q25']:.6f}")
            report.append(f"  75%分位: {stats['q75']:.6f}")

        # 原子统计
        report.append("\n【原子统计】")
        report.append("-" * 60)
        atom_stats = self.summary_stats['atom_statistics']
        report.append(f"总原子数: {atom_stats['num_atoms']['total']}")
        report.append(f"平均每结构原子数: {atom_stats['num_atoms']['mean']:.2f}")
        report.append(f"原子数范围: {atom_stats['num_atoms']['min']} - {atom_stats['num_atoms']['max']}")
        report.append(f"平均元素种类数: {atom_stats['num_elements']['mean']:.2f}")
        report.append(f"元素种类范围: {atom_stats['num_elements']['min']} - {atom_stats['num_elements']['max']}")

        # 元素分布
        report.append("\n【元素分布 Top 10】")
        report.append("-" * 60)
        elements_top = sorted(atom_stats['element_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        for element, count in elements_top:
            report.append(f"  {element}: {count} 个原子位点")

        # 化学式统计
        report.append("\n【化学式统计】")
        report.append("-" * 60)
        formula_stats = self.summary_stats['chemical_formulas']
        report.append(f"唯一化学式数量: {formula_stats['unique_formulas']}")
        report.append(f"唯一化学式总和数量: {formula_stats['unique_formula_sums']}")
        report.append("\n最常见化学式 Top 5:")
        for formula, count in formula_stats['most_common_formulas'][:5]:
            report.append(f"  {formula}: {count} 个结构")

        # 空间群统计
        report.append("\n【空间群统计】")
        report.append("-" * 60)
        sg_stats = self.summary_stats['space_groups']
        report.append(f"唯一空间群数量: {sg_stats['unique']}")
        report.append(f"空间群编号范围: {sg_stats['numbers_range']['min']} - {sg_stats['numbers_range']['max']}")
        report.append("\n最常见空间群 Top 5:")
        for sg, count in sg_stats['most_common'][:5]:
            report.append(f"  {sg}: {count} 个结构")

        # 密度统计
        if self.summary_stats['density']:
            report.append("\n【密度统计】")
            report.append("-" * 60)
            dens = self.summary_stats['density']
            report.append(f"平均密度: {dens['mean']:.3f} g/cm³")
            report.append(f"密度范围: {dens['min']:.3f} - {dens['max']:.3f} g/cm³")
            report.append(f"标准差: {dens['std']:.3f} g/cm³")

        # 无序和空位统计
        report.append("\n【结构无序统计】")
        report.append("-" * 60)
        disorder = self.summary_stats['disorder_statistics']
        report.append(f"含无序结构: {disorder['has_disorder']} ({disorder['disorder_fraction'] * 100:.1f}%)")
        report.append(f"含空位结构: {disorder['has_vacancy']} ({disorder['vacancy_fraction'] * 100:.1f}%)")

        # 晶体系统分布
        report.append("\n【晶体系统分布】")
        report.append("-" * 60)
        for system, count in self.summary_stats['crystal_system_estimate'].items():
            fraction = count / self.summary_stats['total_files'] * 100
            report.append(f"  {system}: {count} ({fraction:.1f}%)")

        # 元素组成统计
        report.append("\n【元素组成分析】")
        report.append("-" * 60)
        for element, stats in sorted(self.summary_stats['element_composition'].items(),
                                     key=lambda x: x[1]['total'], reverse=True)[:10]:
            report.append(f"  {element}: 平均 {stats['mean']:.2f} 个/结构, 总计 {stats['total']} 个")

        # 对称操作统计
        report.append("\n【对称操作统计】")
        report.append("-" * 60)
        sym = self.summary_stats['symmetry_operations']
        report.append(f"对称操作数范围: {sym['min']} - {sym['max']}")
        report.append(f"平均对称操作数: {sym['mean']:.2f}")

        # 占用率统计
        report.append("\n【原子占用率统计】")
        report.append("-" * 60)
        occ = self.summary_stats['atom_statistics']['occupancy']
        report.append(f"平均占用率: {occ['mean']:.3f}")
        report.append(f"占用率范围: {occ['min']:.3f} - {occ['max']:.3f}")
        report.append(f"占用率标准差: {occ['std']:.3f}")

        # 总结
        report.append("\n" + "=" * 80)
        report.append("统计报告生成完成")
        report.append("=" * 80)

        return "\n".join(report)

    def save_statistics_to_json(self, output_file='cif_statistics.json'):
        """保存统计数据到JSON文件"""
        if not self.summary_stats:
            self.calculate_statistics()

        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_stats = convert_to_serializable(self.summary_stats)

        with open(output_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        print(f"统计数据已保存到: {output_file}")

    def save_detailed_data_to_csv(self, output_file='cif_detailed_data.csv'):
        """保存详细数据到CSV文件"""
        if not self.stats_data:
            print("No data to save.")
            return

        # 准备DataFrame
        rows = []
        for d in self.stats_data:
            rows.append({
                'filename': d['filename'],
                'formula': d['formula'],
                'formula_sum': d['formula_sum'],
                'space_group': d['space_group'],
                'space_group_number': d['space_group_number'],
                'cell_a': d['cell_a'],
                'cell_b': d['cell_b'],
                'cell_c': d['cell_c'],
                'cell_alpha': d['cell_alpha'],
                'cell_beta': d['cell_beta'],
                'cell_gamma': d['cell_gamma'],
                'cell_volume': d['cell_volume'],
                'cell_units': d['cell_units'],
                'num_atoms': d['num_atoms'],
                'num_elements': d['num_elements'],
                'elements': ', '.join(d['elements']),
                'density': d['density'],
                'max_occupancy': d['max_occupancy'],
                'min_occupancy': d['min_occupancy'],
                'avg_occupancy': d['avg_occupancy'],
                'has_disorder': d['has_disorder'],
                'has_vacancy': d['has_vacancy'],
                'symmetry_ops': len(d['symmetry_equiv_pos'])
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"详细数据已保存到: {output_file}")


def main():
    """主函数"""
    # 创建统计对象
    stats = CIFStatistics(data_path='data/cif_files')

    # 加载所有CIF文件
    stats.load_all_cif_files()

    # 计算统计
    stats.calculate_statistics()

    # 生成报告
    report = stats.generate_report()
    print(report)

    # 保存结果
    stats.save_statistics_to_json('cif_statistics.json')
    stats.save_detailed_data_to_csv('cif_detailed_data.csv')

    # 保存报告到文件
    with open('cif_statistics_report.txt', 'w') as f:
        f.write(report)
    print("统计报告已保存到: cif_statistics_report.txt")


if __name__ == "__main__":
    main()
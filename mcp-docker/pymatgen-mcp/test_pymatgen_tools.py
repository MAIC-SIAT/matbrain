"""测试PyMatGen工具函数

简单的测试脚本，验证价态检验工具是否正常工作
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.pymatgen_tools import check_chemical_formula_valence_pymatgen, check_structure_atomic_geometry_pymatgen, analyze_thermodynamic_stability_pymatgen


async def test_valence_tool():
    """测试价态检验工具函数"""

    print("=" * 60)
    print("测试PyMatGen价态检验工具")
    print("=" * 60)

    # 测试用例
    test_formulas = [
        "Fe2O3",      # 应该通过 - 铁的氧化物
        "LiFePO4",    # 应该通过 - 磷酸铁锂
        "CaCO3",      # 应该通过 - 碳酸钙
        "NaCl",       # 应该通过 - 氯化钠
        "H2O",        # 应该通过 - 水
        "CO2",        # 应该通过 - 二氧化碳
        "InvalidFormula123",  # 应该失败 - 无效化学式
        "",           # 应该失败 - 空字符串
        "Xx2O3",      # 应该失败 - 无效元素
        "2Fe2O3",     # 应该失败 - 以数字开头
        "Fe2O3@#",    # 应该失败 - 包含非法字符
        "Fe(OH)3",    # 应该通过 - 带括号的化学式
        "CuSO4·5H2O", # 应该通过 - 水合物
    ]

    for i, formula in enumerate(test_formulas, 1):
        print(f"\n测试 {i}: {formula if formula else '(空字符串)'}")
        print("-" * 40)

        try:
            result = await check_chemical_formula_valence_pymatgen(formula)
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 40)


async def test_atomic_geometry_tool():
    """测试原子几何检查工具函数"""

    print("=" * 60)
    print("测试PyMatGen原子几何检查工具")
    print("=" * 60)

    # 测试用例 - 简单的CIF结构
    test_cif_content = """data_Fe2O3
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.038
_cell_length_b   5.038
_cell_length_c   13.772
_cell_angle_alpha   90.00
_cell_angle_beta    90.00
_cell_angle_gamma   120.00
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Fe2O3
_chemical_formula_sum   'Fe2 O3'
_cell_volume   302.735
_cell_formula_units_Z   2
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 x,y,z
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe Fe1 1 0.0000 0.0000 0.3553 1.0000
Fe Fe2 1 0.0000 0.0000 0.6447 1.0000
O O1 1 0.3061 0.0000 0.2500 1.0000
O O2 1 0.0000 0.3061 0.2500 1.0000
O O3 1 0.6939 0.6939 0.2500 1.0000
"""

    print(f"\n测试 1: Fe2O3 结构几何检查")
    print("-" * 40)

    try:
        result = await check_structure_atomic_geometry_pymatgen(
            cif_content=test_cif_content,
            min_distance_factor=0.5,
            bond_tolerance=0.3,
            max_neighbors_distance=5.0
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("-" * 40)

    # 测试参数验证
    print(f"\n测试 2: 参数验证测试")
    print("-" * 40)

    try:
        result = await check_structure_atomic_geometry_pymatgen(
            cif_content=test_cif_content,
            min_distance_factor=1.5,  # 超出范围
            bond_tolerance=0.3,
            max_neighbors_distance=5.0
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("-" * 40)

    # 测试无效CIF
    print(f"\n测试 3: 无效CIF测试")
    print("-" * 40)

    try:
        result = await check_structure_atomic_geometry_pymatgen(
            cif_content="invalid cif content",
            min_distance_factor=0.5,
            bond_tolerance=0.3,
            max_neighbors_distance=5.0
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("-" * 40)

    # 测试 VNi2S3 结构
    print(f"\n测试 4: VNi2S3 结构几何检查")
    print("-" * 40)

    cif_content = "# generated using pymatgen\ndata_VNi2S3\n_symmetry_space_group_name_H-M   Pm\n_cell_length_a   5.75038413\n_cell_length_b   3.28389322\n_cell_length_c   10.38049984\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   95.13728157\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   6\n_chemical_formula_structural   VNi2S3\n_chemical_formula_sum   'V2 Ni4 S6'\n_cell_volume   195.23428264\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  'x, -y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  V  V0  1  0.89456104  0.50000000  0.23105556  1.0\n  V  V1  1  0.95143218  0.00000000  0.80246413  1.0\n  Ni  Ni2  1  0.28666586  0.50000000  0.73290482  1.0\n  Ni  Ni3  1  0.56059709  0.00000000  0.28648669  1.0\n  Ni  Ni4  1  0.80285662  0.50000000  0.97565718  1.0\n  Ni  Ni5  1  0.94667541  0.00000000  0.42686880  1.0\n  S  S6  1  0.15065652  0.00000000  0.61596373  1.0\n  S  S7  1  0.17093140  0.50000000  0.92935354  1.0\n  S  S8  1  0.17826366  0.00000000  0.26765704  1.0\n  S  S9  1  0.66793308  0.50000000  0.76631495  1.0\n  S  S10  1  0.68271106  0.00000000  0.09461117  1.0\n  S  S11  1  0.68617824  0.50000000  0.41793561  1.0\n"

    try:
        result = await check_structure_atomic_geometry_pymatgen(
            cif_content=cif_content,
            min_distance_factor=0.5,
            bond_tolerance=0.3,
            max_neighbors_distance=5.0
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("-" * 40)


async def test_thermodynamic_stability_tool():
    """测试热力学稳定性分析工具函数"""

    print("=" * 60)
    print("测试PyMatGen热力学稳定性分析工具")
    print("=" * 60)

    # 测试用例 - V4CoS8 结构
    #v4cos8_cif = "# generated using pymatgen\ndata_V4CoS8\n_symmetry_space_group_name_H-M   C2/m\n_cell_length_a   11.05360861\n_cell_length_b   6.65087000\n_cell_length_c   7.84845000\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   133.60805263\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   12\n_chemical_formula_structural   V4CoS8\n_chemical_formula_sum   'V8 Co2 S16'\n_cell_volume   417.78221663\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  '-x, -y, -z'\n  3  '-x, y, -z'\n  4  'x, -y, z'\n  5  'x+1/2, y+1/2, z'\n  6  '-x+1/2, -y+1/2, -z'\n  7  '-x+1/2, y+1/2, -z'\n  8  'x+1/2, -y+1/2, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  V  V0  4  0.00000000  0.22548000  0.00000000  1.0\n  V  V1  4  0.23189000  0.00000000  0.47985000  1.0\n  Co  Co2  2  0.00000000  0.50000000  0.50000000  1.0\n  S  S4  8  0.21062000  0.24994000  0.67738000  1.0\n  S  S5  4  0.01558000  0.50000000  0.81750000  1.0\n  S  S6  4  0.05355000  0.00000000  0.82743000  1.0\n"
    Cs2ErAgBr6_cif="""data_Cs2ErAgBr6_optimized
_symmetry_space_group_name_H-M          'F m -3 m'
_cell_length_a                          11.371346
_cell_length_b                          11.371346
_cell_length_c                          11.371346
_cell_angle_alpha                       90.000000
_cell_angle_beta                        90.000000
_cell_angle_gamma                       90.000000
_symmetry_Int_Tables_number             225
_chemical_formula_structural           Cs2ErAgBr6
_chemical_formula_sum                  'Cs8 Er4 Ag4 Br24'
_cell_volume                            1470.40
_cell_formula_units_Z                   4
loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_symmetry_multiplicity
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
  Cs  Cs0  8  0.250000  0.250000  0.250000  1
  Er  Er1  4  0.500000  0.000000  0.000000  1
  Ag  Ag2  4  0.000000  0.000000  0.000000  1
  Br  Br3  24  0.256175  0.000000  0.000000  1"""

    print(f"\n测试 1: V4CoS8 结构热力学稳定性分析（默认参数）")
    print("-" * 40)

    try:
        result = await asyncio.wait_for(analyze_thermodynamic_stability_pymatgen(
            cif_string=Cs2ErAgBr6_cif,optimize_structure=True
        ),timeout=300)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

    print("-" * 40)


async def run_all_tests():
    """运行所有测试"""
    # await test_valence_tool()
    # await test_atomic_geometry_tool()
    await test_thermodynamic_stability_tool()



if __name__ == "__main__":
    asyncio.run(run_all_tests())

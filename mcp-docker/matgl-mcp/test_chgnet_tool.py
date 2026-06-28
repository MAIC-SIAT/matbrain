#!/usr/bin/env python3
"""
测试CHGNet内部能量预测工具函数

这个脚本用于验证新添加的CHGNet工具函数是否能正常工作。
"""

import asyncio
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from tools.matgl_tools import predict_internal_energy_CHGNet


# 测试用的LiMnO2 CIF结构（与官方示例相同）
TEST_CIF = """data_LiMnO2
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   2.868779
_cell_length_b   4.634475
_cell_length_c   5.832507
_cell_angle_alpha   90.000000
_cell_angle_beta   90.000000
_cell_angle_gamma   90.000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   LiMnO2
_chemical_formula_sum   'Li2 Mn2 O4'
_cell_volume   77.6
_cell_formula_units_Z   2
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1  'x, y, z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Li  Li0  1  0.5  0.5  0.37975  1
Li  Li1  1  0.0  0.0  0.62025  1
Mn  Mn2  1  0.5  0.5  0.863252  1
Mn  Mn3  1  0.0  0.0  0.136747  1
O  O4  1  0.5  0.0  0.360824  1
O  O5  1  0.0  0.5  0.098514  1
O  O6  1  0.5  0.0  0.901486  1
O  O7  1  0.0  0.5  0.639176  1
"""
TEST_CIF1 ="""
data_V4FeS8

_audit_creation_method           'pymatgen'
_symmetry_space_group_name_H-M   'C 2/m'
_symmetry_Int_Tables_number      12
_symmetry_cell_setting           monoclinic

_cell_length_a                   11.15096561
_cell_length_b                   6.64555000
_cell_length_c                   7.87124000
_cell_angle_alpha                90.00000000
_cell_angle_beta                 133.77729191
_cell_angle_gamma                90.00000000
_cell_volume                     421.15746579
_cell_formula_units_Z            2

_chemical_formula_structural     V4FeS8
_chemical_formula_sum            'V8 Fe2 S16'

loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1  'x, y, z'
2  '-x, -y, -z'
3  '-x, y, -z'
4  'x, -y, z'
5  'x+1/2, y+1/2, z'
6  '-x+1/2, -y+1/2, -z'
7  '-x+1/2, y+1/2, -z'
8  'x+1/2, -y+1/2, z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_symmetry_multiplicity
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
V1   V   4  i  0.00000000  0.21620000  0.00000000  1.000
V2   V   4  i  0.23295000  0.00000000  0.48156000  1.000
Fe1  Fe  4  f  0.25000000  0.25000000  0.00000000  0.056
Fe2  Fe  2  d  0.00000000  0.50000000  0.50000000  0.888
S1   S   8  j  0.20785000  0.74837000  0.67501000  1.000
S2   S   4  i  0.01960000  0.50000000  0.81160000  1.000
S3   S   4  i  0.04840000  0.00000000  0.82310000  1.000
"""

TEST_CIF2="""
# generated using pymatgen
data_V2FeS4
_symmetry_space_group_name_H-M   Cm
_cell_length_a   12.44061858
_cell_length_b   3.19114537
_cell_length_c   5.80984939
_cell_angle_alpha   90.00000000
_cell_angle_beta   116.06471614
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   8
_chemical_formula_structural   V2FeS4
_chemical_formula_sum   'V4 Fe2 S8'
_cell_volume   207.19249947
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  'x, -y, z'
  3  'x+1/2, y+1/2, z'
  4  'x+1/2, -y+1/2, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  V  V0  2  0.22715101  0.50000000  0.14441046  1.0
  V  V1  2  0.46687002  0.00000000  0.83157605  1.0
  Fe  Fe2  2  0.48202688  0.50000000  0.46511817  1.0
  S  S3  2  0.08475203  0.00000000  0.16806005  1.0
  S  S4  2  0.11067481  0.50000000  0.68802142  1.0
  S  S5  2  0.33323065  0.50000000  0.59060854  1.0
  S  S6  2  0.36606000  0.00000000  0.13065241  1.0
"""
async def test_chgnet_prediction():
    """测试CHGNet内部能量预测功能"""
    print("=" * 60)
    print("测试CHGNet内部能量预测工具函数")
    print("=" * 60)

    try:
        print("正在使用CHGNet模型预测LiMnO2的内部能量...")
        print("测试参数:")
        print("- 结构: LiMnO2 (与官方示例相同)")
        print("- 优化结构: True")
        print("- 力收敛阈值: 0.01 eV/Å")
        print()


        # 调用CHGNet预测函数
        result = await predict_internal_energy_CHGNet(
            cif_string=TEST_CIF2,
            optimize_structure=True,
            fmax=0.01
        )

        print("预测结果:")
        print("-" * 40)
        print(result)
        print("-" * 40)
        print("✅ CHGNet内部能量预测测试成功!")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


async def test_chgnet_without_optimization():
    """测试不优化结构的CHGNet预测"""
    print("\n" + "=" * 60)
    print("测试CHGNet预测（不优化结构）")
    print("=" * 60)

    try:
        print("正在使用CHGNet模型预测LiMnO2的内部能量（不优化结构）...")
        print("测试参数:")
        print("- 结构: LiMnO2")
        print("- 优化结构: False")
        print()

        # 调用CHGNet预测函数（不优化结构）
        result = await predict_internal_energy_CHGNet(
            cif_string=TEST_CIF2,
            optimize_structure=False
        )

        print("预测结果:")
        print("-" * 40)
        print(result)
        print("-" * 40)
        print("✅ CHGNet预测（不优化结构）测试成功!")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


async def main():
    """主测试函数"""
    print("开始测试CHGNet工具函数...")
    print(f"Python路径: {sys.path[0]}")
    print(f"当前工作目录: {os.getcwd()}")
    print()

    # 检查模型文件是否存在
    model_path = os.getenv("CHGNET_MODEL_PATH", "models/CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES")
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return

    required_files = ["model.json", "model.pt", "state.pt"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            print(f"❌ 模型文件不存在: {file_path}")
            return
        else:
            print(f"✅ 找到模型文件: {file_path}")

    print()

    # 运行测试
    success_count = 0
    total_tests = 2

    if await test_chgnet_prediction():
        success_count += 1

    if await test_chgnet_without_optimization():
        success_count += 1

    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"成功测试数: {success_count}")
    print(f"失败测试数: {total_tests - success_count}")

    if success_count == total_tests:
        print("🎉 所有测试都通过了!")
    else:
        print("⚠️  部分测试失败，请检查错误信息")


if __name__ == "__main__":
    asyncio.run(main())

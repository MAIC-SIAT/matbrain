#!/usr/bin/env python3
"""
测试XRD衍射图谱模拟工具函数
"""

import asyncio
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.pymatgen_tools import simulate_xrd_pattern_pymatgen

# 测试用的CIF数据 - Cs2Er1Ag1Br6双钙钛矿结构
TEST_CIF = """
data_Cs2Er1Ag1Br6
_symmetry_space_group_name_H-M          Fm-3m
_cell_length_a                          11.5404
_cell_length_b                          11.5404
_cell_length_c                          11.5404
_cell_angle_alpha                       90.0000
_cell_angle_beta                        90.0000
_cell_angle_gamma                       90.0000
_symmetry_Int_Tables_number             225
_chemical_formula_structural           Cs2ErAgBr6
_chemical_formula_sum                  'Cs8 Er4 Ag4 Br24'
_cell_volume                            1536.8733
_cell_formula_units_Z                   4
loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_symmetry_multiplicity
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
  Cs  Cs0  8  0.2500  0.2500  0.2500  1
  Er  Er1  4  0.0000  0.0000  0.5000  1
  Ag  Ag2  4  0.0000  0.0000  0.0000  1
  Br  Br3  24  0.0000  0.0000  0.2562  1
"""

async def test_xrd_simulation():
    """测试XRD衍射图谱模拟功能"""
    print("开始测试XRD衍射图谱模拟工具函数...")
    print("=" * 60)

    try:
        # 测试基本功能
        print("测试1: 基本XRD模拟 (默认参数)")
        result1 = await simulate_xrd_pattern_pymatgen(TEST_CIF)
        print("✓ 基本测试通过")
        print(f"结果长度: {len(result1)} 字符")
        print()

        # 测试不同波长
        print("测试2: 使用MoKa射线源")
        result2 = await simulate_xrd_pattern_pymatgen(
            TEST_CIF,
            wavelength="MoKa"
        )
        print("✓ 不同波长测试通过")
        print()

        # 测试自定义角度范围
        print("测试3: 自定义角度范围 (20-60度)")
        result3 = await simulate_xrd_pattern_pymatgen(
            TEST_CIF,
            two_theta_range=(20.0, 60.0)
        )
        print("✓ 自定义角度范围测试通过")
        print()

        # 测试不同强度阈值
        print("测试4: 低强度阈值 (0.1%)")
        result4 = await simulate_xrd_pattern_pymatgen(
            TEST_CIF,
            min_intensity_threshold=0.1
        )
        print("✓ 不同强度阈值测试通过")
        print()

        # 显示第一个测试的完整结果
        print("=" * 60)
        print("测试1的完整输出结果:")
        print("=" * 60)
        print(result1)

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试错误处理功能...")
    print("=" * 60)

    # 测试无效CIF
    print("测试5: 无效CIF数据")
    try:
        result = await simulate_xrd_pattern_pymatgen("invalid cif data")
        if result.startswith("Error:"):
            print("✓ 无效CIF错误处理正确")
        else:
            print("❌ 无效CIF错误处理失败")
    except Exception as e:
        print(f"❌ 无效CIF测试异常: {str(e)}")

    # 测试无效波长
    print("测试6: 无效波长类型")
    try:
        result = await simulate_xrd_pattern_pymatgen(TEST_CIF, wavelength="InvalidWavelength")
        if result.startswith("Error:"):
            print("✓ 无效波长错误处理正确")
        else:
            print("❌ 无效波长错误处理失败")
    except Exception as e:
        print(f"❌ 无效波长测试异常: {str(e)}")

    # 测试无效角度范围
    print("测试7: 无效角度范围")
    try:
        result = await simulate_xrd_pattern_pymatgen(TEST_CIF, two_theta_range=(80.0, 20.0))
        if result.startswith("Error:"):
            print("✓ 无效角度范围错误处理正确")
        else:
            print("❌ 无效角度范围错误处理失败")
    except Exception as e:
        print(f"❌ 无效角度范围测试异常: {str(e)}")

if __name__ == "__main__":
    print("PyMatGen XRD工具函数测试")
    print("=" * 60)

    # 运行测试
    asyncio.run(test_xrd_simulation())
    asyncio.run(test_error_handling())

    print("\n" + "=" * 60)
    print("测试完成!")

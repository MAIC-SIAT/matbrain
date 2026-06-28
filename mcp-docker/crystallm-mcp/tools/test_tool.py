"""
CrystaLLM 工具函数测试

测试 CrystaLLM 晶体结构生成工具的功能
"""

import asyncio
import sys
import os

# 设置本地测试环境变量
def setup_test_environment():
    """设置本地测试环境的配置"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # crystallm-mcp 根目录

    # 设置本地测试路径
    os.environ["CRYSTALLM_TEMP_DIR"] = "/tmp/crystallm_test"
    os.environ["CRYSTALLM_MODEL_DIR"] = os.path.join(project_root, "models")
    os.environ["CRYSTALLM_ROOT"] = os.path.join(project_root, "tools", "CrystaLLM")

    # 设置生成参数（使用较小的值进行快速测试）
    os.environ["CRYSTALLM_NUM_SAMPLES"] = "1"
    os.environ["CRYSTALLM_TEMPERATURE"] = "0.8"
    os.environ["CRYSTALLM_TOP_K"] = "10"
    os.environ["CRYSTALLM_MAX_NEW_TOKENS"] = "2000"  # 减少token数量加快测试
    os.environ["CRYSTALLM_DEVICE"] = "cuda"  # 使用CPU进行测试

    print("测试环境配置:")
    print(f"  临时目录: {os.environ['CRYSTALLM_TEMP_DIR']}")
    print(f"  模型目录: {os.environ['CRYSTALLM_MODEL_DIR']}")
    print(f"  CrystaLLM根目录: {os.environ['CRYSTALLM_ROOT']}")
    print(f"  设备: {os.environ['CRYSTALLM_DEVICE']}")
    print()

# 设置测试环境
setup_test_environment()

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystallm_gen_tool import generate_crystal_structures_crystallm


async def test_crystallm_generation():
    """测试 CrystaLLM 晶体结构生成功能"""
    print("=" * 60)
    print("CrystaLLM 晶体结构生成工具测试")
    print("=" * 60)

    # 测试参数
    test_formula = "CsPbBr3"
    test_space_group = "Pm-3m"
    test_num_samples = 1

    print(f"测试参数:")
    print(f"  化学式: {test_formula}")
    print(f"  空间群: {test_space_group}")
    print(f"  生成数量: {test_num_samples}")
    print("-" * 60)

    try:
        print("开始生成晶体结构...")
        result = await generate_crystal_structures_crystallm(
            formula=test_formula,
            space_group=test_space_group,
            num_samples=test_num_samples
        )

        print("✅ 生成成功!")
        print("-" * 60)
        print("生成结果:")
        print(result)
        print("-" * 60)
        print("测试完成!")

    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()


async def test_formula_validation():
    """测试化学式验证功能"""
    print("\n" + "=" * 60)
    print("化学式验证测试")
    print("=" * 60)

    test_cases = [
        ("CsPbBr3", True),      # 有效化学式
        ("Na2Cl2", True),       # 有效化学式
        ("H2O", True),          # 有效化学式
        ("InvalidFormula", False),  # 无效化学式
        ("123ABC", False),      # 无效化学式
        ("", False),            # 空字符串
    ]

    from core.utils import validate_chemical_formula

    for formula, should_pass in test_cases:
        try:
            validated = validate_chemical_formula(formula)
            if should_pass:
                print(f"✅ '{formula}' -> '{validated}' (验证通过)")
            else:
                print(f"❌ '{formula}' 应该验证失败但通过了")
        except Exception as e:
            if not should_pass:
                print(f"✅ '{formula}' -> 验证失败: {str(e)} (预期行为)")
            else:
                print(f"❌ '{formula}' 应该验证通过但失败了: {str(e)}")


async def main():
    """主测试函数"""
    print("开始 CrystaLLM 工具测试...")

    # 测试化学式验证
   # await test_formula_validation()

    # 测试晶体结构生成
    await test_crystallm_generation()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
测试 verl 版实现的MCP工具  的初始化和调用 是否可行
可行性✅ 20251015
"""

import asyncio
import yaml
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
mars_tool_root = Path(__file__).resolve().parents[1] / "mars_mcp_tools"
sys.path.append(str(mars_tool_root))

from mars_tools import SearchCrystalStructuresFromMaterialsProjectTool
from verl.tools.schemas import OpenAIFunctionToolSchema

def load_config():
    """从 YAML 配置文件加载配置"""
    config_path = os.getenv("MARS_TOOL_CONFIG", str(mars_tool_root / "mars_tool_config.yaml"))

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # 查找 SearchCrystalStructuresFromMaterialsProjectTool 的配置
    for tool_config in config_data['tools']:
        if 'SearchCrystalStructuresFromMaterialsProjectTool' in tool_config['class_name']:
            # 将字典格式的 tool_schema 转换为 OpenAIFunctionToolSchema 对象
            tool_schema_dict = tool_config['tool_schema']
            tool_schema = OpenAIFunctionToolSchema(**tool_schema_dict)
            return tool_config['config'], tool_schema

    raise ValueError("未找到 SearchCrystalStructuresFromMaterialsProjectTool 的配置")


async def test_tool_initialization():
    """测试工具类的初始化"""
    print("=" * 60)
    print("测试 SearchCrystalStructuresFromMaterialsProjectTool 初始化")
    print("=" * 60)

    try:
        # 加载配置
        config, tool_schema = load_config()
        print(f"✓ 成功加载配置: {config}")
        print(f"✓ 成功加载工具模式: {tool_schema}")

        # 初始化工具类
        tool = SearchCrystalStructuresFromMaterialsProjectTool(config, tool_schema)
        print("✓ 成功初始化 SearchCrystalStructuresFromMaterialsProjectTool")

        # 打印工具信息
        print(f"工具名称: {tool.name}")
        print(f"工具描述: {tool.description}")
        print(f"SSE URL: {tool.sse_url}")
        print(f"超时时间: {tool.toolcall_timeout}")

        return tool

    except Exception as e:
        print(f"✗ 初始化失败: {str(e)}")
        return None


async def test_valid_call(tool):
    """测试正常的工具调用"""
    print("\n" + "=" * 60)
    print("测试正常的工具调用")
    print("=" * 60)

    # 正常参数示例
    valid_args = {
        "formula": "Fe2O3",
        "conventional_unit_cell": True,
        "symprec": 0.1
    }

    print(f"调用参数: {valid_args}")

    try:
        result = await tool.execute(valid_args)
        print(f"✓ 调用成功")
        print(f"返回结果: {result}")

    except Exception as e:
        print(f"✗ 调用失败: {str(e)}")


async def test_invalid_call(tool):
    """测试参数不合法的工具调用"""
    print("\n" + "=" * 60)
    print("测试参数不合法的工具调用")
    print("=" * 60)

    # 测试用例1: 缺少必需参数
    print("\n--- 测试用例1: 缺少必需参数 'formula' ---")
    invalid_args1 = {
        "conventional_unit_cell": True,
        "symprec": 0.1
    }

    print(f"调用参数: {invalid_args1}")

    try:
        result = await tool.execute(invalid_args1)
        print(f"返回结果: {result}")

    except Exception as e:
        print(f"异常: {str(e)}")

    # 测试用例2: 无效参数
    print("\n--- 测试用例2: 包含无效参数 ---")
    invalid_args2 = {
        "formula": "Fe2O3",
        "invalid_param": "invalid_value",
        "another_invalid": 123
    }

    print(f"调用参数: {invalid_args2}")

    try:
        result = await tool.execute(invalid_args2)
        print(f"返回结果: {result}")

    except Exception as e:
        print(f"异常: {str(e)}")

    # 测试用例3: 空的必需参数
    print("\n--- 测试用例3: 空的必需参数 ---")
    invalid_args3 = {
        "formula": "",  # 空字符串
        "conventional_unit_cell": True
    }

    print(f"调用参数: {invalid_args3}")

    try:
        result = await tool.execute(invalid_args3)
        print(f"返回结果: {result}")

    except Exception as e:
        print(f"异常: {str(e)}")

    # 测试用例4: None 参数
    print("\n--- 测试用例4: None 参数 ---")

    print(f"调用参数: None")

    try:
        result = await tool.execute(None)
        print(f"返回结果: {result}")

    except Exception as e:
        print(f"异常: {str(e)}")


async def main():
    """主函数"""
    print("开始测试 SearchCrystalStructuresFromMaterialsProjectTool")

    # 初始化工具
    tool = await test_tool_initialization()

    if tool is None:
        print("初始化失败，退出测试")
        return

    # 测试正常调用
    await test_valid_call(tool)

    # 测试参数不合法的调用
    await test_invalid_call(tool)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main())

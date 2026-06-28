"""PyMatGen MCP 配置文件"""

import os
from typing import Optional

class PyMatGenConfig:
    """PyMatGen MCP 配置类"""

    def __init__(self):
        # Materials Project配置
        self.MP_API_KEY: str = os.getenv("MP_API_KEY", "PMASAg256b814q3OaSRWeVc7MKx4mlKI")

        # 热力学分析配置
        self.DEFAULT_ENERGY_THRESHOLD: float = float(os.getenv("ENERGY_THRESHOLD", "0.025"))

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if not self.MP_API_KEY:
            print("警告: 未设置 MP_API_KEY 环境变量")
            return False
        return True

    def print_config(self):
        """打印当前配置信息"""
        print("=" * 50)
        print("PyMatGen MCP 配置:")
        print(f"  MP API Key: {'已设置' if self.MP_API_KEY else '未设置'}")
        print(f"  默认能量阈值: {self.DEFAULT_ENERGY_THRESHOLD} eV/atom")
        print("=" * 50)

# 全局配置实例
pymatgen_config = PyMatGenConfig()

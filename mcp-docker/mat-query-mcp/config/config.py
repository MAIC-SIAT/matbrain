"""Materials Project API 配置文件"""

import os
from typing import Optional

class MPConfig:
    """Materials Project API 配置类"""

    def __init__(self):
        # Materials Project API配置
        self.MP_API_KEY: Optional[str] = os.getenv("MP_API_KEY")

        # 本地数据路径配置
        self.LOCAL_MP_CIF_ROOT: str = os.getenv("LOCAL_MP_CIF_ROOT", "/data/mp_cifs")
        self.LOCAL_MP_PROPS_ROOT: str = os.getenv("LOCAL_MP_PROPS_ROOT", "/data/mp_props")

        # 查询限制配置
        self.MP_TOPK: int = int(os.getenv("MP_TOPK", "5"))

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if not self.MP_API_KEY:
            print("警告: 未设置 MP_API_KEY 环境变量")
            return False

        if not os.path.exists(self.LOCAL_MP_CIF_ROOT):
            print(f"警告: CIF文件路径不存在: {self.LOCAL_MP_CIF_ROOT}")

        if not os.path.exists(self.LOCAL_MP_PROPS_ROOT):
            print(f"警告: 属性文件路径不存在: {self.LOCAL_MP_PROPS_ROOT}")

        return True

    def print_config(self):
        """打印当前配置信息"""
        print("=" * 50)
        print("Materials Project API 配置:")
        print(f"  API Key: {'已设置' if self.MP_API_KEY else '未设置'}")
        print(f"  CIF文件根目录: {self.LOCAL_MP_CIF_ROOT}")
        print(f"  属性文件根目录: {self.LOCAL_MP_PROPS_ROOT}")
        print(f"  查询结果数量限制: {self.MP_TOPK}")
        print("=" * 50)

# 全局配置实例
mp_config = MPConfig()

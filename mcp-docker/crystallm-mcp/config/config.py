"""CrystaLLM 配置文件"""

import os
from typing import Optional

class CrystaLLMConfig:
    """CrystaLLM 配置类"""

    def __init__(self):
        # 临时目录配置
        self.TEMP_DIR: str = os.getenv("CRYSTALLM_TEMP_DIR", "/tmp/")
        self.TEMP_ROOT: str = self.TEMP_DIR  # 为了兼容性保留原有的 TEMP_ROOT 属性

        # 模型目录配置
        self.CRYSTALLM_MODEL_DIR: str = os.getenv("CRYSTALLM_MODEL_DIR", "/app/models")

        # CrystaLLM 根目录配置
        self.CRYSTALLM_ROOT: str = os.getenv("CRYSTALLM_ROOT", "/app/tools/CrystaLLM")




        # 生成参数配置
        self.DEFAULT_NUM_SAMPLES: int = int(os.getenv("CRYSTALLM_NUM_SAMPLES", "2"))
        self.DEFAULT_TEMPERATURE: float = float(os.getenv("CRYSTALLM_TEMPERATURE", "0.8"))
        self.DEFAULT_TOP_K: int = int(os.getenv("CRYSTALLM_TOP_K", "10"))
        self.DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("CRYSTALLM_MAX_NEW_TOKENS", "5000"))
        self.DEFAULT_DEVICE: str = os.getenv("CRYSTALLM_DEVICE", "cuda")

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        valid = True

        # 检查临时目录
        if not os.path.exists(self.TEMP_DIR):
            try:
                os.makedirs(self.TEMP_DIR, exist_ok=True)
                print(f"创建临时目录: {self.TEMP_DIR}")
            except Exception as e:
                print(f"错误: 无法创建临时目录 {self.TEMP_DIR}: {e}")
                valid = False

        # 检查模型目录
        if not os.path.exists(self.CRYSTALLM_MODEL_DIR):
            print(f"警告: 模型目录不存在: {self.CRYSTALLM_MODEL_DIR}")
            valid = False

        # 检查 CrystaLLM 根目录
        if not os.path.exists(self.CRYSTALLM_ROOT):
            print(f"警告: CrystaLLM 根目录不存在: {self.CRYSTALLM_ROOT}")
            valid = False


        return valid

    def print_config(self):
        """打印当前配置信息"""
        print("=" * 50)
        print("CrystaLLM 配置:")
        print(f"  临时目录: {self.TEMP_DIR}")
        print(f"  模型目录: {self.CRYSTALLM_MODEL_DIR}")
        print(f"  CrystaLLM 根目录: {self.CRYSTALLM_ROOT}")
        print(f"  默认采样数量: {self.DEFAULT_NUM_SAMPLES}")
        print(f"  默认温度: {self.DEFAULT_TEMPERATURE}")
        print(f"  默认 Top-K: {self.DEFAULT_TOP_K}")
        print(f"  默认最大新令牌数: {self.DEFAULT_MAX_NEW_TOKENS}")
        print(f"  默认设备: {self.DEFAULT_DEVICE}")
        print("=" * 50)

# 全局配置实例
crystallm_config = CrystaLLMConfig()

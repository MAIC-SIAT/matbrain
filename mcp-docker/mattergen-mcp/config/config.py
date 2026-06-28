"""MatterGen 配置文件"""

import os
from typing import Optional

class MatterGenConfig:
    """MatterGen 配置类"""

    def __init__(self):
        # 获取当前配置文件所在目录的父目录作为基础目录
        self._BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # MatterGen 路径配置
        self.MATTERGEN_ROOT: str = os.getenv(
            "MATTERGEN_ROOT",
            os.path.join(self._BASE_DIR, 'mattergen')
        )
        self.MATTERGENMODEL_ROOT: str = os.getenv(
            "MATTERGENMODEL_ROOT",
            os.path.join(self._BASE_DIR, 'mattergen_ckpt')
        )

        # 临时目录配置
        self.TEMP_DIR: str = os.getenv("MATTERGEN_TEMP_DIR", "/tmp/mattergen")
        self.TEMP_ROOT: str = self.TEMP_DIR  # 为了兼容性保留原有的 TEMP_ROOT 属性

        # GPU 映射配置 - 不同模型使用不同的 GPU
        self.MODEL_TO_GPU = {
            "mattergen_base": "0",
            "dft_mag_density": "1",
            "dft_bulk_modulus": "2",
            "dft_shear_modulus": "3",
            "energy_above_hull": "4",
            "formation_energy_per_atom": "5",
            "space_group": "6",
            "hhi_score": "7",
            "ml_bulk_modulus": "0",
            "chemical_system": "1",
            "dft_band_gap": "2",
            "chemical_system_energy_above_hull": "3",
            "dft_mag_density_hhi_score": "4",
        }

        # 生成参数默认值
        self.DEFAULT_BATCH_SIZE: int = int(os.getenv("MATTERGEN_BATCH_SIZE", "2"))
        self.DEFAULT_NUM_BATCHES: int = int(os.getenv("MATTERGEN_NUM_BATCHES", "1"))
        self.DEFAULT_DIFFUSION_GUIDANCE_FACTOR: float = float(os.getenv("MATTERGEN_GUIDANCE_FACTOR", "2.0"))

        # 设备配置
        self.DEFAULT_DEVICE: str = os.getenv("MATTERGEN_DEVICE", "cuda")

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

        # 检查 MatterGen 根目录
        if not os.path.exists(self.MATTERGEN_ROOT):
            print(f"警告: MatterGen 根目录不存在: {self.MATTERGEN_ROOT}")
            valid = False

        # 检查模型目录
        if not os.path.exists(self.MATTERGENMODEL_ROOT):
            print(f"警告: MatterGen 模型目录不存在: {self.MATTERGENMODEL_ROOT}")
            valid = False

        return valid

    def print_config(self):
        """打印当前配置信息"""
        print("=" * 50)
        print("MatterGen 配置:")
        print(f"  MatterGen 根目录: {self.MATTERGEN_ROOT}")
        print(f"  模型目录: {self.MATTERGENMODEL_ROOT}")
        print(f"  临时目录: {self.TEMP_DIR}")
        print(f"  默认批次大小: {self.DEFAULT_BATCH_SIZE}")
        print(f"  默认批次数量: {self.DEFAULT_NUM_BATCHES}")
        print(f"  默认扩散引导因子: {self.DEFAULT_DIFFUSION_GUIDANCE_FACTOR}")
        print(f"  默认设备: {self.DEFAULT_DEVICE}")
        print(f"  GPU 映射: {len(self.MODEL_TO_GPU)} 个模型")
        print("=" * 50)

# 全局配置实例
mattergen_config = MatterGenConfig()

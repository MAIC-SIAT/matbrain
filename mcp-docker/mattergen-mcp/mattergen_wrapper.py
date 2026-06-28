"""
Simplified wrapper module for MatterGen
"""
import sys
import os
from pathlib import Path
from config.config import mattergen_config

# Add the mattergen directory to the Python path
mattergen_dir = mattergen_config.MATTERGEN_ROOT
if mattergen_dir not in sys.path:
    sys.path.insert(0, mattergen_dir)

# Import the necessary modules from the mattergen package
try:
    from mattergen import generator
    from mattergen.common.data.types import TargetProperty
    from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
    from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME

    # Create convenient aliases
    CrystalGenerator = generator.CrystalGenerator

except ImportError as e:
    print(f"Error importing mattergen modules: {e}")
    print(f"MatterGen path: {mattergen_dir}")
    raise

# Re-export the modules
__all__ = [
    'generator',
    'TargetProperty',
    'MatterGenCheckpointInfo',
    'PRETRAINED_MODEL_NAME',
    'CrystalGenerator'
]

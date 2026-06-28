"""MatterGen MCP 配置模块

提供 MatterGen 和服务器的配置管理。
"""

from .config import mattergen_config
from .server_config import server_config

__all__ = ['mattergen_config', 'server_config']

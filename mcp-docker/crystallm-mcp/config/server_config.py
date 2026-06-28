"""CrystaLLM MCP 服务器配置模块

提供服务器运行所需的配置参数，从环境变量读取配置。
"""

import os
from typing import Optional


class ServerConfig:
    """CrystaLLM MCP 服务器配置类"""

    def __init__(self):
        # 服务器配置
        self.SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
        self.SERVER_PORT: int = int(os.getenv("SERVER_PORT", "5669"))

        # 工具调用超时时间（秒）
        self.TOOL_CALL_TIMEOUT: int = int(os.getenv("TOOL_CALL_TIMEOUT", "60"))

        # 日志级别
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate_config(self) -> bool:
        """
        验证配置的有效性

        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证端口范围
            if not (1 <= self.SERVER_PORT <= 65535):
                print(f"警告: SERVER_PORT {self.SERVER_PORT} 不在有效范围内 (1-65535)")
                return False

            # 验证超时时间
            if self.TOOL_CALL_TIMEOUT <= 0:
                print(f"警告: TOOL_CALL_TIMEOUT {self.TOOL_CALL_TIMEOUT} 必须大于0")
                return False

            # 验证日志级别
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.LOG_LEVEL.upper() not in valid_log_levels:
                print(f"警告: LOG_LEVEL {self.LOG_LEVEL} 不是有效的日志级别")
                print(f"有效的日志级别: {', '.join(valid_log_levels)}")
                return False

            return True

        except Exception as e:
            print(f"配置验证时出错: {e}")
            return False

    def print_config(self):
        """打印当前配置信息"""
        print("=" * 50)
        print("CrystaLLM MCP 服务器配置:")
        print(f"  服务器地址: {self.SERVER_HOST}")
        print(f"  服务器端口: {self.SERVER_PORT}")
        print(f"  工具调用超时: {self.TOOL_CALL_TIMEOUT}秒")
        print(f"  日志级别: {self.LOG_LEVEL}")
        print("=" * 50)


# 创建全局配置实例
server_config = ServerConfig()

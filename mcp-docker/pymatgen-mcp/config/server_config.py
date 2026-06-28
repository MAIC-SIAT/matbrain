"""PyMatGen MCP 服务器配置文件"""

import os
from typing import Optional

class ServerConfig:
    """服务器配置类"""

    def __init__(self):
        # 服务器基础配置
        self.SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
        self.SERVER_PORT: int = int(os.getenv("SERVER_PORT", "5672"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.TOOL_CALL_TIMEOUT: int = int(os.getenv("TOOL_CALL_TIMEOUT", "300"))

        # MatGL-MCP服务配置（合并为单一URL）
        self.MATGL_MCP_SSE_URL: str = os.getenv(
            "MATGL_MCP_SSE_URL",
            "http://matgl-mcp:5668/sse"
        )

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if not self.MATGL_MCP_SSE_URL:
            print("警告: 未设置 MATGL_MCP_SSE_URL")
            return False
        return True

    def print_config(self):
        """打印当前配置信息"""
        print("=" * 50)
        print("PyMatGen MCP 服务器配置:")
        print(f"  服务器地址: {self.SERVER_HOST}:{self.SERVER_PORT}")
        print(f"  日志级别: {self.LOG_LEVEL}")
        print(f"  工具调用超时: {self.TOOL_CALL_TIMEOUT}秒")
        print(f"  MatGL MCP URL: {self.MATGL_MCP_SSE_URL}")
        print("=" * 50)

# 全局配置实例
server_config = ServerConfig()

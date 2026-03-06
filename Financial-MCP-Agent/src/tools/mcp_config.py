"""
MCP服务器配置模块 - 包含连接A股MCP服务器的配置信息
"""

import sys

_PYTHON = sys.executable

SERVER_CONFIGS = {
    "a_share_mcp_v2": {  
        # 使用 `python -m uv` 比直接调用 `uv` 更稳健（避免 PATH 问题）
        "command": _PYTHON,
        "args": [
            "-m",
            "uv",
            "run",
            "--directory",
            r"/Users/xuanxuanzi/Downloads/Finance/a-share-mcp-is-just-i-need",  # 本机 MCP 服务器项目路径
            _PYTHON,
            "mcp_server.py"  # MCP服务器脚本
        ],
        "transport": "stdio",
    }
}
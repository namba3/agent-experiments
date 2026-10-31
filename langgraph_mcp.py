"""Optional Docker MCP adapter for the LangGraph workflow.

This module and its adapter dependency are loaded only when MCP is enabled.
"""

from __future__ import annotations

import os
from typing import Any


async def load_docker_mcp_tools() -> list[Any]:
    """Start Docker MCP over stdio and return its advertised tools."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as error:
        raise RuntimeError(
            "Docker MCP support requires langchain-mcp-adapters."
        ) from error

    client = MultiServerMCPClient(
        {
            "docker": {
                "transport": "stdio",
                "command": os.environ.get("DOCKER_MCP_COMMAND", "docker"),
                "args": ["mcp", "gateway", "run", "--profile", "default"],
            }
        }
    )
    return list(await client.get_tools())

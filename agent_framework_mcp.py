"""Optional Docker MCP adapter for the Agent Framework workflow.

Import this module only when the caller explicitly enables Docker MCP.
"""

from __future__ import annotations

import os
from typing import Any


def create_docker_mcp_tool() -> Any:
    """Create the framework-specific MCP tool only on demand."""
    try:
        from agent_framework import MCPStdioTool
    except ImportError as error:
        raise RuntimeError(
            "Docker MCP support requires a compatible Agent Framework installation."
        ) from error

    return MCPStdioTool(
        name="DockerMcp",
        command=os.environ.get("DOCKER_MCP_COMMAND", "docker"),
        args=["mcp", "gateway", "run", "--profile", "default"],
    )

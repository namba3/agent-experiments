"""Ensure MCP adapters are isolated from ordinary startup paths."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch, sentinel


class McpIsolationTests(unittest.TestCase):
    def test_agent_imports_without_mcp_adapters(self) -> None:
        # A fresh process avoids test-order-dependent sys.modules state.
        script = textwrap.dedent(
            """
            import sys

            sys.modules["agent_framework_mcp"] = None
            sys.modules["langgraph_mcp"] = None
            sys.modules["langchain_mcp_adapters.client"] = None

            import agent_framework_agent
            import langgraph_agent

            assert "agent_framework_mcp" not in vars(agent_framework_agent)
            assert "langgraph_mcp" not in vars(langgraph_agent)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_agent_framework_mcp_tool_is_constructed_on_demand(self) -> None:
        from agent_framework_mcp import create_docker_mcp_tool

        with (
            patch("agent_framework.MCPStdioTool") as factory,
            patch.dict(os.environ, {"DOCKER_MCP_COMMAND": "custom-docker"}),
        ):
            tool = create_docker_mcp_tool()

        self.assertIs(tool, factory.return_value)
        factory.assert_called_once_with(
            name="DockerMcp",
            command="custom-docker",
            args=["mcp", "gateway", "run", "--profile", "default"],
        )

    def test_langgraph_mcp_loads_tools_on_demand(self) -> None:
        from langgraph_mcp import load_docker_mcp_tools

        client = Mock()
        client.get_tools = AsyncMock(return_value=[sentinel.tool])
        factory = Mock(return_value=client)
        stub = types.ModuleType("langchain_mcp_adapters.client")
        stub.MultiServerMCPClient = factory

        with (
            patch.dict(sys.modules, {"langchain_mcp_adapters.client": stub}),
            patch.dict(os.environ, {"DOCKER_MCP_COMMAND": "custom-docker"}),
        ):
            tools = asyncio.run(load_docker_mcp_tools())

        self.assertEqual(tools, [sentinel.tool])
        factory.assert_called_once_with(
            {
                "docker": {
                    "transport": "stdio",
                    "command": "custom-docker",
                    "args": ["mcp", "gateway", "run", "--profile", "default"],
                }
            }
        )
        client.get_tools.assert_awaited_once()

    def test_missing_langgraph_mcp_adapter_reports_actionable_error(self) -> None:
        from langgraph_mcp import load_docker_mcp_tools

        with patch.dict(sys.modules, {"langchain_mcp_adapters.client": None}):
            with self.assertRaisesRegex(
                RuntimeError, "requires langchain-mcp-adapters"
            ):
                asyncio.run(load_docker_mcp_tools())


if __name__ == "__main__":
    unittest.main()

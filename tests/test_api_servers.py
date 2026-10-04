"""HTTP contract tests for both local OpenAI-compatible API servers."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

import agent_framework_agent
import langgraph_agent


class FakeGraph:
    async def ainvoke(self, _graph_input: dict[str, object]) -> dict[str, object]:
        return {"messages": [AIMessage(content="graph answer")]}

    async def astream_events(self, _graph_input: dict[str, object], *, version: str):
        assert version == "v2"
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "answer"},
            "data": {"chunk": AIMessageChunk(content="graph stream")},
        }


def create_langgraph_client(graph: FakeGraph) -> TestClient:
    captured_apps = []

    def capture_app(app, *_args, **_kwargs) -> None:
        captured_apps.append(app)

    with (
        patch.object(
            langgraph_agent,
            "build_graph",
            new=AsyncMock(return_value=graph),
        ),
        patch("uvicorn.run", side_effect=capture_app),
        patch.object(langgraph_agent, "unload_ollama_model"),
    ):
        langgraph_agent.run_server(
            model="test-model",
            without_docker_mcp=True,
            context_limit=12000,
            max_refine_loops=2,
            host="127.0.0.1",
            port=8000,
        )

    return TestClient(captured_apps[0])


def create_agent_framework_client() -> TestClient:
    captured_apps = []

    def capture_app(app, *_args, **_kwargs) -> None:
        captured_apps.append(app)

    with (
        patch("uvicorn.run", side_effect=capture_app),
        patch.object(agent_framework_agent, "unload_ollama_model"),
    ):
        agent_framework_agent.run_server(
            model="test-model",
            without_docker_mcp=True,
            context_limit=12000,
            max_refine_loops=2,
            host="127.0.0.1",
            port=8000,
            disable_research=True,
            disable_complicated=False,
            disable_direct=False,
            temperature=0.7,
            seed=None,
            top_p=None,
            num_predict=None,
        )

    return TestClient(captured_apps[0])


class ApiServerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = FakeGraph()
        self.langgraph_client = create_langgraph_client(self.graph)
        self.agent_framework_client = create_agent_framework_client()
        self.clients = (
            ("LangGraph", self.langgraph_client),
            ("Agent Framework", self.agent_framework_client),
        )

    def tearDown(self) -> None:
        self.langgraph_client.close()
        self.agent_framework_client.close()

    def test_models_endpoint_reports_the_configured_model(self) -> None:
        for name, client in self.clients:
            with self.subTest(server=name):
                response = client.get("/v1/models")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["object"], "list")
                self.assertEqual(response.json()["data"][0]["id"], "test-model")

    def test_empty_messages_are_rejected(self) -> None:
        for name, client in self.clients:
            with self.subTest(server=name):
                response = client.post(
                    "/v1/chat/completions",
                    json={"messages": []},
                )

                self.assertEqual(response.status_code, 400)

    def test_invalid_sampling_options_are_rejected(self) -> None:
        for name, client in self.clients:
            with self.subTest(server=name):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hello"}],
                        "temperature": True,
                    },
                )

                self.assertEqual(response.status_code, 400)

    def test_non_streaming_completion_returns_openai_shape(self) -> None:
        with (
            patch.object(
                agent_framework_agent,
                "prepare_turn",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                agent_framework_agent,
                "run_final_answer",
                new=AsyncMock(return_value=("framework answer", "")),
            ),
        ):
            framework_response = self.agent_framework_client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        graph_response = self.langgraph_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        for name, response, expected_content in (
            ("LangGraph", graph_response, "graph answer"),
            ("Agent Framework", framework_response, "framework answer"),
        ):
            with self.subTest(server=name):
                body = response.json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(body["object"], "chat.completion")
                self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
                self.assertEqual(
                    body["choices"][0]["message"]["content"], expected_content
                )

    def test_streaming_completion_emits_content_and_done_marker(self) -> None:
        async def fake_stream(*_args, **_kwargs):
            yield "framework "
            yield "stream"

        with (
            patch.object(
                agent_framework_agent,
                "prepare_turn",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                agent_framework_agent,
                "stream_final_answer",
                new=fake_stream,
            ),
        ):
            framework_response = self.agent_framework_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )

        graph_response = self.langgraph_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        for name, response, expected_content in (
            ("LangGraph", graph_response, "graph stream"),
            ("Agent Framework", framework_response, "framework stream"),
        ):
            with self.subTest(server=name):
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                chunks = [
                    json.loads(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: {")
                ]
                streamed_content = "".join(
                    choice["delta"].get("content", "")
                    for chunk in chunks
                    for choice in chunk["choices"]
                )
                self.assertEqual(streamed_content, expected_content)
                self.assertIn("data: [DONE]", response.text)


if __name__ == "__main__":
    unittest.main()

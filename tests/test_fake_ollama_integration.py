"""Integration checks against the in-process Ollama HTTP API simulator."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ollama import Client

import agent_framework_agent
import langgraph_agent
from tests.fake_ollama_server import FakeOllamaServer


class FakeOllamaProtocolTests(unittest.TestCase):
    def test_ollama_client_can_list_models_and_read_streamed_chat(self) -> None:
        with FakeOllamaServer(response="simulated response", chunk_size=5) as server:
            client = Client(host=server.base_url)

            models = client.list()
            chunks = list(
                client.chat(
                    model="fake-model",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                )
            )

        self.assertEqual(models.models[0].model, "fake-model")
        self.assertEqual(
            "".join(chunk.message.content or "" for chunk in chunks),
            "simulated response",
        )
        self.assertTrue(chunks[-1].done)


class AgentClientIntegrationTests(unittest.TestCase):
    def test_langgraph_client_uses_configured_fake_ollama_server(self) -> None:
        with FakeOllamaServer(response="Japanese") as server:
            with patch.object(langgraph_agent, "OLLAMA_HOST", server.base_url):
                language = asyncio.run(
                    langgraph_agent.detect_answer_language(
                        "こんにちは", "fake-model", temperature=0
                    )
                )

        self.assertEqual(language, "Japanese")


class AgentWorkflowIntegrationTests(unittest.TestCase):
    @staticmethod
    def responder(request: dict[str, object]) -> str:
        serialized = json.dumps(request)
        if "Determine the response language" in serialized:
            return "English"
        if "Classify the user's request" in serialized:
            return "SIMPLE"
        if "You are the main agent." in serialized:
            return "simulated final response"
        return "unexpected prompt"

    @staticmethod
    def retry_responder(route_label: str):
        draft_instruction = (
            "You are the standard drafting phase"
            if route_label == "STANDARD"
            else "You are the deep-reasoning drafting phase"
        )
        requests: list[dict[str, object]] = []
        counts = {"draft": 0, "verify": 0}

        def respond(request: dict[str, object]) -> str:
            serialized = json.dumps(request)
            requests.append(request)
            if "Determine the response language" in serialized:
                return "English"
            if "Classify the user's request" in serialized:
                return route_label
            if draft_instruction in serialized:
                counts["draft"] += 1
                return f"candidate draft {counts['draft']}"
            if "You are the verification phase. Check" in serialized:
                counts["verify"] += 1
                if counts["verify"] == 1:
                    return "Revise the unsupported date.\nSTATUS: NEEDS_REVISION"
                return "The revised draft is sound.\nSTATUS: OK"
            if "You are the main agent." in serialized:
                return "verified final response"
            return "unexpected prompt"

        return respond, requests, counts

    def assert_retry(
        self,
        requests: list[dict[str, object]],
        counts: dict[str, int],
        route_label: str,
    ) -> None:
        draft_instruction = (
            "You are the standard drafting phase"
            if route_label == "STANDARD"
            else "You are the deep-reasoning drafting phase"
        )
        self.assertEqual(counts, {"draft": 2, "verify": 2})
        sequence = []
        for request_payload in requests:
            request = json.dumps(request_payload)
            if "Determine the response language" in request:
                sequence.append("language")
            elif "Classify the user's request" in request:
                sequence.append("route")
            elif draft_instruction in request:
                sequence.append("draft")
            elif "You are the verification phase. Check" in request:
                sequence.append("verify")
            elif "You are the main agent." in request:
                sequence.append("answer")

        self.assertEqual(
            sequence,
            ["language", "route", "draft", "verify", "draft", "verify", "answer"],
        )
        self.assertIn("Revise the unsupported date.", json.dumps(requests[4]))

    def test_langgraph_simple_workflow_runs_against_fake_ollama(self) -> None:
        with FakeOllamaServer(responder=self.responder) as server:
            with patch.object(langgraph_agent, "OLLAMA_HOST", server.base_url):
                with redirect_stdout(io.StringIO()):
                    asyncio.run(
                        langgraph_agent.main(
                            model="fake-model",
                            without_docker_mcp=True,
                            message="hello",
                        )
                    )

            prompts = [json.dumps(request) for request in server.requests]

        self.assertEqual(len(prompts), 3)
        self.assertTrue(any("Determine the response language" in p for p in prompts))
        self.assertTrue(any("Classify the user's request" in p for p in prompts))
        self.assertTrue(any("You are the main agent." in p for p in prompts))

    def test_agent_framework_simple_workflow_runs_against_fake_ollama(self) -> None:
        args = SimpleNamespace(
            temperature=0.7,
            seed=None,
            top_p=None,
            num_predict=None,
            without_docker_mcp=True,
            context_limit=12000,
            max_refine_loops=2,
            disable_research=True,
            disable_complicated=False,
            disable_direct=False,
        )
        messages = [agent_framework_agent.Message("user", ["hello"])]

        with FakeOllamaServer(responder=self.responder) as server:
            with patch.object(agent_framework_agent, "OLLAMA_HOST", server.base_url):
                result, _summary = asyncio.run(
                    agent_framework_agent.run_cli_turn(
                        model="fake-model",
                        messages=messages,
                        summary="",
                        args=args,
                    )
                )

            prompts = [json.dumps(request) for request in server.requests]

        self.assertEqual(result.answer, "simulated final response")
        self.assertEqual(len(prompts), 3)
        self.assertTrue(any("Determine the response language" in p for p in prompts))
        self.assertTrue(any("Classify the user's request" in p for p in prompts))
        self.assertTrue(any("You are the main agent." in p for p in prompts))

    def test_langgraph_standard_workflow_refines_after_verification(self) -> None:
        responder, requests, counts = self.retry_responder("STANDARD")
        output = io.StringIO()
        with FakeOllamaServer(responder=responder) as server:
            with patch.object(langgraph_agent, "OLLAMA_HOST", server.base_url):
                with redirect_stdout(output):
                    asyncio.run(
                        langgraph_agent.main(
                            model="fake-model",
                            without_docker_mcp=True,
                            message="check this carefully",
                        )
                    )

        self.assertIn("verified final response", output.getvalue())
        self.assert_retry(requests, counts, "STANDARD")

    def test_agent_framework_standard_workflow_refines_after_verification(self) -> None:
        responder, requests, counts = self.retry_responder("STANDARD")
        args = SimpleNamespace(
            temperature=0.7,
            seed=None,
            top_p=None,
            num_predict=None,
            without_docker_mcp=True,
            context_limit=12000,
            max_refine_loops=2,
            disable_research=True,
            disable_complicated=False,
            disable_direct=False,
        )
        messages = [agent_framework_agent.Message("user", ["check this carefully"])]

        with FakeOllamaServer(responder=responder) as server:
            with patch.object(agent_framework_agent, "OLLAMA_HOST", server.base_url):
                result, _summary = asyncio.run(
                    agent_framework_agent.run_cli_turn(
                        model="fake-model",
                        messages=messages,
                        summary="",
                        args=args,
                    )
                )

        self.assertEqual(result.answer, "verified final response")
        self.assert_retry(requests, counts, "STANDARD")

    def test_langgraph_deep_workflow_refines_after_verification(self) -> None:
        responder, requests, counts = self.retry_responder("DEEP")
        output = io.StringIO()
        with FakeOllamaServer(responder=responder) as server:
            with patch.object(langgraph_agent, "OLLAMA_HOST", server.base_url):
                with redirect_stdout(output):
                    asyncio.run(
                        langgraph_agent.main(
                            model="fake-model",
                            without_docker_mcp=True,
                            message="reason through this carefully",
                        )
                    )

        self.assertIn("verified final response", output.getvalue())
        self.assert_retry(requests, counts, "DEEP")

    def test_agent_framework_deep_workflow_refines_after_verification(self) -> None:
        responder, requests, counts = self.retry_responder("DEEP")
        args = SimpleNamespace(
            temperature=0.7,
            seed=None,
            top_p=None,
            num_predict=None,
            without_docker_mcp=True,
            context_limit=12000,
            max_refine_loops=2,
            disable_research=True,
            disable_complicated=False,
            disable_direct=False,
        )
        messages = [agent_framework_agent.Message("user", ["reason carefully"])]

        with FakeOllamaServer(responder=responder) as server:
            with patch.object(agent_framework_agent, "OLLAMA_HOST", server.base_url):
                result, _summary = asyncio.run(
                    agent_framework_agent.run_cli_turn(
                        model="fake-model",
                        messages=messages,
                        summary="",
                        args=args,
                    )
                )

        self.assertEqual(result.answer, "verified final response")
        self.assert_retry(requests, counts, "DEEP")

    def test_agent_framework_client_uses_configured_fake_ollama_server(self) -> None:
        with FakeOllamaServer(response="Japanese") as server:
            with patch.object(agent_framework_agent, "OLLAMA_HOST", server.base_url):
                language = asyncio.run(
                    agent_framework_agent.detect_answer_language(
                        "こんにちは", "fake-model", options={"temperature": 0}
                    )
                )

        self.assertEqual(language, "Japanese")


if __name__ == "__main__":
    unittest.main()

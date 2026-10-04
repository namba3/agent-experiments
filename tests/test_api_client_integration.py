"""HTTP integration tests for the OpenAI-compatible API client."""

from __future__ import annotations

import asyncio
import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from collections.abc import Callable
from typing import Any

import httpx
from openai import AsyncOpenAI

import agent_api_client


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://agent.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def completion_response(content: str = "hello") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class ApiClientHttpIntegrationTests(unittest.TestCase):
    def run_async(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)

    def test_resolve_model_uses_requested_id_without_http_request(self) -> None:
        def unexpected_request(request: httpx.Request) -> httpx.Response:
            self.fail(f"unexpected HTTP request: {request.url}")

        client = make_client(unexpected_request)
        try:
            model = self.run_async(agent_api_client.resolve_model(client, "chosen"))
        finally:
            self.run_async(client.close())

        self.assertEqual(model, "chosen")

    def test_resolve_model_fetches_first_available_model(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "first",
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        },
                        {
                            "id": "second",
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        },
                    ],
                },
            )

        client = make_client(handler)
        try:
            model = self.run_async(agent_api_client.resolve_model(client, None))
        finally:
            self.run_async(client.close())

        self.assertEqual(model, "first")
        self.assertEqual(
            [(request.method, request.url.path) for request in requests],
            [("GET", "/v1/models")],
        )

    def test_resolve_model_rejects_empty_model_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list", "data": []})

        client = make_client(handler)
        try:
            with self.assertRaisesRegex(RuntimeError, "利用可能なモデルがありません"):
                self.run_async(agent_api_client.resolve_model(client, None))
        finally:
            self.run_async(client.close())

    def test_non_stream_completion_sends_options_and_returns_metadata(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=completion_response("answer"))

        client = make_client(handler)
        messages = [{"role": "user", "content": "question"}]
        try:
            result = self.run_async(
                agent_api_client.api_call_with_spinner(
                    client,
                    "test-model",
                    messages,
                    timeout=17,
                    show_spinner=False,
                    temperature=0.2,
                    seed=9,
                    top_p=0.75,
                    max_tokens=42,
                )
            )
        finally:
            self.run_async(client.close())

        self.assertEqual(result.content, "answer")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["total_tokens"], 5)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(
            (request.method, request.url.path), ("POST", "/v1/chat/completions")
        )
        self.assertEqual(float(request.extensions["timeout"]["read"]), 17)
        self.assertEqual(
            json.loads(request.content),
            {
                "model": "test-model",
                "messages": messages,
                "stream": False,
                "temperature": 0.2,
                "seed": 9,
                "top_p": 0.75,
                "max_tokens": 42,
            },
        )

    def test_stream_completion_collects_chunks_and_finish_reason(self) -> None:
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"content": "hel"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"content": "lo"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        body = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )

        client = make_client(handler)
        output = StringIO()
        try:
            with redirect_stdout(output):
                result = self.run_async(
                    agent_api_client.api_call_with_spinner(
                        client,
                        "test-model",
                        [{"role": "user", "content": "question"}],
                        timeout=10,
                        stream=True,
                        show_spinner=False,
                    )
                )
        finally:
            self.run_async(client.close())

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNone(result.usage)
        self.assertIn("hello", output.getvalue())

    def test_send_turn_transmits_image_and_appends_assistant_reply(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=completion_response("a cat"))

        client = make_client(handler)
        messages = [{"role": "system", "content": "be concise"}]
        args = Namespace(
            history_turns=4,
            timeout=12,
            stream=False,
            temperature=None,
            seed=None,
            top_p=None,
            max_tokens=None,
        )
        image = "data:image/png;base64,Y2F0"
        output = StringIO()
        try:
            with redirect_stdout(output):
                answer = self.run_async(
                    agent_api_client.send_turn(
                        client=client,
                        model="test-model",
                        messages=messages,
                        message_text="describe this",
                        image_data_url=image,
                        args=args,
                        json_output=True,
                    )
                )
        finally:
            self.run_async(client.close())

        self.assertEqual(answer, "a cat")
        self.assertEqual(messages[-1], {"role": "assistant", "content": "a cat"})
        body = json.loads(requests[0].content)
        self.assertEqual(
            body["messages"][1]["content"][0], {"type": "text", "text": "describe this"}
        )
        self.assertEqual(body["messages"][1]["content"][1]["image_url"]["url"], image)
        emitted = json.loads(output.getvalue())
        self.assertEqual(
            (emitted["model"], emitted["output"], emitted["finish_reason"]),
            ("test-model", "a cat", "stop"),
        )
        self.assertEqual(emitted["usage"]["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()

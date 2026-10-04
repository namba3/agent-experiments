"""HTTP integration tests for the OpenAI-compatible API client."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

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


class ImageEncodingTests(unittest.TestCase):
    def test_encode_supported_image_as_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image_path.write_bytes(b"hello")

            encoded = asyncio.run(agent_api_client.encode_image(image_path))

        self.assertEqual(encoded, "data:image/png;base64,aGVsbG8=")

    def test_encode_rejects_missing_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "画像ファイルがありません"):
                asyncio.run(agent_api_client.encode_image(directory / "missing.png"))

            unsupported_path = directory / "notes.txt"
            unsupported_path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未対応の画像形式です"):
                asyncio.run(agent_api_client.encode_image(unsupported_path))

    def test_encode_rejects_oversized_file_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "large.png"
            image_path.write_bytes(b"12345")
            with (
                patch.object(agent_api_client, "MAX_IMAGE_BYTES", 4),
                patch.object(Path, "read_bytes") as read_bytes,
            ):
                with self.assertRaisesRegex(ValueError, "20 MiB 以下"):
                    asyncio.run(agent_api_client.encode_image(image_path))

        read_bytes.assert_not_called()

    def test_encode_rechecks_size_after_reading_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "growing.png"
            image_path.write_bytes(b"x")
            with (
                patch.object(agent_api_client, "MAX_IMAGE_BYTES", 4),
                patch.object(Path, "read_bytes", return_value=b"12345"),
            ):
                with self.assertRaisesRegex(ValueError, "20 MiB 以下"):
                    asyncio.run(agent_api_client.encode_image(image_path))


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


class ApiClientCliIntegrationTests(unittest.TestCase):
    def test_main_interactive_mode_sends_turns_and_stops_on_exit_command(self) -> None:
        requests: list[httpx.Request] = []
        clients: list[AsyncOpenAI] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, json=completion_response(f"answer-{len(requests)}")
            )

        def client_factory(**kwargs: Any) -> AsyncOpenAI:
            client = AsyncOpenAI(
                **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            clients.append(client)
            return client

        argv = [
            "agent_api_client.py",
            "--model",
            "test-model",
            "--interactive",
            "--system",
            "be concise",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)),
            patch.object(
                agent_api_client,
                "AsyncOpenAI",
                side_effect=client_factory,
            ),
            patch.object(
                agent_api_client.console,
                "input",
                side_effect=["first question", "second question", "/exit"],
            ),
            patch.object(agent_api_client.console, "print"),
        ):
            exit_code = asyncio.run(agent_api_client.main())

        self.assertEqual(exit_code, 0)
        self.assertTrue(clients[0].is_closed())
        self.assertEqual(len(requests), 2)
        first_turn = json.loads(requests[0].content)["messages"]
        second_turn = json.loads(requests[1].content)["messages"]
        self.assertEqual(
            first_turn,
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "first question"},
            ],
        )
        self.assertEqual(
            second_turn,
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer-1"},
                {"role": "user", "content": "second question"},
            ],
        )

    def test_main_sends_cli_message_and_image_and_prints_json(self) -> None:
        requests: list[httpx.Request] = []
        clients: list[AsyncOpenAI] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=completion_response("a cat"))

        def client_factory(**kwargs: Any) -> AsyncOpenAI:
            self.assertEqual(kwargs["base_url"], "https://agent.test/v1")
            client = AsyncOpenAI(
                **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.png"
            image_path.write_bytes(b"hello")
            argv = [
                "agent_api_client.py",
                "--base-url",
                "https://agent.test",
                "--model",
                "test-model",
                "--message",
                "describe this",
                "--image",
                str(image_path),
                "--temperature",
                "0.25",
                "--json",
            ]
            output = StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    agent_api_client, "AsyncOpenAI", side_effect=client_factory
                ),
                redirect_stdout(output),
            ):
                exit_code = asyncio.run(agent_api_client.main())

        self.assertEqual(exit_code, 0)
        self.assertTrue(clients[0].is_closed())
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/chat/completions")
        request_body = json.loads(requests[0].content)
        self.assertEqual(request_body["temperature"], 0.25)
        image_parts = request_body["messages"][0]["content"]
        self.assertEqual(image_parts[0], {"type": "text", "text": "describe this"})
        self.assertEqual(
            image_parts[1]["image_url"]["url"], "data:image/png;base64,aGVsbG8="
        )
        self.assertEqual(
            json.loads(output.getvalue())["output"],
            "a cat",
        )

    def test_main_returns_error_and_closes_client_on_api_failure(self) -> None:
        clients: list[AsyncOpenAI] = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})

        def client_factory(**kwargs: Any) -> AsyncOpenAI:
            client = AsyncOpenAI(
                **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            clients.append(client)
            return client

        argv = [
            "agent_api_client.py",
            "--model",
            "test-model",
            "--message",
            "question",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(agent_api_client, "AsyncOpenAI", side_effect=client_factory),
        ):
            exit_code = asyncio.run(agent_api_client.main())

        self.assertEqual(exit_code, 1)
        self.assertTrue(clients[0].is_closed())


if __name__ == "__main__":
    unittest.main()

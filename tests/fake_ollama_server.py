"""Small Ollama HTTP API simulator for local integration tests."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Callable


class FakeOllamaServer:
    """Serve deterministic `/api/chat` and `/api/tags` responses."""

    def __init__(
        self,
        *,
        response: str = "Simulated Ollama response.",
        responder: Callable[[dict[str, object]], str] | None = None,
        chunk_size: int = 8,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.response = response
        self.responder = responder
        self.chunk_size = chunk_size
        self.requests: list[dict[str, object]] = []
        self._requests_lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                if self.path != "/api/tags":
                    self.send_error(404)
                    return
                self._send_json(
                    200,
                    {
                        "models": [
                            {
                                "name": "fake-model",
                                "model": "fake-model",
                                "modified_at": "2026-01-01T00:00:00Z",
                                "size": 0,
                                "digest": "fake",
                                "details": {"family": "fake"},
                            }
                        ]
                    },
                )

            def do_POST(self) -> None:
                if self.path != "/api/chat":
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(size))
                    if not isinstance(request, dict):
                        raise ValueError("request body must be a JSON object")
                except (ValueError, json.JSONDecodeError):
                    self._send_json(400, {"error": "invalid JSON request"})
                    return

                with owner._requests_lock:
                    owner.requests.append(request)
                response_text = (
                    owner.responder(request)
                    if owner.responder is not None
                    else owner.response
                )
                if request.get("stream", True):
                    self._send_stream(response_text, request)
                else:
                    self._send_json(
                        200,
                        owner._response_payload(response_text, request, done=True),
                    )

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

            def _send_stream(
                self, response_text: str, request: dict[str, object]
            ) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                for offset in range(0, len(response_text), owner.chunk_size):
                    part = response_text[offset : offset + owner.chunk_size]
                    self._write_chunk(
                        owner._response_payload(part, request, done=False)
                    )
                self._write_chunk(
                    owner._response_payload("", request, done=True), final=True
                )

            def _write_chunk(
                self, payload: dict[str, object], *, final: bool = False
            ) -> None:
                encoded = json.dumps(payload).encode("utf-8") + b"\n"
                self.wfile.write(f"{len(encoded):X}\r\n".encode("ascii"))
                self.wfile.write(encoded + b"\r\n")
                if final:
                    self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fake-ollama-server",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def _response_payload(
        self,
        content: str,
        request: dict[str, object],
        *,
        done: bool,
    ) -> dict[str, object]:
        return {
            "model": str(request.get("model", "fake-model")),
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": done,
            "done_reason": "stop" if done else None,
            "total_duration": 1,
            "load_duration": 0,
            "prompt_eval_count": 1,
            "prompt_eval_duration": 1,
            "eval_count": len(content),
            "eval_duration": 1,
        }

    def start(self) -> FakeOllamaServer:
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()

    def __enter__(self) -> FakeOllamaServer:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fake local Ollama API server")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--response", default="Simulated Ollama response.")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    server = FakeOllamaServer(
        response=args.response,
        chunk_size=args.chunk_size,
        host=args.host,
        port=args.port,
    ).start()
    print(f"Fake Ollama API listening at {server.base_url}")
    print("Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()

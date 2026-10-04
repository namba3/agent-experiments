"""Single-agent LangGraph workflow with research, verification and compaction."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
from datetime import datetime
import gc
import itertools
import json
import math
import os
import re
import shlex
import sys
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import Client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from agent_prompts import (
    answer_prompt,
    draft_prompt,
    language_detection_prompt,
    research_prompt,
    route_prompt,
    summary_prompt,
    verification_prompt,
)

DOCKER_MCP_COMMAND = os.environ.get("DOCKER_MCP_COMMAND", "docker")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DOCKER_MCP_ARGS = ["mcp", "gateway", "run", "--profile", "default"]
DOCKER_MCP_SILENT_COMMAND = "sh"
DEFAULT_CONTEXT_LIMIT = 12000
MESSAGES_TO_KEEP = 6
DEFAULT_MAX_REFINE_LOOPS = 2
MAX_API_IMAGE_BYTES = 20 * 1024 * 1024

class AgentState(MessagesState):
    """Graph state including compacted context and phase notes."""

    summary: str
    research_notes: str
    draft_notes: str
    verification_notes: str
    verification_status: str
    refine_count: int
    answer_language: str
    conversation_started_at: str
    task_route: str
    request_id: str
    generation_options: dict[str, Any]

@asynccontextmanager
async def spinner(message: str):
    """Display a lightweight terminal spinner while an async task runs."""
    stop = asyncio.Event()
    frames = itertools.cycle("|/-\\")

    async def render() -> None:
        while not stop.is_set():
            sys.stderr.write(f"\r{next(frames)} {message}")
            sys.stderr.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.12)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(render())
    try:
        yield
    finally:
        stop.set()
        await task
        sys.stderr.write(f"\r{' ' * (len(message) + 2)}\r\n")
        sys.stderr.flush()

def positive_int(value: str) -> int:
    """argparse 用の 1 以上の整数型。"""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("整数を指定してください") from error
    if number < 1:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください")
    return number

def non_negative_float(value: str) -> float:
    """argparse 用の 0 以上の有限数型。"""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("0 以上の有限数を指定してください") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0 以上の有限数を指定してください")
    return number

def top_p_value(value: str) -> float:
    """argparse 用の 0 から 1 までの有限数型。"""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("0 から 1 の有限数を指定してください") from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("0 から 1 の有限数を指定してください")
    return number

def new_request_id() -> str:
    """Short id used to correlate all node logs for a single request/turn."""
    return uuid.uuid4().hex[:8]


async def invoke_and_close_chat_model(llm: Any, messages: list[Any]) -> Any:
    """Invoke a ChatOllama runnable and close its request-scoped async client."""
    try:
        return await llm.ainvoke(messages)
    finally:
        # ChatOllama does not expose a public close method; a tools binding wraps
        # the model in `.bound`, so walk through that wrapper when needed.
        current = llm
        while current is not None:
            client = getattr(current, "_async_client", None)
            if client is not None:
                await client.close()
                break
            current = getattr(current, "bound", None)


async def detect_answer_language(
    message: str,
    model: str,
    temperature: float = 0.0,
    seed: int | None = None,
    top_p: float | None = None,
    num_predict: int | None = None,
) -> str:
    """Use the LLM to decide whether the answer should be Japanese or English."""
    text = (message or "").strip()
    if not text:
        return "English"

    llm_kwargs: dict[str, Any] = {
        "base_url": OLLAMA_HOST,
        "model": model,
        "reasoning": False,
        "temperature": temperature,
    }
    if seed is not None:
        llm_kwargs["seed"] = seed
    if top_p is not None:
        llm_kwargs["top_p"] = top_p
    if num_predict is not None:
        llm_kwargs["num_predict"] = num_predict

    llm = ChatOllama(**llm_kwargs)
    try:
        response = await invoke_and_close_chat_model(
            llm,
            [
                SystemMessage(content=language_detection_prompt()),
                HumanMessage(content=text),
            ]
        )
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content)
        normalized = content.strip().lower()
        if "japanese" in normalized:
            return "Japanese"
        if "english" in normalized:
            return "English"
    except Exception:
        pass

    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    return "English"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama を使う LangGraph WeatherAgent のテスト"
    )
    parser.add_argument("--model", required=True, help="使用する Ollama モデル名（必須）")
    mcp_group = parser.add_mutually_exclusive_group()
    mcp_group.add_argument(
        "--without-docker-mcp",
        dest="without_docker_mcp",
        action="store_true",
        help="Docker MCP を接続せず、ツールなしで実行する（デフォルト）",
    )
    mcp_group.add_argument(
        "--with-docker-mcp",
        dest="without_docker_mcp",
        action="store_false",
        help="Docker MCP に接続する",
    )
    parser.set_defaults(without_docker_mcp=True)
    parser.add_argument(
        "--message",
        default=None,
        help="エージェントに渡すメッセージ本文。省略時はCUIチャット",
    )
    parser.add_argument(
        "--context-limit",
        type=positive_int,
        default=DEFAULT_CONTEXT_LIMIT,
        help="Context compaction を開始する概算文字数（デフォルト: 12000）",
    )
    parser.add_argument(
        "--max-refine-loops",
        type=positive_int,
        default=DEFAULT_MAX_REFINE_LOOPS,
        help="verify→refine ループの最大反復回数（デフォルト: 2）",
    )
    parser.add_argument(
        "--temperature",
        type=non_negative_float,
        default=0.7,
        help="Ollama の temperature（デフォルト: 0.7）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Ollama の seed（省略時は未指定）",
    )
    parser.add_argument(
        "--top-p",
        type=top_p_value,
        default=None,
        help="Ollama の top_p（省略時は未指定）",
    )
    parser.add_argument(
        "--num-predict",
        type=positive_int,
        default=None,
        help="Ollama の num_predict（省略時は未指定）",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="OpenAI 互換 API サーバーを起動する",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API サーバーの待受ホスト")
    parser.add_argument("--port", type=positive_int, default=8000, help="API サーバーのポート")
    research_group = parser.add_mutually_exclusive_group()
    research_group.add_argument(
        "--enable-research",
        dest="disable_research",
        action="store_false",
        help="Enable the RESEARCH route",
    )
    research_group.add_argument(
        "--disable-research",
        dest="disable_research",
        action="store_true",
        help="Disable the RESEARCH route (default; use STANDARD if detected)",
    )
    parser.set_defaults(disable_research=True)
    parser.add_argument(
        "--disable-deep",
        "--disable-complicated",
        dest="disable_complicated",
        action="store_true",
        help="Disable DEEP route (force to STANDARD if detected)",
    )
    parser.add_argument(
        "--disable-simple",
        "--disable-direct",
        dest="disable_direct",
        action="store_true",
        help="Disable SIMPLE route (force to STANDARD if detected)",
    )
    args = parser.parse_args()
    if not args.disable_research and args.without_docker_mcp:
        parser.error("--enable-research には --with-docker-mcp も必要です")
    return args

def extract_reasoning(message: AIMessage) -> str:
    """Extract reasoning exposed by ChatOllama, across supported formats."""
    additional_kwargs: dict[str, Any] = message.additional_kwargs or {}
    for key in ("reasoning_content", "thinking", "reasoning"):
        value = additional_kwargs.get(key)
        if isinstance(value, str) and value:
            return value
    blocks = getattr(message, "content_blocks", None) or []
    return "".join(
        block.get("reasoning") or block.get("thinking") or block.get("text", "")
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") in {"reasoning", "thinking"}
    )

def unload_ollama_model(model: str) -> None:
    """Unload the selected Ollama model from memory."""
    try:
        Client(host=OLLAMA_HOST).chat(
            model=model,
            messages=[],
            keep_alive=0,
        )
        print(f"Ollama model unloaded: {model}", file=sys.stderr)
    except Exception as error:
        print(f"Could not unload Ollama model {model}: {error}", file=sys.stderr)

def api_content_text(content: Any) -> str:
    """Extract text portions from OpenAI-compatible multimodal content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def api_messages_to_langchain(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert OpenAI text and base64 image messages to LangChain messages."""
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role", "user")
        if not isinstance(role, str):
            raise ValueError("message role must be a string")
        content = message.get("content", "")
        if role in {"system", "developer", "assistant"}:
            if not isinstance(content, str):
                raise ValueError(f"{role} message content must be text")
        elif role == "user" and isinstance(content, list):
            if not content:
                raise ValueError("user content array must not be empty")
            normalized_content: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("user content parts must be objects")
                part_type = part.get("type")
                if part_type == "text" and isinstance(part.get("text"), str):
                    normalized_content.append({"type": "text", "text": part["text"]})
                elif part_type == "image_url":
                    image_value = part.get("image_url")
                    image_url = (
                        image_value.get("url")
                        if isinstance(image_value, dict)
                        else image_value
                    )
                    if not isinstance(image_url, str):
                        raise ValueError("image_url must contain a URL string")
                    header, separator, encoded = image_url.partition(",")
                    allowed_headers = {
                        "data:image/jpeg;base64",
                        "data:image/png;base64",
                        "data:image/gif;base64",
                        "data:image/webp;base64",
                    }
                    if header not in allowed_headers or not separator or not encoded:
                        raise ValueError(
                            "images must use a base64 data URL with JPEG, PNG, GIF, or WebP"
                        )
                    if len(encoded) > ((MAX_API_IMAGE_BYTES + 2) // 3) * 4:
                        raise ValueError("image exceeds the 20 MiB API limit")
                    try:
                        image_bytes = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError) as error:
                        raise ValueError("image data is not valid base64") from error
                    if len(image_bytes) > MAX_API_IMAGE_BYTES:
                        raise ValueError("image exceeds the 20 MiB API limit")
                    normalized_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
                else:
                    raise ValueError("user content supports only text and image_url parts")
            content = normalized_content
        elif role == "user" and not isinstance(content, str):
            raise ValueError("user content must be text or an array of content parts")
        elif role != "user":
            raise ValueError(f"unsupported message role: {role}")

        if role in {"system", "developer"}:
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        elif isinstance(content, list):
            converted.append(HumanMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def api_generation_options(
    request: dict[str, Any],
    *,
    temperature: float,
    seed: int | None,
    top_p: float | None,
    num_predict: int | None,
) -> dict[str, Any]:
    """Validate OpenAI-style sampling options and map them to Ollama names."""
    request_temperature = request.get("temperature", temperature)
    if request_temperature is None:
        request_temperature = temperature
    if isinstance(request_temperature, bool) or not isinstance(
        request_temperature, (int, float)
    ) or not math.isfinite(request_temperature) or request_temperature < 0:
        raise ValueError("temperature must be a non-negative number")

    request_seed = request.get("seed", seed)
    if request_seed is None:
        request_seed = seed
    if request_seed is not None and (
        isinstance(request_seed, bool) or not isinstance(request_seed, int)
    ):
        raise ValueError("seed must be an integer")

    request_top_p = request.get("top_p", top_p)
    if request_top_p is None:
        request_top_p = top_p
    if request_top_p is not None and (
        isinstance(request_top_p, bool)
        or not isinstance(request_top_p, (int, float))
        or not 0 <= request_top_p <= 1
    ):
        raise ValueError("top_p must be a number between 0 and 1")

    request_num_predict = request.get("max_tokens", num_predict)
    if request_num_predict is None:
        request_num_predict = num_predict
    if request_num_predict is not None and (
        isinstance(request_num_predict, bool)
        or not isinstance(request_num_predict, int)
        or request_num_predict < 1
    ):
        raise ValueError("max_tokens must be a positive integer")

    options: dict[str, Any] = {"temperature": request_temperature}
    if request_seed is not None:
        options["seed"] = request_seed
    if request_top_p is not None:
        options["top_p"] = request_top_p
    if request_num_predict is not None:
        options["num_predict"] = request_num_predict
    return options

def run_server(
    model: str,
    without_docker_mcp: bool,
    context_limit: int,
    max_refine_loops: int,
    host: str,
    port: int,
    disable_research: bool = True,
    disable_complicated: bool = False,
    disable_direct: bool = False,
    temperature: float = 0.7,
    seed: int | None = None,
    top_p: float | None = None,
    num_predict: int | None = None,
) -> None:
    """Start an OpenAI-compatible FastAPI server around the graph."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        import uvicorn
    except ImportError as error:
        raise SystemExit(
            "APIサーバーには fastapi と uvicorn が必要です。"
            " `pip install fastapi uvicorn` を実行してください。"
        ) from error

    graph = asyncio.run(
        build_graph(
            model,
            without_docker_mcp,
            context_limit,
            max_refine_loops,
            disable_research,
            disable_complicated,
            disable_direct,
            temperature,
            seed,
            top_p,
            num_predict,
        )
    )
    app = FastAPI(title="LangGraph OpenAI-Compatible API")

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "owned_by": "ollama",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, Any]):
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        request_id = new_request_id()
        request_started_at = time.perf_counter()
        print(f"[{request_id}] [server] chat_completions request received", flush=True)

        try:
            generation_options = api_generation_options(
                request,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                num_predict=num_predict,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        try:
            langchain_messages = api_messages_to_langchain(messages)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        latest_user_message = next(
            (
                api_content_text(item.get("content", ""))
                for item in reversed(messages)
                if item.get("role") == "user"
            ),
            "",
        )

        answer_language = await detect_answer_language(
            latest_user_message,
            model,
            temperature=generation_options["temperature"],
            seed=generation_options.get("seed"),
            top_p=generation_options.get("top_p"),
            num_predict=generation_options.get("num_predict"),
        )
        graph_input = {
            "messages": langchain_messages,
            "answer_language": answer_language,
            "conversation_started_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "request_id": request_id,
            "generation_options": generation_options,
        }
        completion_id = f"chatcmpl-{request_id}"
        created = int(datetime.now().timestamp())

        if request.get("stream"):
            async def stream_response():
                role = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(role, ensure_ascii=False)}\n\n"
                sent_text = False
                try:
                    async for event in graph.astream_events(graph_input, version="v2"):
                        if event.get("metadata", {}).get("langgraph_node") != "answer":
                            continue
                        event_data = event.get("data", {})
                        content = ""
                        if event.get("event") == "on_chat_model_stream":
                            chunk = event_data.get("chunk")
                            content = getattr(chunk, "content", "")
                        elif event.get("event") == "on_chain_end" and not sent_text:
                            output = event_data.get("output")
                            if isinstance(output, dict):
                                output_messages = output.get("messages", [])
                                if output_messages:
                                    content = getattr(output_messages[-1], "content", "")
                        if not isinstance(content, str) or not content:
                            continue
                        sent_text = True
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except Exception as error:
                    print(f"[{request_id}] [server] stream failed: {error}", flush=True)
                    payload = {
                        "error": {
                            "message": "The response stream failed.",
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                elapsed = time.perf_counter() - request_started_at
                print(f"[{request_id}] [server] answer completed in {elapsed:.3f}s", flush=True)
                finish = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
                yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await graph.ainvoke(graph_input)
        elapsed = time.perf_counter() - request_started_at
        print(f"[{request_id}] [server] answer completed in {elapsed:.3f}s", flush=True)
        content = result["messages"][-1].content
        if not isinstance(content, str):
            content = str(content)
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(response)

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        unload_ollama_model(model)

async def build_graph(
    model: str,
    without_docker_mcp: bool = True,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    max_refine_loops: int = DEFAULT_MAX_REFINE_LOOPS,
    disable_research: bool = True,
    disable_complicated: bool = False,
    disable_direct: bool = False,
    temperature: float = 0.7,
    seed: int | None = None,
    top_p: float | None = None,
    num_predict: int | None = None,
):
    """Build a single-agent graph with research, verification, refinement and
    compaction."""
    tools: list[Any] = []
    if not without_docker_mcp:
        client = MultiServerMCPClient(
            {
                "docker": {
                    "transport": "stdio",
                    # MCP 通信の stdout は維持し、gateway の stderr だけ抑制する。
                    "command": DOCKER_MCP_SILENT_COMMAND,
                    "args": [
                        "-c",
                        f"exec {shlex.join([DOCKER_MCP_COMMAND, *DOCKER_MCP_ARGS])} "
                        "2>/dev/null",
                    ],
                }
            }
        )
        tools.extend(await client.get_tools())
    if not disable_research and not tools:
        raise ValueError(
            "The RESEARCH route requires available Docker MCP tools; "
            "enable Docker MCP and check that the gateway provides tools."
        )

    llm_kwargs: dict[str, Any] = {
        "base_url": OLLAMA_HOST,
        "model": model,
        "temperature": temperature,
    }
    if seed is not None:
        llm_kwargs["seed"] = seed
    if top_p is not None:
        llm_kwargs["top_p"] = top_p
    if num_predict is not None:
        llm_kwargs["num_predict"] = num_predict

    tool_node = ToolNode(tools, handle_tool_errors=True) if tools else None

    def llm_for_state(
        state: AgentState, *, reasoning: bool, bind_tools: bool = False
    ):
        request_options = state.get("generation_options", {})
        options = dict(llm_kwargs)
        options.update(
            {
                key: request_options[key]
                for key in ("temperature", "seed", "top_p", "num_predict")
                if key in request_options
            }
        )
        llm = ChatOllama(reasoning=reasoning, **options)
        return llm.bind_tools(tools) if bind_tools and tools else llm

    def message_text(message: Any) -> str:
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else str(content)

    def log_line(state: AgentState, node: str, message: str) -> None:
        """Print a log line tagged with the request id and node name, so all
        node logs for a single request/turn can be grepped/correlated
        together across concurrent requests."""
        request_id = state.get("request_id") or "-"
        print(f"[{request_id}] [{node}] {message}", flush=True)

    def log_elapsed(state: AgentState, label: str, started_at: float) -> None:
        """Log total elapsed time for a named phase."""
        elapsed = time.perf_counter() - started_at
        log_line(state, label, f"completed in {elapsed:.3f}s")

    def context_size(state: AgentState) -> int:
        return (
            len(state.get("summary", ""))
            + len(state.get("research_notes", ""))
            + len(state.get("draft_notes", ""))
            + len(state.get("verification_notes", ""))
            + sum(len(message_text(message)) for message in state["messages"])
        )

    async def compact_context(state: AgentState) -> dict[str, Any]:
        """Summarize old messages when the approximate context gets too large."""
        messages = state["messages"]
        size = context_size(state)
        if size <= context_limit or len(messages) <= MESSAGES_TO_KEEP:
            log_line(
                state,
                "compact",
                f"skipped (size={size}/{context_limit}, messages={len(messages)})",
            )
            return {}

        log_line(
            state,
            "compact",
            f"compacting (size={size}/{context_limit}, messages={len(messages)})",
        )
        old_messages = messages[:-MESSAGES_TO_KEEP]
        transcript = "\n".join(
            f"{getattr(message, 'type', 'message')}: {message_text(message)}"
            for message in old_messages
        )
        prompt = summary_prompt(
            state.get("summary", ""),
            research_notes=state.get("research_notes", ""),
            draft_notes=state.get("draft_notes", ""),
            verification_notes=state.get("verification_notes", ""),
            transcript=transcript,
        )
        summary_message = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=True),
            [HumanMessage(content=prompt)],
        )
        removed = [
            RemoveMessage(id=message.id)
            for message in old_messages
            if getattr(message, "id", None)
        ]
        log_line(
            state,
            "compact",
            f"done (removed={len(removed)}, summary_len={len(message_text(summary_message))})",
        )
        return {
            "summary": message_text(summary_message),
            "messages": removed,
        }

    def context_messages(state: AgentState) -> list[Any]:
        context: list[Any] = []
        if state.get("summary"):
            context.append(
                SystemMessage(
                    content=(
                        "Conversation summary from earlier context. Use it as "
                        "background, while prioritizing the latest messages:\n"
                        f"{state['summary']}"
                    )
                )
            )
        return [*context, *state["messages"]]

    async def research(state: AgentState) -> dict[str, Any]:
        refine_count = state.get("refine_count", 0)
        log_line(state, "research", f"invoked (refine_count={refine_count})")
        instruction = research_prompt(
            started_at=state.get("conversation_started_at", "unknown"),
            refine_count=(refine_count if state.get("verification_status") != "OK" else 0),
            max_refine_loops=max_refine_loops,
            verification_notes=state.get("verification_notes", ""),
        )
        response = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=True, bind_tools=True),
            [SystemMessage(content=instruction), *context_messages(state)]
        )
        tool_call_count = len(getattr(response, "tool_calls", None) or [])
        log_line(
            state,
            "research",
            f"done (notes_len={len(message_text(response))}, tool_calls={tool_call_count})",
        )
        return {"messages": [response], "research_notes": message_text(response)}

    async def route_task(state: AgentState) -> dict[str, str]:
        """Decide whether this request needs research, careful drafting with
        verification, a standard draft, or a simple answer."""
        log_line(state, "route", "invoked")
        response = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=False),
            [
                SystemMessage(
                    content=(
                        route_prompt(
                            state.get("conversation_started_at", "unknown")
                        )
                    )
                ),
                *context_messages(state),
            ]
        )
        route = message_text(response).strip().upper()
        if "RESEARCH" in route and not disable_research:
            task_route = "RESEARCH"
        elif "DEEP" in route and not disable_complicated:
            task_route = "DEEP"
        elif "SIMPLE" in route and not disable_direct:
            task_route = "SIMPLE"
        else:
            task_route = "STANDARD"
        log_line(state, "route", f"task_route={task_route!r}")
        return {"task_route": task_route}

    async def call_tools(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []
        for tool_call in tool_calls:
            tool_args = tool_call.get("args", {})
            arg_count = len(tool_args) if isinstance(tool_args, dict) else 0
            log_line(
                state,
                "tools",
                f"call name={tool_call['name']} arg_count={arg_count}",
            )
        assert tool_node is not None
        result = await tool_node.ainvoke(state)
        log_line(state, "tools", f"done ({len(tool_calls)} call(s))")
        return result

    def route_research(state: AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "verify"

    def route_answer(state: AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    async def draft(state: AgentState) -> dict[str, Any]:
        """Produce a careful candidate answer for DEEP tasks that need
        no external investigation, ahead of the verify-refine loop."""
        refine_count = state.get("refine_count", 0)
        log_line(state, "draft", f"invoked (refine_count={refine_count})")
        instruction = draft_prompt(
            started_at=state.get("conversation_started_at", "unknown"),
            standard=False,
            refine_count=(refine_count if state.get("verification_status") != "OK" else 0),
            max_refine_loops=max_refine_loops,
            verification_notes=state.get("verification_notes", ""),
        )
        response = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=True),
            [SystemMessage(content=instruction), *context_messages(state)]
        )
        log_line(state, "draft", f"done (draft_len={len(message_text(response))})")
        return {"draft_notes": message_text(response)}

    async def standard_draft(state: AgentState) -> dict[str, Any]:
        """Produce a standard candidate answer with verification and no reasoning.
        Used for tasks between DEEP and SIMPLE in complexity."""
        refine_count = state.get("refine_count", 0)
        log_line(state, "standard_draft", f"invoked (refine_count={refine_count})")
        instruction = draft_prompt(
            started_at=state.get("conversation_started_at", "unknown"),
            standard=True,
            refine_count=(refine_count if state.get("verification_status") != "OK" else 0),
            max_refine_loops=max_refine_loops,
            verification_notes=state.get("verification_notes", ""),
        )
        response = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=False),
            [SystemMessage(content=instruction), *context_messages(state)]
        )
        log_line(state, "standard_draft", f"done (draft_len={len(message_text(response))})")
        return {"draft_notes": message_text(response)}

    def route_after_classification(state: AgentState) -> str:
        task_route = state.get("task_route")
        if task_route == "RESEARCH":
            return "research"
        if task_route == "DEEP":
            return "draft"
        if task_route == "STANDARD":
            return "standard_draft"
        return "answer"

    def route_after_verify(state: AgentState) -> str:
        """Loop back to the appropriate drafting node when verification found
        unresolved issues, up to max_refine_loops iterations; otherwise
        proceed to answer."""
        status = state.get("verification_status", "OK")
        refine_count = state.get("refine_count", 0)
        if status in {"NEEDS_REVISION", "INVALID"} and refine_count <= max_refine_loops:
            task_route = state.get("task_route")
            if task_route == "RESEARCH":
                return "refine_research"
            elif task_route == "STANDARD":
                return "refine_standard"
            else:
                return "refine_draft"
        return "answer"

    async def verify(state: AgentState) -> dict[str, Any]:
        refine_count = state.get("refine_count", 0)
        is_research = state.get("task_route") == "RESEARCH"
        is_standard = state.get("task_route") == "STANDARD"
        material_label = "research findings" if is_research else "draft answer"
        material = state.get("research_notes" if is_research else "draft_notes", "")
        log_line(state, "verify", f"invoked (checking {material_label})")
        response = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=not is_standard),
            [
                SystemMessage(
                    content=verification_prompt(
                        started_at=state.get("conversation_started_at", "unknown"),
                        material_label=material_label,
                        material=material,
                    )
                ),
                *context_messages(state),
            ]
        )
        raw_notes = message_text(response)
        status_match = re.search(r"STATUS:\s*(OK|NEEDS_REVISION)\s*$", raw_notes.strip())
        if status_match:
            status = status_match.group(1)
            notes = raw_notes[: status_match.start()].strip()
        else:
            status = "INVALID"
            notes = (
                raw_notes.strip()
                + "\nVerifier output did not end with a valid STATUS line; "
                "this result is not considered verified."
            ).strip()

        # Unverified or malformed verifier output consumes the same bounded
        # retry budget as a requested revision.
        new_refine_count = (
            refine_count + 1 if status in {"NEEDS_REVISION", "INVALID"} else refine_count
        )
        log_line(
            state,
            "verify",
            f"status={status} refine_count={new_refine_count}/{max_refine_loops}",
        )
        return {
            "verification_notes": notes,
            "verification_status": status,
            "refine_count": new_refine_count,
        }

    async def answer(state: AgentState) -> dict[str, list[AIMessage]]:
        answer_language = state.get("answer_language", "English")
        answer_started_at = time.perf_counter()
        log_line(state, "answer", f"invoked (language={answer_language})")
        instruction = answer_prompt(
            answer_language=answer_language,
            started_at=state.get("conversation_started_at", "unknown"),
            research_notes=state.get("research_notes", ""),
            draft_notes=state.get("draft_notes", ""),
            verification_notes=state.get("verification_notes", ""),
            verification_status=state.get("verification_status", "OK"),
        )
        response = await invoke_and_close_chat_model(
            llm_for_state(state, reasoning=True, bind_tools=True),
            [SystemMessage(content=instruction), *context_messages(state)]
        )
        if not message_text(response).strip():
            # 一部の reasoning 対応モデルが空の本文を返した場合の保険。
            fallback = (
                state.get("verification_notes")
                or state.get("draft_notes")
                or state.get("research_notes")
            )
            if fallback:
                log_line(state, "answer", "empty response, using fallback notes")
                response = AIMessage(content=fallback)
        tool_call_count = len(getattr(response, "tool_calls", None) or [])
        log_line(
            state,
            "answer",
            f"done (answer_len={len(message_text(response))}, tool_calls={tool_call_count}, elapsed={time.perf_counter() - answer_started_at:.3f}s)",
        )
        gc.collect()
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("compact", compact_context)
    builder.add_node("route", route_task)
    builder.add_node("research", research)
    builder.add_node("draft", draft)
    builder.add_node("standard_draft", standard_draft)
    builder.add_node("verify", verify)
    builder.add_node("answer", answer)
    builder.add_edge(START, "compact")
    builder.add_edge("compact", "route")
    builder.add_conditional_edges(
        "route",
        route_after_classification,
        {"research": "research", "draft": "draft", "standard_draft": "standard_draft", "answer": "answer"},
    )
    # draft and standard_draft never use tools (no external investigation),
    # so they always go straight to verification regardless of tool availability.
    builder.add_edge("draft", "verify")
    builder.add_edge("standard_draft", "verify")
    if tools:
        builder.add_node("research_tools", call_tools)
        builder.add_node("answer_tools", call_tools)
        builder.add_conditional_edges(
            "research",
            route_research,
            {"tools": "research_tools", "verify": "verify"},
        )
        builder.add_edge("research_tools", "compact")
        builder.add_conditional_edges(
            "answer", route_answer, {"tools": "answer_tools", "end": END}
        )
        builder.add_edge("answer_tools", "answer")
    else:
        builder.add_edge("research", "verify")
        builder.add_edge("answer", END)
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"refine_research": "research", "refine_draft": "draft", "refine_standard": "standard_draft", "answer": "answer"},
    )
    return builder.compile()

async def main(
    model: str,
    without_docker_mcp: bool = True,
    message: str | None = None,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    max_refine_loops: int = DEFAULT_MAX_REFINE_LOOPS,
    disable_research: bool = True,
    disable_complicated: bool = False,
    disable_direct: bool = False,
    temperature: float = 0.7,
    seed: int | None = None,
    top_p: float | None = None,
    num_predict: int | None = None,
) -> None:
    graph = await build_graph(
        model,
        without_docker_mcp,
        context_limit,
        max_refine_loops,
        disable_research,
        disable_complicated,
        disable_direct,
        temperature,
        seed,
        top_p,
        num_predict,
    )
    conversation_started_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    if message is not None:
        messages = [HumanMessage(content=message)]
        request_id = new_request_id()
        request_started_at = time.perf_counter()
        answer_language = await detect_answer_language(
            message,
            model,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            num_predict=num_predict,
        )
        async with spinner("メインエージェントが処理中"):
            result = await graph.ainvoke(
                {
                    "messages": messages,
                    "answer_language": answer_language,
                    "conversation_started_at": conversation_started_at,
                    "request_id": request_id,
                }
            )

        elapsed = time.perf_counter() - request_started_at
        print(f"[{request_id}] [cli] answer completed in {elapsed:.3f}s")
        print_answer(result["messages"][-1])
        return

    print("CUI チャットを開始しました。終了するには /exit または /quit を入力してください。")
    state: dict[str, Any] = {
        "messages": [],
        "conversation_started_at": conversation_started_at,
    }
    while True:
        try:
            user_message = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print("\nチャットを終了します。")
            break
        if user_message.strip().lower() in {"/exit", "/quit"}:
            break
        if not user_message.strip():
            continue

        state["messages"] = [
            *state.get("messages", []),
            HumanMessage(content=user_message),
        ]
        state["answer_language"] = await detect_answer_language(
            user_message,
            model,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            num_predict=num_predict,
        )
        # Each new user turn gets a fresh verify-refine budget and clean
        # scratch fields; these are per-turn artifacts, not conversation state.
        state["refine_count"] = 0
        state["verification_status"] = "OK"
        state["research_notes"] = ""
        state["draft_notes"] = ""
        state["verification_notes"] = ""
        state["request_id"] = new_request_id()
        request_started_at = time.perf_counter()
        async with spinner("メインエージェントが処理中"):
            state = await graph.ainvoke(state)

        elapsed = time.perf_counter() - request_started_at
        print(f"[{state['request_id']}] [cli] answer completed in {elapsed:.3f}s")
        print_answer(state["messages"][-1])

def print_answer(last_message: Any) -> None:
    reasoning = extract_reasoning(last_message) if isinstance(last_message, AIMessage) else ""
    print(f"Reasoning: {reasoning}")
    print("==================================")
    print(last_message.content)

if __name__ == "__main__":
    args = parse_arguments()
    if args.serve:
        run_server(
            args.model,
            args.without_docker_mcp,
            args.context_limit,
            args.max_refine_loops,
            args.host,
            args.port,
            args.disable_research,
            args.disable_complicated,
            args.disable_direct,
            args.temperature,
            args.seed,
            args.top_p,
            args.num_predict,
        )
    else:
        try:
            asyncio.run(
                main(
                    args.model,
                    args.without_docker_mcp,
                    args.message,
                    args.context_limit,
                    args.max_refine_loops,
                    args.disable_research,
                    args.disable_complicated,
                    args.disable_direct,
                    args.temperature,
                    args.seed,
                    args.top_p,
                    args.num_predict,
                )
            )
        except KeyboardInterrupt:
            print("\n処理を中断しました。")
        finally:
            unload_ollama_model(args.model)

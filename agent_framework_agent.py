"""Agent Framework implementation of the local Ollama agent workflow."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import math
import os
import re
import shlex
import sys
import time
import uuid
from typing import Any, AsyncIterator

from agent_framework import Agent, Content, MCPStdioTool, Message
from agent_framework.ollama import OllamaChatClient
from ollama import Client

DOCKER_MCP_COMMAND = os.environ.get("DOCKER_MCP_COMMAND", "docker")
DOCKER_MCP_ARGS = ["mcp", "gateway", "run", "--profile", "default"]
DEFAULT_CONTEXT_LIMIT = 12000
MESSAGES_TO_KEEP = 6
DEFAULT_MAX_REFINE_LOOPS = 2
MAX_API_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

@asynccontextmanager
async def spinner(message: str):
    """Display a small asynchronous terminal spinner."""
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
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("整数を指定してください") from error
    if number < 1:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください")
    return number


def non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("0 以上の有限数を指定してください") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0 以上の有限数を指定してください")
    return number


def top_p_value(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("0 から 1 の有限数を指定してください") from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("0 から 1 の有限数を指定してください")
    return number


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent Framework と Ollama を使うエージェント"
    )
    parser.add_argument("--model", required=True, help="使用する Ollama モデル名")
    mcp_group = parser.add_mutually_exclusive_group()
    mcp_group.add_argument(
        "--without-docker-mcp",
        dest="without_docker_mcp",
        action="store_true",
        help="Docker MCP を使わない（デフォルト）",
    )
    mcp_group.add_argument(
        "--with-docker-mcp",
        dest="without_docker_mcp",
        action="store_false",
        help="Docker MCP gateway のツールを使う",
    )
    parser.set_defaults(without_docker_mcp=True)
    parser.add_argument("--message", help="1回だけ送信するメッセージ。省略時は対話モード")
    parser.add_argument(
        "--context-limit",
        type=positive_int,
        default=DEFAULT_CONTEXT_LIMIT,
        help=f"会話圧縮を始める概算文字数（デフォルト: {DEFAULT_CONTEXT_LIMIT}）",
    )
    parser.add_argument(
        "--max-refine-loops",
        type=positive_int,
        default=DEFAULT_MAX_REFINE_LOOPS,
        help=f"検証後の最大再試行回数（デフォルト: {DEFAULT_MAX_REFINE_LOOPS}）",
    )
    parser.add_argument(
        "--temperature", type=non_negative_float, default=0.7, help="Ollama temperature"
    )
    parser.add_argument("--seed", type=int, default=None, help="Ollama seed")
    parser.add_argument("--top-p", type=top_p_value, default=None, help="Ollama top_p")
    parser.add_argument(
        "--num-predict", type=positive_int, default=None, help="最大生成トークン数"
    )
    parser.add_argument("--serve", action="store_true", help="OpenAI 互換 API サーバーを起動")
    parser.add_argument("--host", default="127.0.0.1", help="API サーバーの待受ホスト")
    parser.add_argument("--port", type=positive_int, default=8000, help="API サーバーのポート")
    research_group = parser.add_mutually_exclusive_group()
    research_group.add_argument(
        "--enable-research",
        dest="disable_research",
        action="store_false",
        help="RESEARCH 経路を有効にする（Docker MCP ツールが必要）",
    )
    research_group.add_argument(
        "--disable-research",
        dest="disable_research",
        action="store_true",
        help="RESEARCH 経路を無効にする（デフォルト）",
    )
    parser.set_defaults(disable_research=True)
    parser.add_argument("--disable-complicated", action="store_true")
    parser.add_argument("--disable-direct", action="store_true")
    args = parser.parse_args()
    if not args.disable_research and args.without_docker_mcp:
        parser.error("--enable-research には --with-docker-mcp も必要です")
    return args


@dataclass
class TurnPlan:
    request_id: str
    history: list[Message]
    summary: str
    answer_language: str
    answer_instructions: str
    fallback_answer: str
    generation_options: dict[str, Any]
    use_tools: bool


@dataclass
class TurnResult:
    plan: TurnPlan
    answer: str
    reasoning: str


def message_text(message: Message) -> str:
    return message.text


def response_text(response: Any) -> str:
    for message in reversed(getattr(response, "messages", []) or []):
        if getattr(message, "role", None) == "assistant" and message.text.strip():
            return message.text
    return getattr(response, "text", "") or ""


def response_reasoning(response: Any) -> str:
    parts: list[str] = []
    for message in getattr(response, "messages", []) or []:
        if getattr(message, "role", None) != "assistant":
            continue
        parts.extend(
            content.text or ""
            for content in message.contents
            if content.type == "text_reasoning" and content.text
        )
    return "".join(parts)


def log_line(request_id: str, node: str, message: str) -> None:
    print(f"[{request_id}] [{node}] {message}", flush=True)


def generation_options(
    *,
    temperature: float,
    seed: int | None,
    top_p: float | None,
    num_predict: int | None,
    reasoning: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": temperature, "think": reasoning}
    if seed is not None:
        options["seed"] = seed
    if top_p is not None:
        options["top_p"] = top_p
    if num_predict is not None:
        options["num_predict"] = num_predict
    return options


def mcp_tool() -> MCPStdioTool:
    return MCPStdioTool(
        name="DockerMcp",
        command=DOCKER_MCP_COMMAND,
        args=list(DOCKER_MCP_ARGS),
    )


async def run_agent(
    *,
    model: str,
    instructions: str,
    messages: list[Message],
    options: dict[str, Any],
    use_tools: bool,
    require_tools: bool = False,
) -> Any:
    client = OllamaChatClient(host="http://localhost:11434", model=model)
    tools = [mcp_tool()] if use_tools else None
    agent = Agent(
        client=client,
        instructions=instructions,
        name="LocalAssistant",
        tools=tools,
    )
    async with agent:
        if require_tools and not any(getattr(tool, "_functions", None) for tool in tools or []):
            raise ValueError(
                "Docker MCP is enabled but the gateway did not provide any tools."
            )
        return await agent.run(messages=messages, options=options)


async def stream_agent(
    *,
    model: str,
    instructions: str,
    messages: list[Message],
    options: dict[str, Any],
    use_tools: bool,
) -> AsyncIterator[str]:
    client = OllamaChatClient(host="http://localhost:11434", model=model)
    tools = [mcp_tool()] if use_tools else None
    agent = Agent(
        client=client,
        instructions=instructions,
        name="LocalAssistant",
        tools=tools,
    )
    async with agent:
        response_stream = agent.run(messages=messages, options=options, stream=True)
        async with response_stream:
            async for update in response_stream:
                if update.role in (None, "assistant") and update.text:
                    yield update.text
            await response_stream.get_final_response()


async def detect_answer_language(
    message: str,
    model: str,
    *,
    options: dict[str, Any],
) -> str:
    text = message.strip()
    if not text:
        return "English"
    try:
        response = await run_agent(
            model=model,
            instructions=(
                "Determine the response language for the user's message. Reply with exactly "
                "one word: Japanese or English. Only return Japanese if the message is "
                "clearly in Japanese; otherwise return English."
            ),
            messages=[Message("user", [text])],
            options={**options, "think": False},
            use_tools=False,
        )
        normalized = response_text(response).strip().lower()
        if "japanese" in normalized:
            return "Japanese"
        if "english" in normalized:
            return "English"
    except Exception:
        pass
    return "Japanese" if re.search(r"[\u3040-\u30ff]", text) else "English"


def context_size(messages: list[Message], summary: str) -> int:
    total = len(summary) + sum(len(message_text(message)) for message in messages)
    for message in messages:
        for content in message.contents:
            if content.type in {"data", "uri"}:
                total += len(content.uri or "")
    return total


def context_messages(messages: list[Message], summary: str) -> list[Message]:
    if not summary:
        return list(messages)
    return [
        Message(
            "system",
            [
                "Conversation summary from earlier context. Use it as background, while "
                "prioritizing the latest messages:\n" + summary
            ],
        ),
        *messages,
    ]


async def compact_context(
    *,
    model: str,
    request_id: str,
    messages: list[Message],
    summary: str,
    context_limit: int,
    options: dict[str, Any],
) -> tuple[list[Message], str]:
    size = context_size(messages, summary)
    if size <= context_limit or len(messages) <= MESSAGES_TO_KEEP:
        log_line(request_id, "compact", f"skipped (size={size}/{context_limit}, messages={len(messages)})")
        return messages, summary

    log_line(request_id, "compact", f"compacting (size={size}/{context_limit}, messages={len(messages)})")
    old_messages = messages[:-MESSAGES_TO_KEEP]
    prompt = (
        "Create a concise, factual conversation summary for the main agent. Preserve user "
        "requirements, decisions, relevant tool results, errors, and unresolved tasks. "
        "Do not add new facts. Summarize the supplied conversation messages, including "
        "information conveyed by attached images.\n\n"
        f"Previous summary:\n{summary or '(none)'}"
    )
    response = await run_agent(
        model=model,
        instructions="You summarize conversation context accurately and concisely.",
        messages=[Message("system", [prompt]), *old_messages],
        options={**options, "think": True},
        use_tools=False,
    )
    summary = response_text(response)
    messages = messages[-MESSAGES_TO_KEEP:]
    log_line(request_id, "compact", f"done (removed={len(old_messages)}, summary_len={len(summary)})")
    return messages, summary


def session_time_instruction(started_at: str) -> str:
    return f"Chat start date and time: {started_at}.\n"


async def verify_material(
    *,
    model: str,
    request_id: str,
    material_label: str,
    material: str,
    messages: list[Message],
    summary: str,
    started_at: str,
    options: dict[str, Any],
    reasoning: bool,
) -> tuple[str, str]:
    prompt = (
        session_time_instruction(started_at)
        + f"You are the verification phase. Check the {material_label} below for contradictions, "
        "missing information, unsupported claims, dates, times, units, logical errors, and any "
        "requirements from the user's request that were not met. Return concise verification "
        "notes and corrections.\n\n"
        f"{material_label.capitalize()}:\n{material}\n\n"
        "End your reply with exactly one final line, with no other text on it: 'STATUS: OK' if "
        "this is sound and sufficient to answer the user, or 'STATUS: NEEDS_REVISION' if it "
        "must be corrected or completed before answering."
    )
    response = await run_agent(
        model=model,
        instructions=prompt,
        messages=context_messages(messages, summary),
        options={**options, "think": reasoning},
        use_tools=False,
    )
    raw_notes = response_text(response)
    status_match = re.search(r"STATUS:\s*(OK|NEEDS_REVISION)\s*$", raw_notes.strip())
    if not status_match:
        notes = (
            raw_notes.strip()
            + "\nVerifier output did not end with a valid STATUS line; this result is not considered verified."
        ).strip()
        log_line(request_id, "verify", "status=INVALID")
        return "INVALID", notes
    status = status_match.group(1)
    notes = raw_notes[: status_match.start()].strip()
    log_line(request_id, "verify", f"status={status}")
    return status, notes


def build_answer_instructions(
    *,
    answer_language: str,
    started_at: str,
    research_notes: str = "",
    draft_notes: str = "",
    verification_notes: str = "",
    verification_status: str = "OK",
) -> str:
    notes_section = ""
    if research_notes:
        notes_section += f"Research notes:\n{research_notes}\n\n"
    if draft_notes:
        notes_section += f"Draft answer:\n{draft_notes}\n\n"
    if verification_notes:
        notes_section += f"Verification notes:\n{verification_notes}\n\n"
    instructions = (
        session_time_instruction(started_at)
        + "You are the main agent. "
        + ("Answer the user's request using the notes below." if notes_section else "Answer the user's request.")
        + " Do not mention internal phases or hidden reasoning. Answer entirely in "
        f"{answer_language}.\n\n{notes_section}"
    )
    if verification_status != "OK":
        instructions += (
            "The verification phase did not confirm the material as sound. Be transparent about "
            "any unresolved or unverified points in your answer. Do not present them as verified facts.\n"
        )
    return instructions


async def prepare_turn(
    *,
    model: str,
    messages: list[Message],
    summary: str,
    with_docker_mcp: bool,
    context_limit: int,
    max_refine_loops: int,
    disable_research: bool,
    disable_complicated: bool,
    disable_direct: bool,
    options: dict[str, Any],
    request_id: str | None = None,
) -> TurnPlan:
    request_id = request_id or new_request_id()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    messages, summary = await compact_context(
        model=model,
        request_id=request_id,
        messages=messages,
        summary=summary,
        context_limit=context_limit,
        options=options,
    )
    latest_user = next((message.text for message in reversed(messages) if message.role == "user"), "")
    answer_language = await detect_answer_language(latest_user, model, options=options)

    log_line(request_id, "route", "invoked")
    route_response = await run_agent(
        model=model,
        instructions=(
            session_time_instruction(started_at)
            + "Classify the user's request. Reply with exactly one word:\n"
            "RESEARCH if it needs external facts, current information, MCP tools, or verification "
            "against real-world data.\n"
            "COMPLICATED if it does NOT need external research or tools, but is a non-trivial "
            "reasoning, math, logic, coding, or writing task where a careful draft should be checked "
            "and refined before answering.\n"
            "DIRECT for casual conversation, simple translation, simple rewriting, or other trivial "
            "tasks that need no verification.\nFor anything else or if uncertain, reply MODERATED."
        ),
        messages=context_messages(messages, summary),
        options={**options, "think": False},
        use_tools=False,
    )
    route = response_text(route_response).strip().upper()
    if "RESEARCH" in route and not disable_research and with_docker_mcp:
        task_route = "RESEARCH"
    elif "COMPLICATED" in route and not disable_complicated:
        task_route = "COMPLICATED"
    elif "DIRECT" in route and not disable_direct:
        task_route = "DIRECT"
    else:
        task_route = "MODERATED"
    log_line(request_id, "route", f"task_route={task_route!r}")

    research_notes = ""
    draft_notes = ""
    verification_notes = ""
    verification_status = "OK"
    if task_route in {"RESEARCH", "COMPLICATED", "MODERATED"}:
        is_research = task_route == "RESEARCH"
        is_moderated = task_route == "MODERATED"
        material_label = "research findings" if is_research else "draft answer"
        phase = "research" if is_research else "moderated_draft" if is_moderated else "draft"
        for refine_count in range(max_refine_loops + 1):
            log_line(request_id, phase, f"invoked (refine_count={refine_count})")
            if is_research:
                phase_instructions = (
                    session_time_instruction(started_at)
                    + "You are the research phase of the main agent. Investigate the user's request "
                    "and use available tools when useful. Return factual findings and unresolved points."
                )
                if refine_count and verification_status != "OK":
                    phase_instructions += (
                        f"\n\nThis is a refinement pass (attempt {refine_count + 1}/{max_refine_loops + 1}). "
                        "Address the following verification issues before returning updated findings:\n"
                        f"{verification_notes}"
                    )
            else:
                phase_instructions = (
                    session_time_instruction(started_at)
                    + (
                        "You are the moderated drafting phase of the main agent. This task requires "
                        "some careful consideration but does NOT require external research or tools. "
                        "Produce a candidate answer for the verification phase to check."
                        if is_moderated
                        else "You are the drafting phase of the main agent. This task is complicated "
                        "but does NOT require external research or tools. Work through it carefully "
                        "and produce a complete candidate answer for verification."
                    )
                )
                if refine_count and verification_status != "OK":
                    phase_instructions += (
                        f"\n\nThis is a refinement pass (attempt {refine_count + 1}/{max_refine_loops + 1}). "
                        "Address the following verification issues before returning an updated draft:\n"
                        f"{verification_notes}"
                    )
            phase_response = await run_agent(
                model=model,
                instructions=phase_instructions,
                messages=context_messages(messages, summary),
                options={**options, "think": not is_moderated},
                use_tools=is_research and with_docker_mcp,
                require_tools=is_research,
            )
            material = response_text(phase_response)
            if is_research:
                research_notes = material
            else:
                draft_notes = material
            log_line(request_id, phase, f"done ({material_label}_len={len(material)})")

            verification_status, verification_notes = await verify_material(
                model=model,
                request_id=request_id,
                material_label=material_label,
                material=material,
                messages=context_messages(messages, summary),
                summary="",
                started_at=started_at,
                options=options,
                reasoning=not is_moderated,
            )
            if verification_status == "OK":
                break

        if task_route == "RESEARCH":
            draft_notes = ""

    answer_instructions = build_answer_instructions(
        answer_language=answer_language,
        started_at=started_at,
        research_notes=research_notes,
        draft_notes=draft_notes,
        verification_notes=verification_notes,
        verification_status=verification_status,
    )
    return TurnPlan(
        request_id=request_id,
        history=messages,
        summary=summary,
        answer_language=answer_language,
        answer_instructions=answer_instructions,
        fallback_answer="\n\n".join(
            part for part in (verification_notes, draft_notes, research_notes) if part
        ),
        generation_options=options,
        use_tools=with_docker_mcp,
    )


async def run_final_answer(model: str, plan: TurnPlan) -> tuple[str, str]:
    started = time.perf_counter()
    log_line(plan.request_id, "answer", f"invoked (language={plan.answer_language})")
    response = await run_agent(
        model=model,
        instructions=plan.answer_instructions,
        messages=context_messages(plan.history, plan.summary),
        options={**plan.generation_options, "think": True},
        use_tools=plan.use_tools,
    )
    answer = response_text(response)
    reasoning = response_reasoning(response)
    if not answer.strip():
        if plan.fallback_answer:
            log_line(plan.request_id, "answer", "empty response, using fallback notes")
            answer = plan.fallback_answer
    log_line(plan.request_id, "answer", f"done (answer_len={len(answer)}, elapsed={time.perf_counter() - started:.3f}s)")
    return answer, reasoning


async def stream_final_answer(model: str, plan: TurnPlan) -> AsyncIterator[str]:
    log_line(plan.request_id, "answer", f"invoked (language={plan.answer_language})")
    async for chunk in stream_agent(
        model=model,
        instructions=plan.answer_instructions,
        messages=context_messages(plan.history, plan.summary),
        options={**plan.generation_options, "think": True},
        use_tools=plan.use_tools,
    ):
        yield chunk
    log_line(plan.request_id, "answer", "stream completed")


def print_answer(answer: str, reasoning: str = "") -> None:
    if reasoning:
        print(f"Reasoning: {reasoning}")
        print("==================================")
    print(answer)


def unload_ollama_model(model: str) -> None:
    try:
        Client(host="http://localhost:11434").chat(
            model=model,
            messages=[],
            keep_alive=0,
        )
        print(f"Ollama model unloaded: {model}", file=sys.stderr)
    except Exception as error:
        print(f"Could not unload Ollama model {model}: {error}", file=sys.stderr)


def api_messages_to_agent(messages: list[dict[str, Any]]) -> list[Message]:
    converted: list[Message] = []
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
            converted.append(Message("system" if role == "developer" else role, [content]))
            continue
        if role != "user":
            raise ValueError(f"unsupported message role: {role}")
        if isinstance(content, str):
            converted.append(Message("user", [content]))
            continue
        if not isinstance(content, list) or not content:
            raise ValueError("user content must be text or a non-empty array of content parts")
        parts: list[Content] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("user content parts must be objects")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(Content.from_text(part["text"]))
                continue
            if part.get("type") != "image_url":
                raise ValueError("user content supports only text and image_url parts")
            image_value = part.get("image_url")
            image_url = image_value.get("url") if isinstance(image_value, dict) else image_value
            if not isinstance(image_url, str):
                raise ValueError("image_url must contain a URL string")
            header, separator, encoded = image_url.partition(",")
            media_type = header.removeprefix("data:").removesuffix(";base64")
            if header != f"data:{media_type};base64" or media_type not in SUPPORTED_IMAGE_TYPES or not separator or not encoded:
                raise ValueError("images must use a base64 data URL with JPEG, PNG, GIF, or WebP")
            if len(encoded) > ((MAX_API_IMAGE_BYTES + 2) // 3) * 4:
                raise ValueError("image exceeds the 20 MiB API limit")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("image data is not valid base64") from error
            if len(data) > MAX_API_IMAGE_BYTES:
                raise ValueError("image exceeds the 20 MiB API limit")
            parts.append(Content.from_data(data, media_type))
        converted.append(Message("user", parts))
    return converted


def api_generation_options(
    request: dict[str, Any],
    *,
    temperature: float,
    seed: int | None,
    top_p: float | None,
    num_predict: int | None,
) -> dict[str, Any]:
    request_temperature = request.get("temperature", temperature)
    if request_temperature is None:
        request_temperature = temperature
    if isinstance(request_temperature, bool) or not isinstance(request_temperature, (int, float)) or not math.isfinite(request_temperature) or request_temperature < 0:
        raise ValueError("temperature must be a non-negative number")
    request_seed = request.get("seed", seed)
    if request_seed is None:
        request_seed = seed
    if request_seed is not None and (isinstance(request_seed, bool) or not isinstance(request_seed, int)):
        raise ValueError("seed must be an integer")
    request_top_p = request.get("top_p", top_p)
    if request_top_p is None:
        request_top_p = top_p
    if request_top_p is not None and (
        isinstance(request_top_p, bool)
        or not isinstance(request_top_p, (int, float))
        or not math.isfinite(request_top_p)
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
    result: dict[str, Any] = {"temperature": request_temperature}
    if request_seed is not None:
        result["seed"] = request_seed
    if request_top_p is not None:
        result["top_p"] = request_top_p
    if request_num_predict is not None:
        result["num_predict"] = request_num_predict
    return result


def prepare_server_messages(messages: list[dict[str, Any]]) -> list[Message]:
    try:
        return api_messages_to_agent(messages)
    except ValueError as error:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(error)) from error


def run_server(
    *,
    model: str,
    without_docker_mcp: bool,
    context_limit: int,
    max_refine_loops: int,
    host: str,
    port: int,
    disable_research: bool,
    disable_complicated: bool,
    disable_direct: bool,
    temperature: float,
    seed: int | None,
    top_p: float | None,
    num_predict: int | None,
) -> None:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        import uvicorn
    except ImportError as error:
        raise SystemExit("API サーバーには fastapi と uvicorn が必要です。") from error

    app = FastAPI(title="Agent Framework OpenAI-Compatible API")

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "ollama"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, Any]):
        raw_messages = request.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise HTTPException(status_code=400, detail="messages is required")
        messages = prepare_server_messages(raw_messages)
        request_id = new_request_id()
        started = time.perf_counter()
        try:
            options = api_generation_options(
                request,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                num_predict=num_predict,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        async def prepare_request_plan() -> TurnPlan:
            try:
                return await prepare_turn(
                    model=model,
                    messages=messages,
                    summary="",
                    with_docker_mcp=not without_docker_mcp,
                    context_limit=context_limit,
                    max_refine_loops=max_refine_loops,
                    disable_research=disable_research,
                    disable_complicated=disable_complicated,
                    disable_direct=disable_direct,
                    options=options,
                    request_id=request_id,
                )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

        completion_id = f"chatcmpl-{request_id}"
        created = int(datetime.now().timestamp())

        if request.get("stream"):
            async def stream_response() -> AsyncIterator[str]:
                role = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role, ensure_ascii=False)}\n\n"
                try:
                    plan = await prepare_request_plan()
                    async for chunk in stream_final_answer(model, plan):
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except Exception as error:
                    print(f"[{request_id}] [server] stream failed: {error}", flush=True)
                    yield 'data: {"error":{"message":"The response stream failed.","type":"server_error"}}\n\n'
                    yield "data: [DONE]\n\n"
                    return
                print(f"[{request_id}] [server] answer completed in {time.perf_counter() - started:.3f}s", flush=True)
                finish = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        plan = await prepare_request_plan()
        answer, _ = await run_final_answer(model, plan)
        print(f"[{request_id}] [server] answer completed in {time.perf_counter() - started:.3f}s", flush=True)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            }
        )

    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        unload_ollama_model(model)


async def run_cli_turn(
    *,
    model: str,
    messages: list[Message],
    summary: str,
    args: argparse.Namespace,
) -> tuple[TurnResult, str]:
    options = generation_options(
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        num_predict=args.num_predict,
        reasoning=True,
    )
    request_id = new_request_id()
    plan = await prepare_turn(
        model=model,
        messages=messages,
        summary=summary,
        with_docker_mcp=not args.without_docker_mcp,
        context_limit=args.context_limit,
        max_refine_loops=args.max_refine_loops,
        disable_research=args.disable_research,
        disable_complicated=args.disable_complicated,
        disable_direct=args.disable_direct,
        options=options,
        request_id=request_id,
    )
    answer, reasoning = await run_final_answer(model, plan)
    return TurnResult(plan, answer, reasoning), plan.summary


async def main(args: argparse.Namespace) -> None:
    model = args.model
    summary = ""
    messages: list[Message] = []
    if args.message is not None:
        messages.append(Message("user", [args.message]))
        started = time.perf_counter()
        async with spinner("エージェントが処理中"):
            result, summary = await run_cli_turn(
                model=model, messages=messages, summary=summary, args=args
            )
        print(f"[{result.plan.request_id}] [cli] answer completed in {time.perf_counter() - started:.3f}s")
        print_answer(result.answer, result.reasoning)
        return

    print("CUI チャットを開始しました。終了するには /exit または /quit を入力してください。")
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
        messages.append(Message("user", [user_message]))
        started = time.perf_counter()
        async with spinner("エージェントが処理中"):
            result, summary = await run_cli_turn(
                model=model, messages=messages, summary=summary, args=args
            )
        messages = result.plan.history
        messages.append(Message("assistant", [result.answer]))
        print(f"[{result.plan.request_id}] [cli] answer completed in {time.perf_counter() - started:.3f}s")
        print_answer(result.answer, result.reasoning)


if __name__ == "__main__":
    args = parse_arguments()
    if args.serve:
        run_server(
            model=args.model,
            without_docker_mcp=args.without_docker_mcp,
            context_limit=args.context_limit,
            max_refine_loops=args.max_refine_loops,
            host=args.host,
            port=args.port,
            disable_research=args.disable_research,
            disable_complicated=args.disable_complicated,
            disable_direct=args.disable_direct,
            temperature=args.temperature,
            seed=args.seed,
            top_p=args.top_p,
            num_predict=args.num_predict,
        )
    else:
        try:
            asyncio.run(main(args))
        except KeyboardInterrupt:
            print("\n処理を中断しました。")
        finally:
            unload_ollama_model(args.model)

"""OpenAI-compatible CLI client for the local agent API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import mimetypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.logging import RichHandler


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024

console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(RichHandler(console=Console(stderr=True), rich_tracebacks=True))
logger.propagate = False


@dataclass
class ApiResult:
    content: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください") from error
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
        raise argparse.ArgumentTypeError("top-p は 0 から 1 の有限数で指定してください")
    return number


async def read_stdin_message() -> str:
    """Read one interactive line or all piped stdin, preserving line breaks."""
    if sys.stdin.isatty():
        return await asyncio.to_thread(console.input, "[cyan]メッセージ > [/cyan]")
    return await asyncio.to_thread(sys.stdin.read)


async def encode_image(path: Path) -> str:
    """Encode an image as a data URL accepted by the local Ollama endpoint."""
    if not path.is_file():
        raise ValueError(f"画像ファイルがありません: {path}")
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in SUPPORTED_IMAGE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_TYPES))
        raise ValueError(f"未対応の画像形式です。対応形式: {supported}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("画像サイズは 20 MiB 以下にしてください")
    image_data = await asyncio.to_thread(path.read_bytes)
    if len(image_data) > MAX_IMAGE_BYTES:
        raise ValueError("画像サイズは 20 MiB 以下にしてください")
    encoded = base64.b64encode(image_data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def user_message(text: str, image_data_url: str | None = None) -> dict[str, Any]:
    if image_data_url is None:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }


def usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return None


def trim_history(messages: list[dict[str, Any]], keep_turns: int) -> None:
    """Bound interactive history while retaining system prompts and full turns."""
    system_messages = [message for message in messages if message.get("role") == "system"]
    dialogue = [message for message in messages if message.get("role") != "system"]
    keep_messages = keep_turns * 2
    if len(dialogue) > keep_messages:
        messages[:] = system_messages + dialogue[-keep_messages:]


async def api_call_with_spinner(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    timeout: float,
    stream: bool = False,
    show_spinner: bool = True,
    temperature: float | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> ApiResult:
    """Call the chat endpoint and collect its text and completion metadata."""
    if not messages:
        raise ValueError("送信するメッセージがありません")

    request: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if seed is not None:
        request["seed"] = seed
    if top_p is not None:
        request["top_p"] = top_p
    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    if show_spinner:
        with console.status("[cyan]Agent API に問い合わせ中…[/cyan]", spinner="dots"):
            response = await client.chat.completions.create(
                **request, timeout=timeout
            )
    else:
        response = await client.chat.completions.create(
            **request, timeout=timeout
        )

    if stream:
        parts: list[str] = []
        finish_reason = None
        async for chunk in response:
            for choice in chunk.choices:
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                content = choice.delta.content
                if isinstance(content, str) and content:
                    parts.append(content)
                    console.print(content, end="", markup=False, highlight=False)
        console.print()
        return ApiResult("".join(parts), finish_reason, None)

    if not response.choices:
        raise RuntimeError("API が choices を含まない応答を返しました")
    choice = response.choices[0]
    if not isinstance(choice.message.content, str):
        raise RuntimeError("API 応答にテキスト本文がありません")
    return ApiResult(
        choice.message.content,
        choice.finish_reason,
        usage_dict(response.usage),
    )


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature", type=non_negative_float, default=None, help="生成 temperature"
    )
    parser.add_argument("--seed", type=int, default=None, help="生成 seed")
    parser.add_argument("--top-p", type=top_p_value, default=None, help="生成 top-p")
    parser.add_argument(
        "--max-tokens", type=positive_int, default=None, help="最大生成トークン数"
    )


def print_result_metadata(
    *, model: str, elapsed: float, result: ApiResult
) -> None:
    usage = result.usage or {}
    tokens = usage.get("total_tokens")
    details = [f"model={model}", f"time={elapsed:.2f}s"]
    if result.finish_reason:
        details.append(f"finish={result.finish_reason}")
    if tokens is not None:
        details.append(f"tokens={tokens}")
    console.print(f"[dim]{' | '.join(details)}[/dim]")


async def resolve_model(client: AsyncOpenAI, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("API の /v1/models に利用可能なモデルがありません")
    return models.data[0].id


async def send_turn(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    message_text: str,
    image_data_url: str | None,
    args: argparse.Namespace,
    json_output: bool = False,
) -> str:
    trim_history(messages, args.history_turns)
    messages.append(user_message(message_text, image_data_url))
    started = time.perf_counter()
    result = await api_call_with_spinner(
        client,
        model,
        messages,
        timeout=args.timeout,
        stream=args.stream,
        show_spinner=not json_output,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    elapsed = time.perf_counter() - started
    messages.append({"role": "assistant", "content": result.content})

    if json_output:
        sys.stdout.write(
            json.dumps(
                {
                    "model": model,
                    "output": result.content,
                    "elapsed_seconds": round(elapsed, 3),
                    "finish_reason": result.finish_reason,
                    "usage": result.usage,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stdout.flush()
    elif not args.stream:
        console.print(result.content, markup=False, highlight=False)
        print_result_metadata(model=model, elapsed=elapsed, result=result)
    else:
        print_result_metadata(model=model, elapsed=elapsed, result=result)
    return result.content


async def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI 互換 Agent API クライアント")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--message", type=str, default=None, help="送信するメッセージ")
    input_group.add_argument(
        "--message-file", type=Path, default=None, help="UTF-8 のメッセージファイル"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="モデル名。省略時は API の /v1/models から最初のモデルを選ぶ",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"API の URL（デフォルト: {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--timeout", type=positive_int, default=300, help="API タイムアウト秒数"
    )
    parser.add_argument("--system", default=None, help="会話に含める system メッセージ")
    parser.add_argument("--image", type=Path, default=None, help="添付する画像ファイル")
    parser.add_argument(
        "--interactive", action="store_true", help="複数ターンの対話モードを開始する"
    )
    parser.add_argument(
        "--history-turns",
        type=positive_int,
        default=12,
        help="対話モードで保持する直近ターン数（デフォルト: 12）",
    )
    parser.add_argument(
        "--stream", action="store_true", help="生成中のテキストを逐次表示する"
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="応答メタデータを JSON で出力"
    )
    add_sampling_arguments(parser)
    args = parser.parse_args()

    if args.interactive and (args.message is not None or args.message_file is not None):
        parser.error("--interactive と --message/--message-file は同時に使えません")
    if args.interactive and not sys.stdin.isatty():
        parser.error("--interactive は対話端末で実行してください")
    if args.json_output and (args.interactive or args.stream):
        parser.error("--json は --interactive / --stream と同時に使えません")

    if args.message_file is not None:
        try:
            initial_message = await asyncio.to_thread(
                args.message_file.read_text, encoding="utf-8"
            )
        except OSError as error:
            parser.error(f"メッセージファイルを読めません: {error}")
    elif args.message is not None:
        initial_message = args.message
    elif args.interactive:
        initial_message = ""
    else:
        initial_message = await read_stdin_message()

    if not args.interactive and not initial_message.strip():
        parser.error("メッセージが空です。--message、--message-file、または標準入力で指定してください。")
    if args.model is not None and not args.model.strip():
        parser.error("--model に空文字列は指定できません")
    if not args.base_url.strip():
        parser.error("--base-url に空文字列は指定できません")

    image_data_url = None
    if args.image is not None:
        try:
            image_data_url = await encode_image(args.image)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = AsyncOpenAI(base_url=base_url, api_key="local-test", timeout=args.timeout)
    try:
        model = await resolve_model(client, args.model)
        messages: list[dict[str, Any]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})

        if args.interactive:
            console.print(
                f"[dim]model: {model} | /exit または /quit で終了[/dim]"
            )
            pending_image = image_data_url
            while True:
                try:
                    message_text = await asyncio.to_thread(
                        console.input, "[cyan]you > [/cyan]"
                    )
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]対話を終了します。[/dim]")
                    break
                if message_text.strip().lower() in {"/exit", "/quit"}:
                    break
                if not message_text.strip():
                    continue
                if pending_image is not None:
                    console.print(f"[dim]image: {args.image}[/dim]")
                console.print("[bold green]assistant >[/bold green]")
                await send_turn(
                    client=client,
                    model=model,
                    messages=messages,
                    message_text=message_text,
                    image_data_url=pending_image,
                    args=args,
                )
                pending_image = None
            return 0

        if not args.json_output:
            console.print("[bold blue]INPUT[/bold blue]")
            console.print(initial_message, markup=False, highlight=False)
            if image_data_url is not None:
                console.print(f"[dim]image: {args.image}[/dim]")
            console.print(f"[dim]model: {model}[/dim]")
            console.print("[bold cyan]" + "─" * 64 + "[/bold cyan]")
            console.print("[bold green]OUTPUT[/bold green]")
        await send_turn(
            client=client,
            model=model,
            messages=messages,
            message_text=initial_message,
            image_data_url=image_data_url,
            args=args,
            json_output=args.json_output,
        )
        return 0
    except Exception as error:
        logger.error("API 呼び出しに失敗しました (%s): %s", type(error).__name__, error)
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        console.print("\n[yellow]中断しました。[/yellow]")
        raise SystemExit(130)

import asyncio
import argparse
from agent_framework import Agent, tool, MCPStdioTool
from agent_framework.ollama import OllamaChatClient
from rich.console import Console
from rich.live import Live

console = Console()

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # 実際のAPIコールをここに実装
    return f"The weather in {city} is sunny with a high of 25°C."

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ollama を使う WeatherAgent のテスト"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="使用する Ollama モデル名（必須）",
    )
    return parser.parse_args()

async def main(model: str):
    client=OllamaChatClient(
        host="http://localhost:11434", model=model, additional_properties=120.0
    )
    docker_mcp = MCPStdioTool(
        name="DockerMcp",
        # command="docker",
        command="/Docker/host/bin/docker.exe",
        args=["mcp", "gateway", "run", "--profile", "default"]
    )
    agent = Agent(
        client=client,
        default_options={"think": True},  # エージェントレベルで推論を有効化
        name="WeatherAgent",
        instructions="You are a helpful weather assistant.",
        tools=[get_weather, docker_mcp],
        # tools=[get_weather],
    )

    result = await agent.run("What's the weather like in Tokyo?")

    # Rich spinner 表示
    with Live("[spinner] Agent is reasoning...", console=console, refresh_per_second=10) as live:
        # 推論中は何らかの更新が必要（例：少し待ってから更新）
        await asyncio.sleep(0.5)
        live.update("[spinner] Agent is generating response...")

    reasoning = "".join(
        (c.text or "") for c in result.messages[-1].contents if c.type == "text_reasoning"
    )
    print(f"\nReasoning: {reasoning}")
    print("==================================")
    print(result)

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args.model))

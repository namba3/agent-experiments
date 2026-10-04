# Agent Experiments

A small collection of Python agent experiments. It includes Ollama agents built
with Agent Framework and LangGraph, plus an OpenAI-compatible CLI client for the
LangGraph API.

[日本語](README.md)

## Scripts

| File | Description |
| --- | --- |
| `agent_framework_agent.py` | An Agent Framework and Ollama weather-agent example. Its weather tool returns a fixed sample response. It connects to the Docker MCP gateway. |
| `langgraph_agent.py` | A LangGraph agent using Ollama. Run it interactively or start it as an OpenAI-compatible API server. |
| `langgraph_api_client.py` | A CLI client for the OpenAI-compatible API. Supports interactive chat, streaming, and image attachments. |

## Setup

Python 3.10 or later and Ollama are required. Create a virtual environment and
install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Start Ollama and make sure the model you want to use is available. The agents
connect to Ollama at `http://localhost:11434` by default.

## Usage

Run the Agent Framework example. The Docker MCP gateway command must be
available:

```bash
python agent_framework_agent.py --model <ollama-model>
```

Set the `DOCKER_MCP_COMMAND` environment variable to select the Docker
executable. It defaults to `docker`.

Run the LangGraph agent interactively or pass it a single message:

```bash
python langgraph_agent.py --model <ollama-model>
python langgraph_agent.py --model <ollama-model> --message "Explain how RAG works."
```

Start the LangGraph API server, then run the client in another terminal. By
default, the server listens on `127.0.0.1:8000`.

```bash
python langgraph_agent.py --model <ollama-model> --serve
python langgraph_api_client.py --message "Explain how RAG works."
```

If no model is specified, the client selects one from the API's `/v1/models`
endpoint. Streaming and image attachments are also available:

```bash
python langgraph_api_client.py --message "Summarize this image." --image path/to/image.png --stream
```

Run any script with `--help` to see all available options.

## Configuration and behavior

- Docker MCP is disabled by default in the LangGraph agent. Enable it with
  `--with-docker-mcp`. The Agent Framework example always connects to it.
- The RESEARCH route requires available tools, so pass both
  `--enable-research` and `--with-docker-mcp`.
- `langgraph_api_client.py` connects to `http://127.0.0.1:8000/v1` by default.
- Image attachments support JPEG, PNG, GIF, and WebP, up to 20 MiB.

```bash
python langgraph_agent.py --model <ollama-model> --enable-research --with-docker-mcp
```

See [`requirements.txt`](requirements.txt) for dependencies and [`AGENTS.md`](AGENTS.md)
for repository editing guidance.

# Agent Experiments

A small collection of Python agent experiments. It includes Ollama agents built
with Agent Framework and LangGraph, plus a shared OpenAI-compatible API client.

[日本語](README.md)

## Scripts

| File | Description |
| --- | --- |
| `agent_framework_agent.py` | An Agent Framework and Ollama agent with the same CLI settings, RESEARCH/verification flow, context compaction, and OpenAI-compatible API as the LangGraph version. |
| `langgraph_agent.py` | A LangGraph agent using Ollama. Run it interactively or start it as an OpenAI-compatible API server. |
| `agent_api_client.py` | A CLI client for the OpenAI-compatible API. Supports interactive chat, streaming, and image attachments. |

## Setup

Python 3.10 or later and Ollama are required. We recommend uv to create the
virtual environment and install the locked dependencies:

```bash
uv sync --locked
source .venv/bin/activate
```

`uv sync` creates `.venv` and installs runtime dependencies plus Ruff. To use pip
instead, create and activate a virtual environment, then install the locked
`requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In Windows PowerShell, activate the environment after syncing with uv or creating
the pip virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Start Ollama and make sure the model you want to use is available. The agents
connect to Ollama at `http://localhost:11434` by default.
Set the `OLLAMA_HOST` environment variable to use a different endpoint.

## Usage

Run the Agent Framework agent interactively or pass a single message:

```bash
python agent_framework_agent.py --model <ollama-model>
python agent_framework_agent.py --model <ollama-model> --message "Explain how RAG works."
```

Pass `--with-docker-mcp` to use the Docker MCP gateway. Set the
`DOCKER_MCP_COMMAND` environment variable to select the Docker executable; it
defaults to `docker`.

Run the LangGraph agent interactively or pass it a single message:

```bash
python langgraph_agent.py --model <ollama-model>
python langgraph_agent.py --model <ollama-model> --message "Explain how RAG works."
```

Start either OpenAI-compatible API server, then run the client in another terminal. By
default, the server listens on `127.0.0.1:8000`.

```bash
python agent_framework_agent.py --model <ollama-model> --serve
python agent_api_client.py --message "Explain how RAG works."
```

The same server CLI is also available through `langgraph_agent.py --serve`.

If no model is specified, the client selects one from the API's `/v1/models`
endpoint. Streaming and image attachments are also available:

```bash
python agent_api_client.py --message "Summarize this image." --image path/to/image.png --stream
```

Run any script with `--help` to see all available options.

## Configuration and behavior

- Docker MCP is disabled by default in both agents. Enable it with
  `--with-docker-mcp`.
- The RESEARCH route requires available tools, so pass both
  `--enable-research` and `--with-docker-mcp`.
- Route labels are `SIMPLE` (answer directly), `STANDARD` (draft and verify),
  `DEEP` (reason carefully, then verify), and `RESEARCH` (gather and verify
  external information).
- Both agent scripts accept the same `--context-limit`, `--max-refine-loops`,
  `--temperature`, `--seed`, `--top-p`, `--num-predict`, and API `--host` / `--port` options.
- `agent_api_client.py` connects to `http://127.0.0.1:8000/v1` by default.
- Image attachments support JPEG, PNG, GIF, and WebP, up to 20 MiB.

```bash
python agent_framework_agent.py --model <ollama-model> --enable-research --with-docker-mcp
```

Direct dependencies are listed in [`pyproject.toml`](pyproject.toml), and
[`uv.lock`](uv.lock) pins versions and hashes for the full dependency tree.
`requirements.txt` is generated from the lockfile.

## Development checks

Run static checks and tests in the locked development environment:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -W error::ResourceWarning -m unittest discover -v
```

After updating dependencies, run `uv lock`, then regenerate the pip file with
`uv export --no-dev --locked --format requirements-txt --output-file requirements.txt`.
See [`AGENTS.md`](AGENTS.md) for repository editing guidance.

## License

This project is dual-licensed; users may choose either **MIT** or
**Apache-2.0**. See the full text of the [MIT License](LICENSE-MIT) and
[Apache License 2.0](LICENSE-APACHE). The MIT copyright attribution is to
[GitHub user `@namba3`](https://github.com/namba3).

## Tests

Run the tests without starting Ollama or Docker MCP:

```bash
uv run --locked python -m unittest discover -v
```

HTTP integration tests start the lightweight Ollama API simulator in
`tests/fake_ollama_server.py` automatically. To run it manually, use
`python tests/fake_ollama_server.py`; it listens on `127.0.0.1:11435` by default.

# Agent Experiments

小規模な Python エージェント実験用リポジトリです。Agent Framework と
LangGraph の Ollama エージェント、および両エージェント用の OpenAI 互換API
クライアントを収録しています。

[English](README.en.md)

## スクリプト

| ファイル | 内容 |
| --- | --- |
| `agent_framework_agent.py` | Agent Framework と Ollama によるエージェント。LangGraph版と同じCLI設定、RESEARCH/検証フロー、会話圧縮、OpenAI互換APIを備えます。 |
| `langgraph_agent.py` | Ollama を使う LangGraph エージェント。対話実行または OpenAI 互換 API サーバーとして起動できます。 |
| `agent_api_client.py` | OpenAI 互換 API サーバーに接続する CLI クライアント。対話、ストリーミング、画像添付に対応します。 |

## セットアップ

Python 3.10 以降と Ollama が必要です。仮想環境を作成して依存パッケージを
インストールします。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell では、仮想環境の有効化に次を使います。

```powershell
.venv\Scripts\Activate.ps1
```

Ollama を起動し、利用するモデルを事前に用意してください。エージェントは
既定で `http://localhost:11434` の Ollama に接続します。

## 使い方

Agent Framework エージェントを対話モードで起動するか、単発のメッセージを渡します。

```bash
python agent_framework_agent.py --model <ollama-model>
python agent_framework_agent.py --model <ollama-model> --message "Explain how RAG works."
```

Docker MCP gateway を使う場合は `--with-docker-mcp` を指定します。
`DOCKER_MCP_COMMAND` 環境変数で Docker 実行ファイルを指定でき、省略時は `docker` を使います。

LangGraph エージェントを対話モードで起動するか、1つのメッセージを渡します。

```bash
python langgraph_agent.py --model <ollama-model>
python langgraph_agent.py --model <ollama-model> --message "Explain how RAG works."
```

OpenAI互換APIサーバーを起動し、別のターミナルからクライアントを実行します。
サーバーは既定で `127.0.0.1:8000` に待ち受けます。

```bash
python agent_framework_agent.py --model <ollama-model> --serve
python agent_api_client.py --message "Explain how RAG works."
```

同じサーバーCLIは `langgraph_agent.py --serve` でも起動できます。

クライアントはモデル名を省略すると、API の `/v1/models` から選択します。
ストリーミングや画像添付も指定できます。

```bash
python agent_api_client.py --message "Summarize this image." --image path/to/image.png --stream
```

各スクリプトの全オプションは `--help` で確認できます。

## 設定と動作

- Docker MCP は両エージェントとも既定で無効です。`--with-docker-mcp` で有効にできます。
- `--enable-research` でRESEARCH経路を使うには、利用可能なツールが必要なため
  `--with-docker-mcp` も指定してください。
- ルート名は `SIMPLE`（直接回答）、`STANDARD`（下書きを検証）、`DEEP`
  （推論を重ねて検証）、`RESEARCH`（外部情報を調査して検証）です。
- Agent Framework版とLangGraph版は同じ `--context-limit`、`--max-refine-loops`、
  `--temperature`、`--seed`、`--top-p`、`--num-predict`、APIの `--host` / `--port` を受け付けます。
- `agent_api_client.py` は既定で `http://127.0.0.1:8000/v1` に接続します。
- 画像添付は JPEG、PNG、GIF、WebP に対応し、最大サイズは 20 MiB です。

```bash
python agent_framework_agent.py --model <ollama-model> --enable-research --with-docker-mcp
```

依存パッケージは [`requirements.txt`](requirements.txt)、リポジトリの編集ルールは
[`AGENTS.md`](AGENTS.md) を参照してください。

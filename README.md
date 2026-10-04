# Agent Experiments

Python での AI エージェント実装を試すための実験用リポジトリです。
Agent Framework と LangGraph を使った Ollama エージェントを収録し、実装や機能を
比較・検証できます。エージェントに接続する OpenAI 互換 API クライアントも含みます。

[English](README.en.md)

## スクリプト

| ファイル | 内容 |
| --- | --- |
| `agent_framework_agent.py` | Agent Framework と Ollama によるエージェント。LangGraph版と同じCLI設定、RESEARCH/検証フロー、会話圧縮、OpenAI互換APIを備えます。 |
| `langgraph_agent.py` | Ollama を使う LangGraph エージェント。対話実行または OpenAI 互換 API サーバーとして起動できます。 |
| `agent_api_client.py` | OpenAI 互換 API サーバーに接続する CLI クライアント。対話、ストリーミング、画像添付に対応します。 |

## セットアップ

Python 3.10 以降と Ollama が必要です。推奨のuvを使うと、ロックファイルに従って
仮想環境と依存パッケージを用意できます。

```bash
uv sync --locked
source .venv/bin/activate
```

`uv sync` は `.venv` を作成し、実行時依存、Ruff、Coverage.pyをインストールします。
uvを使わずpipでインストールする場合は、仮想環境を作成・有効化してからロック済みの
`requirements.txt` を使います。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell ではuv同期後、またはpip用の仮想環境作成後に次のコマンドで有効化します。

```powershell
.venv\Scripts\Activate.ps1
```

Ollama を起動し、利用するモデルを事前に用意してください。エージェントは
既定で `http://localhost:11434` の Ollama に接続します。
`OLLAMA_HOST` 環境変数で接続先を変更できます。

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

直接依存は [`pyproject.toml`](pyproject.toml) に記載し、[`uv.lock`](uv.lock) に全依存の
バージョンとハッシュを固定しています。`requirements.txt` はロックファイルから生成します。

## 開発チェック

ロック済みの開発環境で静的チェックとテストを実行します。

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -W error::ResourceWarning -m coverage run \
  -m unittest discover -v
uv run --locked coverage report
```

同じチェックはGitHub Actionsでpush、pull request、手動実行時にPython 3.10と3.14で
自動実行されます。

依存の更新後は `uv lock` を実行し、`uv export --no-dev --locked --format requirements-txt --output-file requirements.txt`
でpip用ファイルを再生成してください。リポジトリの編集ルールは [`AGENTS.md`](AGENTS.md) を参照してください。

## ライセンス

このプロジェクトはデュアルライセンスで提供され、利用者は **MIT** または
**Apache-2.0** のいずれかを選択できます。全文は[MITライセンス](LICENSE-MIT)と
[Apache License 2.0](LICENSE-APACHE)を参照してください。MITライセンスの著作者表記は
[GitHubユーザー `@namba3`](https://github.com/namba3) です。

## テスト

OllamaやDocker MCPを起動せずにテストを実行できます。

```bash
uv run --locked python -m unittest discover -v
```

HTTP連携のテストでは `tests/fake_ollama_server.py` の軽量Ollama APIシミュレーターを
自動起動します。手動で使う場合は `python tests/fake_ollama_server.py` で
`127.0.0.1:11435` に起動します。

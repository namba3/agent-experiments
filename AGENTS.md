# Repository guidance

## Scope

This repository contains small Python experiments for agent frameworks and a
client for the local LangGraph API. Keep each script runnable as a standalone
entry point.

## File naming

- Use `<framework>_agent.py` for framework agent examples.
- Use `<service>_api_client.py` for API clients.
- Name files for their role; avoid generic names such as `test.py` when the
  script is an executable example rather than a test suite.

## Configuration and privacy

- Do not commit personal information, credentials, machine-specific settings,
  or local absolute paths.
- Read machine-specific executable paths and other local configuration from
  environment variables. Use portable command names or relative paths as
  defaults where practical.
- Keep `.env` files and other local configuration out of version control.
- List direct third-party Python imports in `requirements.txt`; do not list
  Python standard-library modules.

## Changes

- Keep changes focused on the requested script or documentation.
- Inspect the final diff and stage only files relevant to the task before
  committing.

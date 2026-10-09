# Repository guidance

## Scope

This repository contains small Python experiments for agent frameworks and a
client for the local agent API. Keep each script runnable as a standalone
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
- Declare direct runtime dependencies and development tools in `pyproject.toml`.
- Keep `uv.lock` committed and use `uv sync --locked` to reproduce the development
  environment.
- Generate `requirements.txt` from the lock with
  `uv export --no-dev --locked --format requirements-txt --output-file requirements.txt`;
  do not edit the generated file by hand.

## Changes

- Keep changes focused on the requested script or documentation.
- Inspect the final diff and stage only files relevant to the task before
  committing.

## Code Review

When performing a code review, read and follow the guidelines in
[REVIEW.md](./REVIEW.md). These guidelines apply to code review tasks; for
regular development work, follow the instructions above.

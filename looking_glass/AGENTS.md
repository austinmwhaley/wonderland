# AGENTS.md

> Repo-wide doctrine lives in `../AGENTS.md` (root). This file only records
> looking_glass-local commands and conventions; where the two disagree, the
> root file wins.

## Project overview

looking_glass is a **sklearn-for-event-streams** Python library. Three factory functions (`create_embedding_model`, `create_temporal_core_model`, `create_supervised_model`) turn business event streams into predictive models: train one state-space backbone, then attach cheap task heads for every downstream prediction.

## Commands

- Install: `python3 -m pip install -r requirements.txt`
- Run tests: `.venv/bin/python -m pytest tests/ -q`
- Run example demo (CPU, ~2 min): `.venv/bin/python scripts/example.py`
- Run full benchmark (needs generated db): `.venv/bin/python scripts/run_full.py`

## Code conventions

- **Spaces** (4) in all files — `ruff format` from the repo root is authoritative (older files were tab-indented; they were converted).
- Type annotations using `from __future__ import annotations` everywhere.
- Public API surfaces go through `looking_glass/__init__.py`.
- Tests live in `tests/` and are installed-runnable (`from looking_glass import …`).
- Scripts in `scripts/` are not part of the installable package; they import `looking_glass` as a dependency.
- The reference-app `ddl` uses `scripts/generate_full.py`, which still produces a SQLite database (products, customers, stores, campaigns, events). This is local-only reference-app storage — SQLite is never the canonical stream (root doctrine: Arrow/DuckDB).

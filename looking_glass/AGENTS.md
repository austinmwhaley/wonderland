# AGENTS.md

## Project overview

looking_glass is a **sklearn-for-event-streams** Python library. Three factory functions (`create_embedding_model`, `create_temporal_core_model`, `create_supervised_model`) turn business event streams into predictive models: train one state-space backbone, then attach cheap task heads for every downstream prediction.

## Commands

- Install: `python3 -m pip install -r requirements.txt`
- Run tests: `.venv/bin/python -m pytest tests/ -q`
- Run example demo (CPU, ~2 min): `.venv/bin/python scripts/example.py`
- Run full benchmark (needs generated db): `.venv/bin/python scripts/run_full.py`

## Code conventions

- **Tabs** in all library files under `looking_glass/`.  Yes, tabs.  Match the existing file style; if a file uses spaces, keep spaces in that file.
- Type annotations using `from __future__ import annotations` everywhere.
- Public API surfaces go through `looking_glass/__init__.py`.
- Tests live in `tests/` and are installed-runnable (`from looking_glass import …`).
- Scripts in `scripts/` are not part of the installable package; they import `looking_glass` as a dependency.
- `ddl` uses the `generate_full.py` script which produces a SQLite database with tables for products, customers, stores, campaigns, and events.

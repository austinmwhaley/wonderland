# Autonomous loop — run the agent until a goal is met

The agent does **not** run continuously on its own: a session executes tool calls until
it yields. To run "24/7," an **external driver** restarts bounded agent runs and checks a
**goal predicate** between them. Three pieces:

```
loop/status.sh   the GOAL PREDICATE  (exit 0 = goal met)   <-- you edit this
loop/ITER.md     the per-iteration prompt (ONE bounded step)
loop/run.sh      the DRIVER (loops run.sh until status.sh passes)
```

## Run it

```bash
# foreground (watch it):
bash loop/run.sh

# background:
nohup bash loop/run.sh > loop/loop.log 2>&1 &
tail -f loop/loop.log

# true 24/7 — survives logout/disconnect (recommended): tmux
tmux new -s loop 'bash loop/run.sh 2>&1 | tee loop/loop.log'
#   detach: Ctrl-B d   reattach: tmux attach -t loop
```

Or as a **systemd user service** (`~/.config/systemd/user/wonderland-loop.service`):

```ini
[Unit]
Description=wonderland autonomous loop
After=network-online.target
[Service]
WorkingDirectory=/home/austin-whaley/Documents/wonderland
ExecStart=/usr/bin/env bash /home/austin-whaley/Documents/wonderland/loop/run.sh
Restart=on-failure
RestartSec=10
Environment=LOOP_MODEL=openrouter/deepseek/deepseek-v4.1-flash
[Install]
WantedBy=default.target
```
```bash
systemctl --user daemon-reload
systemctl --user enable --now wonderland-loop
journalctl --user -u wonderland-loop -f
loginctl enable-linger $USER      # keep running after logout
```

## Configure

- **Model:** `-m` in `run.sh` (or env `LOOP_MODEL`).
- **Goal:** edit the `STREAMS` table in `status.sh` (name|db|ckpt_dir|as_of). The loop trains
  into `ckpt_dir`, and `status.sh` picks the newest `registry_*.json` there. GOAL = every
  B card AND every B+C card reads `VERDICT: GREEN`.
- **Caps:** `LOOP_MAX_ITERS` / `LOOP_MAX_HOURS` (default 500 / 24h). Hitting a cap exits `2`
  (so you can tell "gave up" from "done").
- **Continuity is in the repo:** STATUS.md + DECISIONS.md + git. Each iteration commits, so a
  crash/session-loss loses at most one step; `run.sh` just starts a fresh session next pass.

## Cost / safety

- `--auto` **auto-approves** actions not explicitly denied. Treat it as powerful: scope it —
  run in a **container/VM**, on a **dedicated git branch**, and add a `permission` deny list
  in `opencode.json` for anything destructive/outward (push, network, rm). It only commits
  locally unless you allow push.
- The prompt forbids committing data/checkpoints (`*.pt/*.parquet/*.duckdb`) — verify the
  repo stays small. Add a disk-space guard if runs write many checkpoints.
- Watch `opencode stats` for token/cost; the caps bound runaway cost.
- Human checkpoint: the prompt writes any **genuine operator decision** to STATUS.md and exits
  that run; the driver keeps looping, so check `loop/status.out` / STATUS.md periodically.

## Why not "one infinite session"?

A single session eventually hits context limits and can drift. Bounded, repeatable runs that
reload state from the repo are more robust and auditable — each step is a commit with a receipt.

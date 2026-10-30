#!/usr/bin/env bash
# Run a command inside the project's memory-capped container (DEC-442).
#
#   scripts/wonder_container.sh -- .venv/bin/python scripts/measure_judge_law.py --legs rl
#   scripts/wonder_container.sh --build          # (re)build the image only
#   scripts/wonder_container.sh --check          # show the cap and the last run's peak
#
# The point: the machine is shared with an LLM server, and a runaway job has taken the host down.
# `--memory` is enforced by the kernel on the whole container, so nothing started here can take
# the LLM server's memory.
#
# The ceiling is 20 GB, and it is a BUDGET, not a round number: the box has one 121.6 GB unified
# pool, vLLM reserves 0.70 of it for Qwen (85.1 GB, per ~/vllm-qwen-serve.sh), and the operator
# keeps 8 GB for the system. That leaves 121.6 - 85.1 - 8 = 28.5 GB for this project; 20 keeps
# 8.5 GB of slack, 24 leaves 4.5, and 30 does not fit at all. Override with WONDER_MEM=24g if a
# run genuinely needs it — but then say so, because the slack is the thing that keeps Qwen up.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${WONDER_WORK:-$HOME/wonderland_work}"
image="${WONDER_IMAGE:-wonderland-runtime:24.04}"
mem="${WONDER_MEM:-20g}"
name="${WONDER_NAME:-wonderland-run}"

build() {
  docker build \
    --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
    -t "$image" -f "$repo/deploy/Dockerfile.wonder" "$repo/deploy"
}

if [[ "${1:-}" == "--build" ]]; then
  build
  exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "image:      $image ($(docker image inspect -f '{{.Id}}' "$image" 2>/dev/null | cut -c1-19 || echo 'not built'))"
  total_gb=$(awk '/^MemTotal:/{printf "%.1f", $2/1048576}' /proc/meminfo)
  qwen_frac=$(grep -oE 'gpu-memory-utilization[[:space:]]+[0-9.]+' "$HOME/vllm-qwen-serve.sh" 2>/dev/null | awk '{print $2}' | head -1)
  qwen_frac="${qwen_frac:-0.70}"
  cap_gb="${mem%g}"
  python3 - "$total_gb" "$qwen_frac" "$cap_gb" <<'PY'
import sys
total, frac, cap = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
qwen, floor = frac * total, 8.0
print(f"memory cap: {cap:.0f} GB  (WONDER_MEM overrides)")
print(f"budget:     Qwen {qwen:.1f} + project {cap:.0f} + system {floor:.0f} = {qwen+cap+floor:.1f} of {total:.1f} GB -> {total-qwen-cap-floor:+.1f} GB slack")
PY
  echo "repo:       $repo -> /repo (rw)"
  echo "work data:  $work -> /work (ro)"
  echo "host free:  $(awk '/MemAvailable/{printf "%.0f GB", $2/1048576}' /proc/meminfo)"
  exit 0
fi

docker image inspect "$image" >/dev/null 2>&1 || build

cmd=("$@")
[[ "${1:-}" == "--" ]] && cmd=("${@:2}")
if [[ ${#cmd[@]} -eq 0 ]]; then
  echo "usage: scripts/wonder_container.sh -- <command...>" >&2
  exit 2
fi

# --memory and --memory-swap equal => the container gets NO swap: exceeding the ceiling kills the
# job rather than letting it thrash the host. --pids-limit and --init bound the other ways a
# process tree can hurt a shared machine. The work directory is mounted read-only: the inputs the
# receipts describe must not be mutable by the job that measures them.
exec docker run --rm --init \
  --name "$name" \
  --memory="$mem" --memory-swap="$mem" \
  --pids-limit 1024 \
  --user "$(id -u):$(id -g)" \
  --workdir /repo \
  -e HOME=/tmp \
  -e PYTHONPATH=/repo \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  -v "$repo:/repo" \
  -v "$work:/work:ro" \
  ${WONDER_EXTRA_ARGS:-} \
  "$image" "${cmd[@]}"

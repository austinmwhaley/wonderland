#!/usr/bin/env bash
# Run ANY project command inside the memory-capped container (DEC-442, DEC-450).
#
#   scripts/wonder_container.sh -- .venv/bin/python scripts/measure_judge_law.py --legs rl
#   scripts/wonder_container.sh --gpu -- \
#       .venv/bin/python -m looking_glass.customer_foundation_model train \
#       --db /work/demo/customer_event_stream.parquet --out-dir /out
#   scripts/wonder_container.sh --gpu --out ~/wonderland_work/cfm_run:/out -- <training cmd>
#   scripts/wonder_container.sh --build          # (re)build the image only
#   scripts/wonder_container.sh --check          # the budget and every ceiling in force
#   scripts/wonder_container.sh --plan -- ...    # print the exact docker argv, run nothing
#
# The point: the machine is shared with an LLM server, and a runaway job has taken the host
# down. The ceiling is 20 GB and it is a BUDGET, not a round number: one 121.6 GB unified pool,
# vLLM reserves 0.70 for Qwen (85.1 GB), the operator keeps 8 GB for the system. That leaves
# 28.5 GB for this project; 20 keeps 8.5 GB of slack (DEC-443). Override with WONDER_MEM=24g
# only if a run genuinely needs it — say so when you do, because the slack keeps Qwen up.
#
# HOW THE 20 GB IS ENFORCED — three layers, because one is not enough on this hardware:
#
#   1. CONTAINER cgroup (`--memory` == `--memory-swap`, so no swap): a HARD kernel ceiling on
#      host RAM. Exceeding it kills the job (exit 137); the host and the LLM are untouched.
#   2. GPU ALLOCATOR cap: on the GB10 the GPU draws from the SAME pool, but CUDA allocations
#      are NOT charged to the container's cgroup (measured with vLLM: 337 MiB reported while
#      holding 97 GiB). So `looking_glass.gpu_budget` reads this container's `memory.max` and
#      derives the torch allocator fraction that keeps `device + host usage + reserve <= 20 GB`.
#   3. HOST FLOOR guard: `scripts/gpu_watchdog.py --container NAME` watches the real
#      MemAvailable from outside and stops the container if the 8 GB system floor is reached
#      (e.g. a non-torch CUDA allocation that bypasses the allocator cap).
#
# CPU and GPU COMPUTE are not limited here — no --cpus, no --gpus-per-run throttle. Only memory.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${WONDER_WORK:-$HOME/wonderland_work}"
image="${WONDER_IMAGE:-wonderland-runtime:24.04}"
mem="${WONDER_MEM:-20g}"
name="${WONDER_NAME:-wonderland-run}"
floor_gb="${WONDER_FLOOR_GB:-8}"

gpu=0
check=0
plan=0
out_mounts=()
cmd=()

usage() {
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
}

build() {
  docker build \
    --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
    -t "$image" -f "$repo/deploy/Dockerfile.wonder" "$repo/deploy"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) build; exit 0 ;;
    --check) check=1; shift ;;
    --plan) plan=1; shift ;;
    --gpu) gpu=1; shift ;;
    --name) [[ -n "${2:-}" ]] || { echo "--name needs a value" >&2; exit 2; }; name="$2"; shift 2 ;;
    --out)
      [[ -n "${2:-}" ]] || { echo "--out needs HOST[:CONTAINER]" >&2; exit 2; }
      src="${2%%:*}"
      dst="/out"
      [[ "$2" == *:* ]] && dst="${2#*:}"
      [[ -n "$src" && -n "$dst" ]] || { echo "--out: empty path in $2" >&2; exit 2; }
      out_mounts+=(-v "$src:$dst")
      shift 2
      ;;
    --) shift; cmd=("$@"); break ;;
    -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *) echo "commands go after --: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ $check -eq 1 ]]; then
  echo "image:      $image ($(docker image inspect -f '{{.Id}}' "$image" 2>/dev/null | cut -c1-19 || echo 'not built'))"
  total_gb=$(awk '/^MemTotal:/{printf "%.1f", $2/1048576}' /proc/meminfo)
  qwen_frac=$(grep -oE 'gpu-memory-utilization[[:space:]]+[0-9.]+' "$HOME/vllm-qwen-serve.sh" 2>/dev/null | awk '{print $2}' | head -1)
  qwen_frac="${qwen_frac:-0.70}"
  cap_gb="${mem%g}"
  python3 - "$total_gb" "$qwen_frac" "$cap_gb" "$floor_gb" <<'PY'
import sys
total, frac, cap, floor = (float(x) for x in sys.argv[1:5])
qwen = frac * total
print(f"container cap: {cap:.0f} GB per job  (--memory == --memory-swap: no swap escape)")
print(f"budget:        Qwen {qwen:.1f} + project {cap:.0f} + system {floor:.0f} = {qwen+cap+floor:.1f} of {total:.1f} GB -> {total-qwen-cap-floor:+.1f} GB slack")
print(f"GPU inside:    allocator capped so device + host usage + reserve <= {cap:.0f} GB")
print(f"floor guard:   host MemAvailable < {floor:.0f} GB -> the container is stopped")
PY
  echo "compute:    unlimited (no --cpus / no GPU throttle; only memory is capped)"
  echo "repo:       $repo -> /repo (rw)"
  echo "work data:  $work -> /work (ro)"
  echo "host free:  $(awk '/MemAvailable/{printf "%.0f GB", $2/1048576}' /proc/meminfo)"
  if docker info 2>/dev/null | grep -qi "cdi spec"; then
    echo "gpu:        available via --gpu (nvidia device requests supported)"
  else
    echo "gpu:        unknown — this daemon does not advertise CDI; --gpu may fail"
  fi
  exit 0
fi

if [[ ${#cmd[@]} -eq 0 ]]; then
  usage
  exit 2
fi

docker_argv=(
  docker run --rm --init
  --name "$name"
  --memory="$mem" --memory-swap="$mem"
  --pids-limit 1024
  --user "$(id -u):$(id -g)"
  --workdir /repo
  -e HOME=/tmp
  -e PYTHONPATH=/repo
)
# An EMPTY CUDA_VISIBLE_DEVICES means "no GPU" to torch, so it must be omitted, not passed.
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && docker_argv+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
[[ $gpu -eq 1 ]] && docker_argv+=(--gpus all)
docker_argv+=(-v "$repo:/repo" -v "$work:/work:ro")
[[ ${#out_mounts[@]} -gt 0 ]] && docker_argv+=("${out_mounts[@]}")
# shellcheck disable=SC2206  # deliberate: WONDER_EXTRA_ARGS is an ARGUMENT LIST, documented as such
[[ -n "${WONDER_EXTRA_ARGS:-}" ]] && docker_argv+=(${WONDER_EXTRA_ARGS})
docker_argv+=("$image" "${cmd[@]}")

if [[ $plan -eq 1 ]]; then
  printf '%s\n' "${docker_argv[@]}"
  exit 0
fi

docker image inspect "$image" >/dev/null 2>&1 || build

# ---- GPU co-tenant preflight (DEC-451) ------------------------------------------------------
# Measured 2026-10-30: a GPU job started while vLLM was still LOADING derived its budget from
# the transient free memory (40 GB), vLLM then claimed that memory back, MemAvailable fell to
# 6 GiB, this project's guard stopped the trainer -- and vLLM's own load guard (6 GiB floor)
# stopped the LLM. The host was fine, but the sequencing was wrong. A serving model's footprint
# is stable; a loading model's is not. So: if vLLM is up, wait until it answers /health, then
# require the load to have settled before starting anything that allocates GPU memory.
if [[ $gpu -eq 1 ]] && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "vllm-qwen"; then
  wait_s="${WONDER_WAIT_S:-1200}"
  # The port lives in the serve script (one definition); 8000 is its documented fallback.
  vport=$(grep -oE -- '--port[[:space:]]+[0-9]+' "$HOME/vllm-qwen-serve.sh" 2>/dev/null | awk '{print $2}' | head -1)
  vport="${vport:-8000}"
  echo "waiting for vllm-qwen to finish loading (max ${wait_s}s)..." >&2
  ready=0
  for _ in $(seq 1 "$wait_s"); do
    if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:${vport}/health" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ $ready -ne 1 ]]; then
    echo "REFUSING to start: vllm-qwen did not become healthy within ${wait_s}s; a GPU job" >&2
    echo "started during its load can push the host under the LLM's own floor (DEC-451)." >&2
    echo "Check: docker logs --tail 50 vllm-qwen" >&2
    exit 2
  fi
  # health is the model answering; settle = its footprint is no longer moving.
  prev="$(awk '/MemAvailable/{printf "%d", $2}' /proc/meminfo)"
  sleep 5
  cur="$(awk '/MemAvailable/{printf "%d", $2}' /proc/meminfo)"
  drift=$((prev > cur ? prev - cur : cur - prev))
  if [[ $drift -gt 1048576 ]]; then  # >1 GiB in 5 s: still growing, wait one more window
    sleep 5
    cur="$(awk '/MemAvailable/{printf "%d", $2}' /proc/meminfo)"
  fi
  echo "vllm-qwen serving; host free $((cur / 1048576)) GB" >&2
fi

guard_pid=""
cleanup() {
  if [[ -n "$guard_pid" ]]; then
    kill "$guard_pid" 2>/dev/null || true
    wait "$guard_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ $gpu -eq 1 ]]; then
  pidfile="/tmp/opencode/wonder_container_${name}.pid"
  guard_log="/tmp/opencode/wonder_container_${name}.guard.log"
  mkdir -p /tmp/opencode
  "$repo/.venv/bin/python" "$repo/scripts/gpu_watchdog.py" \
    --floor-gb "$floor_gb" --interval 2 --outside-slice ignore \
    --container "$name" --pidfile "$pidfile" >"$guard_log" 2>&1 &
  guard_pid=$!
  sleep 1
  if ! kill -0 "$guard_pid" 2>/dev/null; then
    echo "guard failed to start: $guard_log" >&2
    cat "$guard_log" >&2 || true
    exit 2
  fi
fi

set +e
"${docker_argv[@]}"
rc=$?
set -e
exit "$rc"

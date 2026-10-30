#!/usr/bin/env bash
# Run a command inside the project's memory-capped container (DEC-442).
#
#   scripts/wonder_container.sh -- .venv/bin/python scripts/measure_judge_law.py --legs rl
#   scripts/wonder_container.sh --build          # (re)build the image only
#   scripts/wonder_container.sh --check          # show the cap and the last run's peak
#
# The point: the machine is shared with an LLM server, and a runaway job has taken the host down.
# `--memory` is enforced by the kernel on the whole container, so nothing started here can take
# the LLM server's memory. The ceiling is 24 GB by default (the operator's 20-30 GB band) and is
# the ONLY place the number appears — override with WONDER_MEM=20g.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${WONDER_WORK:-$HOME/wonderland_work}"
image="${WONDER_IMAGE:-wonderland-runtime:24.04}"
mem="${WONDER_MEM:-24g}"
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
  echo "memory cap: $mem  (WONDER_MEM overrides; the operator's band is 20-30 GB)"
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

#!/usr/bin/env bash
# Install the project's memory boundary (DEC-440) from the repo, so it is reproducible.
#
#   scripts/install_memory_slice.sh            # install / update
#   scripts/install_memory_slice.sh --check    # report, change nothing
#
# The ceiling is DERIVED: the smaller of half of physical RAM and the operator's 30 GB band top,
# because the machine also runs an LLM server that must never be starved. Override with
# MEMORY_MAX=24G.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
dest="$unit_dir/wonderland.slice"

total_mb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
half_gb=$(( total_mb / 1024 / 1024 / 2 ))
# The ceiling is the project's BUDGET, derived from the machine's own numbers: the box has one
# 121.6 GB unified pool, Qwen reserves 0.70 of it, and the operator keeps 8 GB for the system.
# That leaves 28.5 GB, and 20 is what keeps ~8.5 GB of slack. It is the same number the container
# uses, so a job is bounded identically whether it runs in the container or is launched directly.
band_top_gb=20
derived_gb=$(( half_gb < band_top_gb ? half_gb : band_top_gb ))
cap="${MEMORY_MAX:-${derived_gb}G}"

render() {
  sed -e "s|__CAP__|$cap|g" \
      -e "s|__CAP_DOC__|Ceiling: $cap = half of ${total_mb} MB physical RAM (derived, not chosen).|g" \
      -e "s|@REPO@|$repo|g" \
      "$repo/deploy/wonderland.slice.in"
}

if [[ "${1:-}" == "--check" ]]; then
  echo "would install $dest with MemoryMax=$cap (derived from ${total_mb} MB RAM)"
  systemctl --user show wonderland.slice -p MemoryMax --value 2>/dev/null | sed 's/^/current MemoryMax: /'
  exit 0
fi

mkdir -p "$unit_dir"
render > "$dest"
systemctl --user daemon-reload
echo "installed $dest"
systemctl --user show wonderland.slice -p MemoryMax -p MemorySwapMax -p MemoryAccounting

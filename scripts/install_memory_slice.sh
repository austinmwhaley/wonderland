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
# The operator's band for the project is 20-30 GB: the machine is shared with an LLM server, and
# the ceiling must leave it room no matter what this repo does. So the cap is the SMALLER of half
# of RAM and the band's top. 30 GB of 121 GB leaves the LLM server ~90 GB.
band_top_gb=30
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

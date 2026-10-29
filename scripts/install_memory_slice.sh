#!/usr/bin/env bash
# Install the project's memory boundary (DEC-440) from the repo, so it is reproducible.
#
#   scripts/install_memory_slice.sh            # install / update
#   scripts/install_memory_slice.sh --check    # report, change nothing
#
# The ceiling is DERIVED, not typed: half of physical RAM, which leaves the other half for the
# LLM server, the desktop and the operator's 8 GB system floor. Override with MEMORY_MAX=40G.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
dest="$unit_dir/wonderland.slice"

total_mb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
derived_gb=$(( total_mb / 1024 / 1024 / 2 ))
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

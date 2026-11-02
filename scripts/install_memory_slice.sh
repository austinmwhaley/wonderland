#!/usr/bin/env bash
# Install the project's memory boundary (DEC-440) from the repo, so it is reproducible.
#
#   scripts/install_memory_slice.sh            # install / update
#   scripts/install_memory_slice.sh --check    # report, change nothing
#
# The ceiling is DERIVED: the smaller of half of physical RAM and the operator's 32 GB band top
# (2026-11-02: "This project should understand that its max is 32GB"), because the machine also
# runs an LLM server that must never be starved. Override with MEMORY_MAX=24G.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
dest="$unit_dir/wonderland.slice"

total_mb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
half_gb=$(( total_mb / 1024 / 1024 / 2 ))
# The ceiling is the project's BUDGET, derived from the machine's own numbers: the box has one
# 121.6 GB unified pool shared with the LLM server and the OS. The band top is the operator's
# 32G working budget (2026-11-02 directive: "This project should understand that its max is
# 32GB"; it supersedes the 2026-11-01 48G band). Measured 2026-11-02 with the server resident:
# `torch.cuda.mem_get_info()` reports 31.84 GiB free of 121.63, so the 48G band assumed memory
# that is not there; S1/S2's post-cap batch multiplier asked for ~91 GB (DEC-561). The GPU
# shares this pool, so 32G is also the GPU working max. The gpu_budget seatbelt
# (looking_glass/gpu_budget.py) caps each job's DEVICE pool below this; the slice is the
# CPU/host containment boundary (the kernel kills inside it, never outside), not the GPU cap.
# History: the 20G band top was measured binding on 2026-11-01 (a B=64 DINO run OOM-killed
# inside this slice while the seatbelt — which had read the container-root cgroup file, 'max'
# on this host — saw 44.9 GB free; gpu_budget now walks /proc/self/cgroup instead).
band_top_gb=32
derived_gb=$(( half_gb < band_top_gb ? half_gb : band_top_gb ))
cap="${MEMORY_MAX:-${derived_gb}G}"
cap_doc="Ceiling: ${derived_gb}G = min(half of ${total_mb} MB physical RAM, the operator's ${band_top_gb}G band top, 2026-11-02). GPU allocations are NOT charged to this cgroup; the per-process seatbelt in looking_glass/gpu_budget.py is their cap."

render() {
  sed -e "s|__CAP__|$cap|g" \
      -e "s|__CAP_DOC__|$cap_doc|g" \
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

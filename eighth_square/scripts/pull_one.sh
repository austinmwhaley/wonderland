#!/bin/bash
set -u
remote="$1"; localdir="$2"; want="$3"
mkdir -p "$localdir"
for uri in $(gio list -u "$remote" 2>&1); do
  info=$(timeout 10 gio info -a standard::display-name,standard::type "$uri" 2>&1)
  disp=$(echo "$info" | grep "display name:" | sed 's/.*display name: //')
  typ=$(echo "$info" | grep "^type:" | sed 's/.*type: //')
  [ -z "$disp" ] && continue
  case "$disp" in __pycache__|".venv"|".git"|checkpoints|data|results|tracking|visualization|experiments|OFFSET|".hypothesis"|".pytest_cache") continue;; esac
  if [ "$typ" = "directory" ]; then
    bash ~/pull_one.sh "$uri" "$localdir/$disp" "$want"
  else
    case "$want" in
      py) case "$disp" in *.py) timeout 20 gio copy "$uri" "$localdir/$disp" 2>&1 | head -n 1;; esac;;
      json) case "$disp" in *.json) timeout 20 gio copy "$uri" "$localdir/$disp" 2>&1 | head -n 1;; esac;;
    esac
  fi
done

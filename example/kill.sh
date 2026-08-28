#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

patterns=(
  "${ROOT_DIR}/example/main.py"
  "example/main.py"
  "${ROOT_DIR}/scripts/run_batch_scene_tracking.py"
  "scripts/run_batch_scene_tracking.py"
)

declare -A seen=()
pids=()

collect_children() {
  local parent="$1"
  local child
  while read -r child; do
    [[ -z "${child}" ]] && continue
    if [[ -z "${seen[${child}]:-}" ]]; then
      seen["${child}"]=1
      pids+=("${child}")
      collect_children "${child}"
    fi
  done < <(pgrep -P "${parent}" || true)
}

while read -r pid args; do
  [[ -z "${pid}" ]] && continue
  [[ "${pid}" == "$$" ]] && continue
  for pattern in "${patterns[@]}"; do
    if [[ "${args}" == *"${pattern}"* ]]; then
      if [[ -z "${seen[${pid}]:-}" ]]; then
        seen["${pid}"]=1
        pids+=("${pid}")
        collect_children "${pid}"
      fi
      break
    fi
  done
done < <(ps -eo pid=,args=)

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No SceneFlowTracker run.sh processes found."
  exit 0
fi

echo "Stopping SceneFlowTracker processes: ${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null || true
sleep 3

alive=()
for pid in "${pids[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    alive+=("${pid}")
  fi
done

if [[ "${#alive[@]}" -gt 0 ]]; then
  echo "Force killing remaining processes: ${alive[*]}"
  kill -KILL "${alive[@]}" 2>/dev/null || true
fi

echo "Done."

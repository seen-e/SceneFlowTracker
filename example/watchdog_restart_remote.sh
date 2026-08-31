#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-${ROOT_DIR}/example/run.sh}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/example/logs}"
MOUNT_CHECK_PATH="${MOUNT_CHECK_PATH:-/mnt/data/chachaxu/dataset}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MOUNT_RECOVERY_TIMEOUT_SECONDS="${MOUNT_RECOVERY_TIMEOUT_SECONDS:-900}"
MOUNT_CHECK_TIMEOUT_SECONDS="${MOUNT_CHECK_TIMEOUT_SECONDS:-10}"
KILL_PATTERN="${KILL_PATTERN:-[p]ython}"
ONCE=0

for arg in "$@"; do
  case "${arg}" in
    --once) ONCE=1 ;;
    -h|--help)
      cat <<USAGE
Usage: watchdog_restart_remote.sh [--once]

Environment:
  PROJECT_DIR                         SceneFlowTracker 根目录，默认自动推断
  MOUNT_CHECK_PATH                    挂载检查路径，默认 /mnt/data/chachaxu/dataset
  CHECK_INTERVAL_SECONDS              循环检查间隔，默认 60
  MOUNT_RECOVERY_TIMEOUT_SECONDS      挂载恢复等待时间，默认 900
  MOUNT_CHECK_TIMEOUT_SECONDS         单次挂载检查超时，默认 10
  KILL_PATTERN                        挂载断开时 pkill -f 的模式，默认 python
USAGE
      exit 0
      ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${LOG_DIR}/watchdog_restart.log}"

log() {
  local msg="$*"
  printf '%s %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${WATCHDOG_LOG}"
}

mount_ok() {
  timeout "${MOUNT_CHECK_TIMEOUT_SECONDS}" bash -lc \
    "test -d '${MOUNT_CHECK_PATH}' && stat '${MOUNT_CHECK_PATH}' >/dev/null" \
    >/dev/null 2>&1
}

find_main_pids() {
  pgrep -f "${ROOT_DIR}/example/main.py|example/main.py --manifest-path" || true
}

task_running() {
  [[ -n "$(find_main_pids)" ]]
}

kill_python_tasks() {
  log "mount check failed; running: pkill -f ${KILL_PATTERN}"
  pkill -TERM -f "${KILL_PATTERN}" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "${KILL_PATTERN}" 2>/dev/null || true
}

wait_for_mount() {
  local deadline=$((SECONDS + MOUNT_RECOVERY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if mount_ok; then
      return 0
    fi
    log "waiting for mount recovery: ${MOUNT_CHECK_PATH}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
  return 1
}

start_task() {
  if task_running; then
    log "task already running; skip restart"
    return 0
  fi
  local ts stdout pid
  ts="$(date +%Y%m%d_%H%M%S)"
  stdout="${LOG_DIR}/run_stdout_${ts}_watchdog.log"
  (
    cd "${ROOT_DIR}"
    nohup bash "${RUN_SCRIPT}" > "${stdout}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${LOG_DIR}/latest.pid"
    echo "${stdout}" > "${LOG_DIR}/latest_stdout_log.txt"
    log "started task pid=${pid} stdout=${stdout}"
  )
}

check_once() {
  if ! mount_ok; then
    kill_python_tasks
    if wait_for_mount; then
      log "mount recovered; restarting task"
      start_task
    else
      log "mount did not recover within ${MOUNT_RECOVERY_TIMEOUT_SECONDS}s; task not restarted"
      return 1
    fi
    return 0
  fi

  if task_running; then
    log "mount ok; task running"
  else
    log "mount ok; task not running; starting task"
    start_task
  fi
}

if (( ONCE )); then
  check_once
  exit $?
fi

log "watchdog started root=${ROOT_DIR} mount=${MOUNT_CHECK_PATH} interval=${CHECK_INTERVAL_SECONDS}s kill_pattern=${KILL_PATTERN}"
while true; do
  check_once || true
  sleep "${CHECK_INTERVAL_SECONDS}"
done

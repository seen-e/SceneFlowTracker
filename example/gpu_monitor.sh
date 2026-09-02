#!/usr/bin/env bash
set -u
interval="${GPU_MONITOR_INTERVAL:-60}"
threshold="${GPU_IDLE_THRESHOLD:-20}"
max_idle_count="${GPU_IDLE_MINUTES:-10}"
idle_count=0
while true; do
  ts=$(date "+%Y-%m-%d %H:%M:%S")
  row=$(nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(printf "%s" "$row" | awk -F, '{gsub(/ /,"",$5); print $5}')
  if [ -z "$util" ]; then
    echo "$ts ERROR nvidia-smi unavailable"
    sleep "$interval"
    continue
  fi
  if [ "$util" -lt "$threshold" ]; then
    idle_count=$((idle_count + 1))
  else
    idle_count=0
  fi
  echo "$ts GPU $row idle_minutes=$((idle_count * interval / 60))"
  if [ "$idle_count" -ge "$max_idle_count" ]; then
    echo "$ts WARN gpu_util_below_${threshold}_percent_for_$((max_idle_count * interval / 60))min"
    idle_count=0
  fi
  sleep "$interval"
done

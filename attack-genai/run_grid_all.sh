#!/usr/bin/env bash
# Orchestrate the whole 99-config grid with N work-stealing streams until ALL done.
# Reaps stale claims (claimed but no matching .DONE and no running train for it),
# relaunches streams that exit early, stops when 99/99 .DONE.
set -u
ROOT=/usa/taikun/rl-attack/rl_atk/attack-genai
RUNDIR=$ROOT/grid_runs
mkdir -p "$RUNDIR"

# Stream endpoints: "STREAMID URL" — add the A40 here later if it comes online.
STREAMS=(
  "0 http://localhost:8000/v1"   # infochain guard (tunnel) — 1 stream per guard
  "1 http://localhost:8001/v1"   # infodeep GPU1 guard (local, fast) — 1 stream per guard
)
# LESSON (2026-05-31): do NOT point 2 streams at one guard — concurrent 16-prompt
# batches OOM the 24GB guard (275MiB free) -> HTTP 500 -> prob=0.5 -> CORRUPTED reward.
# Server now serializes generation (_GEN_LOCK) + client retries on 500 (no silent 0.5),
# but 1-stream-per-guard is still the right design (no queueing). Each new guard = +1 stream.

# Config index range this machine handles. infodeep takes the SMALLER share (0-39)
# because the A40 (single fast LOCAL guard, no tunnel) is faster — it takes 40-98.
export CONFIG_LO="${CONFIG_LO:-0}"
export CONFIG_HI="${CONFIG_HI:-39}"

launch_stream () { # id url
  local id="$1" url="$2"
  CONFIG_LO="$CONFIG_LO" CONFIG_HI="$CONFIG_HI" setsid nohup bash "$ROOT/run_grid_stream.sh" "$id" "$url" \
    > "$RUNDIR/stream${id}.log" 2>&1 < /dev/null &
  echo "  launched stream $id -> $url (pid $!)  [configs $CONFIG_LO-$CONFIG_HI]"
}

reap_stale_claims () {
  # a claim is stale if: no .DONE for it AND no python training proc has its tag in cmdline
  for c in "$RUNDIR"/*.claim; do
    [ -d "$c" ] || continue
    local tag; tag=$(basename "$c" .claim)
    [ -f "$RUNDIR/$tag.DONE" ] && continue                 # finished; keep claim as marker
    # parse tag A_a1.0_K3 -> reward A alpha 1.0 K 3
    local R A K; R=${tag%%_*}; A=$(echo "$tag"|sed -E 's/^._a([0-9.]+)_K.*/\1/'); K=${tag##*K}
    if ! pgrep -af "reward_type $R" 2>/dev/null | grep -q -- "--alpha $A .*--reward_type $R" ; then
      # not currently running -> release
      echo "  reaped stale claim $tag"; rm -rf "$c"
    fi
  done
}

# This machine's target = number of configs in its [CONFIG_LO, CONFIG_HI] range.
MY_TARGET=$(( CONFIG_HI - CONFIG_LO + 1 ))
while :; do
  done_n=$(ls "$RUNDIR"/*.DONE 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$MY_TARGET" ]; then echo "MY_SHARE_DONE ($done_n/$MY_TARGET configs $CONFIG_LO-$CONFIG_HI) $(date +%F_%T)"; break; fi
  reap_stale_claims
  # ensure each stream is alive
  for s in "${STREAMS[@]}"; do
    id=${s%% *}; url=${s#* }
    if ! pgrep -af "run_grid_stream.sh $id " >/dev/null 2>&1; then
      echo "[orch $(date +%T)] stream $id not running (done=$done_n/99); launching"
      launch_stream "$id" "$url"
    fi
  done
  sleep 120
done

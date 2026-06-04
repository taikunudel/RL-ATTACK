#!/usr/bin/env bash
# A40 orchestrator (HOST-side). One work-stealing stream into the LOCAL guard,
# wrapped in the apptainer container. Relaunches the stream if it exits early;
# reaps stale claims; stops when this node's share (CONFIG_LO..CONFIG_HI) is all .DONE.
set +u   # VALET's vpkg_require references unset vars; nounset would kill the script
source /etc/profile.d/valet.sh 2>/dev/null
vpkg_require apptainer/1.4.1 >/dev/null 2>&1
set -u
export APPTAINER_CACHEDIR=/tmp/a40/apptainer/cache APPTAINER_TMPDIR=/tmp/a40/apptainer/tmp
# evaluation_attacker_genai.py:15 opens a diskcache at the infodeep path /usa/taikun/rl-attack/rl_atk/attack-genai;
# bind a writable A40 dir there inside the container so the byte-identical code runs (cold cache = identical results).
mkdir -p /tmp/a40/diskcache /tmp/a40/ctr/usa/taikun/rl-attack/rl_atk/attack-genai 2>/dev/null
export APPTAINER_BIND="/tmp/a40/diskcache:/usa/taikun/rl-attack/rl_atk/attack-genai"
CTR=/tmp/a40/ctr
ROOT=/work/weiqian_stat/taikun/a40_rlatk
RUNDIR=$ROOT/grid_runs
mkdir -p "$RUNDIR"

STREAMS=( "0 http://localhost:8000/v1" )       # single local guard, single stream
export CONFIG_LO="${CONFIG_LO:-40}"
export CONFIG_HI="${CONFIG_HI:-98}"

launch_stream () { # id url
  local id="$1" url="$2"
  APPTAINERENV_CONFIG_LO="$CONFIG_LO" APPTAINERENV_CONFIG_HI="$CONFIG_HI" \
  CONFIG_LO="$CONFIG_LO" CONFIG_HI="$CONFIG_HI" \
  setsid nohup apptainer exec --nv -B /tmp/a40,/work "$CTR" \
    bash "$ROOT/run_a40_stream.sh" "$id" "$url" \
    > "$RUNDIR/stream${id}.log" 2>&1 < /dev/null &
  echo "  launched stream $id -> $url (pid $!)  [configs $CONFIG_LO-$CONFIG_HI]"
}

reap_stale_claims () {
  for c in "$RUNDIR"/*.claim; do
    [ -d "$c" ] || continue
    local tag; tag=$(basename "$c" .claim)
    [ -f "$RUNDIR/$tag.DONE" ] && continue
    local R A K; R=${tag%%_*}; A=$(echo "$tag"|sed -E 's/^._a([0-9.]+)_K.*/\1/'); K=${tag##*K}
    if ! pgrep -af "reward_type $R" 2>/dev/null | grep -q -- "--alpha $A .*--reward_type $R" ; then
      echo "  reaped stale claim $tag"; rm -rf "$c"
    fi
  done
}

MY_TARGET=$(( CONFIG_HI - CONFIG_LO + 1 ))
echo "[orch $(date +%F_%T)] A40 share = configs $CONFIG_LO-$CONFIG_HI ($MY_TARGET configs)"
while :; do
  done_n=$(ls "$RUNDIR"/*.DONE 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$MY_TARGET" ]; then echo "MY_SHARE_DONE ($done_n/$MY_TARGET configs $CONFIG_LO-$CONFIG_HI) $(date +%F_%T)"; break; fi
  reap_stale_claims
  for s in "${STREAMS[@]}"; do
    id=${s%% *}; url=${s#* }
    if ! pgrep -af "run_a40_stream.sh $id " >/dev/null 2>&1; then
      echo "[orch $(date +%T)] stream $id not running (done=$done_n/$MY_TARGET); launching"
      launch_stream "$id" "$url"
    fi
  done
  sleep 120
done

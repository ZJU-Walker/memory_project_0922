#!/usr/bin/env bash
# Generic task1 gate v3 = v2 + SIDECAR/MANIFEST env (lead30 line: battery AND verdict read the same sidecar). Run ON iris-hgx-2 (job 17329416, H200). Env: ACFG AEXP BCFG BEXP ENVVAR [ASTEPS] [MIN_RECALL=5] [SIDECAR] [MANIFEST] [PARALLEL] [NEED_MIB].
# For each stage-A checkpoint in ASTEPS: wait until finalized, three-mode battery (oracle, oracle_evidence, self),
# verdict; PASS = oracle_evidence RECALL (first closing step) >= MIN_RECALL/6 -> ssh hgx-1 launch_B_generic_hgx1.sh,
# then the three-mode battery on every B checkpoint. Log: v6/logs/gate_task1_hgx2.log. Script file on purpose.
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; diag=$root/v6/diagnostics
ckroot=$root/v6/checkpoints; log=$logs/gate_task1_hgx2.log
: "${ACFG:?}" "${AEXP:?}" "${BCFG:?}" "${BEXP:?}" "${ENVVAR:?}"
ASTEPS=${ASTEPS:-"250 500 750 1000 1250 1500 1750 2000"}; MIN_RECALL=${MIN_RECALL:-5}
SIDECAR=${SIDECAR:-$cv6/task1/task1v6_v5_subtask_labels_v1.json}; MANIFEST=${MANIFEST:-$cv6/task1/task1v6_episode_manifest_v1.json}; export SIDECAR MANIFEST
SSH="ssh -i /iris/u/kewalk/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
say() { echo "$(date '+%m/%d %H:%M') [hgx-2] $*" >> "$log"; }
finalized() { [ -e "$ckroot/$1/$2/$3/params" ] && grep -q "\[step=$3\] CheckpointManager Save Finalize is done on all hosts" "$logs/train_$2.log" 2>/dev/null; }
exited() { [ "$(tail -1 $logs/train_$1_status.log 2>/dev/null | cut -c1-5)" = "exit=" ]; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
wait_free() {  # the H200 is shared with the user's other evals/servers (job 17329416): wait for $1 MiB free, log once
  local need=$1 waited=0
  while [ "$(free_mib)" -lt "$need" ]; do [ $waited -eq 0 ] && say "waiting for $need MiB free on the H200 (now $(free_mib) MiB; other jobs' evals/servers)"; waited=1; sleep 60; done
  [ $waited -eq 1 ] && say "H200 has $(free_mib) MiB free -> continuing"
}
battery() {  # modes run SEQUENTIALLY (one ~20 GB process at a time) unless PARALLEL=1
  for m in oracle oracle_evidence self; do
    wait_free ${NEED_MIB:-24000}
    if [ "${PARALLEL:-0}" = "1" ]; then MODES=$m JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_v2_hgx1.sh "$1" "$2" "$3" &
    else MODES=$m JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_v2_hgx1.sh "$1" "$2" "$3"; fi
  done
  wait
  (cd $root/openpi && .venv/bin/python scripts/task1_battery_verdict.py --sidecar "$SIDECAR" "$diag/videos_$2_$3")
}
say "gate v3 armed on $(hostname): $AEXP -> $BEXP; PASS = oracle_evidence recall >= $MIN_RECALL/6; A steps: $ASTEPS; sidecar $(basename $SIDECAR)"
passed=""
for step in $ASTEPS; do
  until finalized $ACFG $AEXP $step; do
    if exited $AEXP && ! finalized $ACFG $AEXP $step; then say "$AEXP exited before ckpt $step ($(tail -1 $logs/train_${AEXP}_status.log)); A loop ends"; step=""; break; fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "$AEXP ckpt $step finalized -> battery (oracle, oracle_evidence, self)"
  verdict=$(battery $ACFG $AEXP $step); echo "$verdict" >> "$log"
  n=$(echo "$verdict" | grep -o "RECALL (first closing step) — oracle [0-9]/6, oracle_evidence [0-9]" | grep -o "[0-9]$")
  if [ -n "$n" ] && [ "$n" -ge "$MIN_RECALL" ]; then passed=$step; break; fi
  say "$AEXP ckpt $step: oracle_evidence recall ${n:-?}/6 < $MIN_RECALL -> next checkpoint"
done
if [ -z "$passed" ]; then say "no $AEXP checkpoint passed the recall gate; gate done"; exit 0; fi
say "$AEXP ckpt $passed PASSED -> $BEXP from it (hgx-1)"
$SSH iris-hgx-1 "bash $cv6/task1/launch_B_generic_hgx1.sh $ACFG $AEXP $passed $BCFG $BEXP $ENVVAR" >> $logs/gate_ssh.out 2>&1 || say "ssh to hgx-1 FAILED (launch by hand)"
for step in 250 500 750 1000 1250 1500 1750 2000; do
  until finalized $BCFG $BEXP $step; do
    if exited $BEXP && ! finalized $BCFG $BEXP $step; then say "$BEXP exited before ckpt $step; B loop ends"; step=""; break; fi
    sleep 60
  done
  [ -z "$step" ] && break
  sleep 30
  say "$BEXP ckpt $step finalized -> battery (oracle, oracle_evidence, self)"
  battery $BCFG $BEXP $step >> "$log"
done
say "gate done"

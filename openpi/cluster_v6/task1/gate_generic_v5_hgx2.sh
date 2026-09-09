#!/usr/bin/env bash
# Generic task1 gate v5 = v4 + BATTERY_MODES env (default all three; "oracle_evidence self" leaves H200 room for a policy server). v4 was: "launch first" (user 2026-09-09 11:00: "if A reaches 250 directly start training B3"): as soon
# as stage-A checkpoint LAUNCH_B_AT is finalized, launch stage B from it (ssh hgx-1 launch_B_generic_hgx1.sh, which
# stops A, protects keep_<step>, starts B), THEN run the A battery (information only) and the battery on every B
# checkpoint. Battery and verdict read SIDECAR/MANIFEST (lead30 line). Run ON iris-hgx-2 (job 17329416, H200).
# Env: ACFG AEXP BCFG BEXP ENVVAR [LAUNCH_B_AT=250] [SIDECAR] [MANIFEST] [PARALLEL] [NEED_MIB] [KRB5CCNAME]
export HOME=/iris/u/kewalk
export KRB5CCNAME=${KRB5CCNAME:-/iris/u/kewalk/.krb5cc_claude_gate}  # the ssh hand-off needs a ticket (NFS copy of the workstation cache)
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6; diag=$root/v6/diagnostics
ckroot=$root/v6/checkpoints; log=$logs/gate_task1_hgx2.log
: "${ACFG:?}" "${AEXP:?}" "${BCFG:?}" "${BEXP:?}" "${ENVVAR:?}"
LAUNCH_B_AT=${LAUNCH_B_AT:-250}; BATTERY_MODES=${BATTERY_MODES:-"oracle oracle_evidence self"}
SIDECAR=${SIDECAR:-$cv6/task1/task1v6_v5_subtask_labels_v1.json}; MANIFEST=${MANIFEST:-$cv6/task1/task1v6_episode_manifest_v1.json}; export SIDECAR MANIFEST
SSH="ssh -i /iris/u/kewalk/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts -o ConnectTimeout=30 -o BatchMode=yes"
say() { echo "$(date '+%m/%d %H:%M') [hgx-2] $*" >> "$log"; }
finalized() { [ -e "$ckroot/$1/$2/$3/params" ] && grep -q "\[step=$3\] CheckpointManager Save Finalize is done on all hosts" "$logs/train_$2.log" 2>/dev/null; }
exited() { [ "$(tail -1 $logs/train_$1_status.log 2>/dev/null | cut -c1-5)" = "exit=" ]; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
wait_free() {
  local need=$1 waited=0
  while [ "$(free_mib)" -lt "$need" ]; do [ $waited -eq 0 ] && say "waiting for $need MiB free on the H200 (now $(free_mib) MiB)"; waited=1; sleep 60; done
  [ $waited -eq 1 ] && say "H200 has $(free_mib) MiB free -> continuing"
}
battery() {  # $1 cfg $2 exp $3 step-dir (a number or keep_<n>); modes in parallel when PARALLEL=1
  for m in $BATTERY_MODES; do
    wait_free ${NEED_MIB:-24000}
    if [ "${PARALLEL:-0}" = "1" ]; then MODES=$m JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_v2_hgx1.sh "$1" "$2" "$3" &
    else MODES=$m JOB=17329416 GPU=0 bash $cv6/task1/run_task1_evals_v2_hgx1.sh "$1" "$2" "$3"; fi
  done
  wait
  (cd $root/openpi && .venv/bin/python scripts/task1_battery_verdict.py --sidecar "$SIDECAR" "$diag/videos_$2_$3")
}
B_ONLY=${B_ONLY:-0}  # 1 = stage B already launched: skip the A section (never re-launch B), run only the B batteries
say "gate v5 armed on $(hostname): $AEXP ckpt $LAUNCH_B_AT -> $BEXP immediately (no gate), A battery afterwards; modes [$BATTERY_MODES]; sidecar $(basename $SIDECAR)"
if [ "$B_ONLY" != "1" ]; then
until finalized $ACFG $AEXP $LAUNCH_B_AT; do
  if exited $AEXP; then say "$AEXP exited before ckpt $LAUNCH_B_AT ($(tail -1 $logs/train_${AEXP}_status.log)); nothing launched"; exit 0; fi
  sleep 30
done
sleep 30
say "$AEXP ckpt $LAUNCH_B_AT finalized -> launching $BEXP from it (hgx-1)"
$SSH iris-hgx-1 "bash $cv6/task1/launch_B_generic_hgx1.sh $ACFG $AEXP $LAUNCH_B_AT $BCFG $BEXP $ENVVAR" >> $logs/gate_ssh.out 2>&1 || say "ssh to hgx-1 FAILED (launch by hand)"
sleep 20; say "hgx-1 says: $(tail -1 $logs/train_${BEXP}_status.log 2>/dev/null | cut -c1-120)"
say "$AEXP ckpt $LAUNCH_B_AT battery (information only, from keep_$LAUNCH_B_AT)"
battery $ACFG $AEXP keep_$LAUNCH_B_AT >> "$log"
fi
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

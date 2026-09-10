#!/usr/bin/env bash
# Run from the workstation (detached): wait until battery job $PREV has ended, then submit the step-1999 battery to
# iris9 and resubmit while it exits 7 (= landed on the ECC-broken L40S idx 6 of iris9). Up to $MAX attempts.
PREV=${PREV:-17359368}; MAX=${MAX:-8}
SSHO="-o BatchMode=yes -o IdentitiesOnly=yes -i /iris/u/kewalk/.ssh/id_ed25519 -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts"
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_xWOW6i  # hard-set: each Claude shell inherits a KRB5CCNAME that names a non-existent cache
log=/iris/u/kewalk/memory_project_v6/v6/logs/battery_1999_resubmit.log
say() { echo "$(date '+%m/%d %H:%M') $*" >> "$log"; }
sc() { timeout 60 ssh $SSHO sc "$@" 2>/dev/null | grep -v "afs\|pubkey"; }
say "waiting for job $PREV to end"
while [ -n "$(sc "squeue -h -j $PREV -o %T")" ]; do sleep 60; done
say "job $PREV ended"
for i in $(seq 1 $MAX); do
  jid=$(sc "cd /iris/u/kewalk/memory_project_v6/openpi && sbatch --parsable --export=ALL,STEPS=1999 --gres=gpu:l40s:1 --nodelist=iris9 cluster_v6/task1/battery_B6_sbatch.sh")
  say "attempt $i: submitted $jid"
  [ -z "$jid" ] && { sleep 60; continue; }
  # wait until it is out of PENDING and either finished the self-test or ended
  while true; do
    st=$(sc "squeue -h -j $jid -o %T"); out=/iris/u/kewalk/memory_project_v6/v6/logs/battery_sbatch-$jid.out
    if [ -z "$st" ]; then
      if grep -q "cannot use the allocated GPU" "$out" 2>/dev/null; then say "attempt $i: $jid hit the broken card, resubmitting"; sleep 20; break; fi
      say "attempt $i: $jid ended: $(tail -1 "$out" 2>/dev/null | cut -c1-120)"; exit 0
    fi
    if grep -q "jax ok" "$out" 2>/dev/null; then say "attempt $i: $jid runs on a good GPU ($(grep -oE 'on iris[0-9]+' "$out" | head -1)); done"; exit 0; fi
    sleep 30
  done
done
say "gave up after $MAX attempts"

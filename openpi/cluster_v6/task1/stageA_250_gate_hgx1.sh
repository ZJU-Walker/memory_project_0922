#!/usr/bin/env bash
# Stage-A step-250 gate (user 2026-09-08 21:09: "wait until stage A reach 250 and test and if it is good lets directly
# start from here for B; for B save every 250 but only keep every 500").
#   1. wait until checkpoint 250 of v6_task1A_20260908_r1 is finalized;
#   2. stop stage A (it holds all four GPUs; the battery needs them) — never the train_hs.py keep-alive;
#   3. development battery on ckpt 250: oracle writes on GPU 0 and self writes on GPU 1 in parallel;
#   4. GATE = oracle-write first decision ("open bin k") correct on >= 5 of the 6 development episodes:
#        PASS -> protect 250 as keep_250, launch stage B (pi05_yam_mem_v6_task1B, save 250 / keep 500) from it,
#                then the battery on every kept B checkpoint;
#        FAIL -> resume stage A from 250 (--resume) and run the battery on its later kept checkpoints.
# Log: v6/logs/gate_task1A_250.log. A script file on purpose (pgrep self-match).
export HOME=/iris/u/kewalk
root=/iris/u/kewalk/memory_project_v6; logs=$root/v6/logs; cv6=$root/openpi/cluster_v6
acfg=pi05_yam_mem_v6_task1A; aexp=v6_task1A_20260908_r1; bcfg=pi05_yam_mem_v6_task1B; bexp=v6_task1B_20260908_r1
ckroot=$root/v6/checkpoints; ck=$ckroot/$acfg/$aexp/250; log=$logs/gate_task1A_250.log; trainlog=$logs/train_${aexp}.log
pat_a="scripts/train.py pi05_yam_mem_v6_task1[A]|queue_task1A_hgx[1]|run_train_hgx1.sh pi05_yam_mem_v6_task1[A]"
say() { echo "$(date '+%m/%d %H:%M') $*" >> "$log"; }
say "gate armed on $(hostname), waiting for $ck"
until [ -e "$ck/params" ] && grep -q "\[step=250\] CheckpointManager Save Finalize is done on all hosts" "$trainlog" 2>/dev/null; do sleep 30; done
sleep 30
say "ckpt 250 finalized -> stopping stage A"
pids=$(pgrep -u kewalk -f "$pat_a"); [ -n "$pids" ] && kill -TERM $pids 2>/dev/null; sleep 20
left=$(pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task1[A]"); [ -n "$left" ] && { kill -KILL $left; sleep 10; }
until ! pgrep -u kewalk -f "scripts/train.py pi05_yam_mem_v6_task1[A]" >/dev/null; do sleep 5; done
sleep 20
say "stage A stopped; keep-alive 3743806 $([ -d /proc/3743806 ] && echo alive || echo GONE); gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
cp -r "$ck" "$ckroot/$acfg/$aexp/keep_250" && say "ckpt 250 protected as keep_250" || say "keep_250 COPY FAILED"
say "battery on 250: oracle (GPU 0) + self (GPU 1) in parallel"
MODES=oracle GPU=0 bash $cv6/task1/run_task1_evals_hgx1.sh $acfg $aexp 250 &
MODES=self GPU=1 bash $cv6/task1/run_task1_evals_hgx1.sh $acfg $aexp 250 &
wait
out=$root/v6/diagnostics/videos_${aexp}_250
verdict=$(cd $root/openpi && .venv/bin/python - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); dev = (12, 13, 26, 27, 67, 68)
rows, passed = [], 0
for mode in ("oracle", "self"):
    for ep in dev:
        f = out / f"ep{ep:02d}_{mode}.json"
        if not f.exists():
            rows.append(f"  {mode:6s} ep{ep:02d}: MISSING (see ep{ep:02d}_{mode}_run.log)"); continue
        s = json.loads(f.read_text())
        ok = bool(s.get("first_decision_correct"))
        passed += int(ok and mode == "oracle")
        rows.append(f"  {mode:6s} ep{ep:02d} {s['stable_id'].split('/')[-1]:14s} prompt={s['prompt']!r:20s} first={s.get('first_decision_pred')!r:14s} "
                    f"{'OK ' if ok else 'BAD'} decisions {s['decision_side_correct']}/{s['decision_steps']} "
                    f"evidence {s['evidence_pred_exact']}/{s['evidence_steps']} writes={s['writes']} bank={s['final_bank']}")
print("\n".join(rows))
print(f"GATE {'PASS' if passed >= 5 else 'FAIL'}: oracle first decision correct on {passed}/6 development episodes")
PY
)
echo "$verdict" >> "$log"
if echo "$verdict" | grep -q "^GATE PASS"; then
  say "-> stage B from keep_250 (save 250 / keep 500)"
  export OPENPI_V6_TASK1_A_PARAMS="v6/checkpoints/$acfg/$aexp/keep_250/params"
  JOB=17315830 GPUS=4 BATCH=8 bash $cv6/run_train_hgx1.sh $bcfg $bexp
  code=$(grep "^exit=" $logs/train_${bexp}_status.log | tail -1); say "stage B $code"
  if echo "$code" | grep -q "exit=0"; then
    for step in $(ls $ckroot/$bcfg/$bexp | grep -E '^[0-9]+$' | sort -n); do
      bash $cv6/task1/run_task1_evals_hgx1.sh $bcfg $bexp $step
      say "B battery $step: $(tail -1 $root/v6/diagnostics/videos_${bexp}_${step}/status.log)"
    done
  fi
else
  say "-> gate failed: resuming stage A from 250"
  JOB=17315830 GPUS=4 BATCH=8 bash $cv6/run_train_hgx1.sh $acfg $aexp
  code=$(grep "^exit=" $logs/train_${aexp}_status.log | tail -1); say "stage A (resumed) $code"
  if echo "$code" | grep -q "exit=0"; then
    for step in $(ls $ckroot/$acfg/$aexp | grep -E '^[0-9]+$' | sort -n); do
      [ "$step" -le 250 ] && continue
      bash $cv6/task1/run_task1_evals_hgx1.sh $acfg $aexp $step
      say "A battery $step: $(tail -1 $root/v6/diagnostics/videos_${aexp}_${step}/status.log)"
    done
  fi
fi
say "gate script done"

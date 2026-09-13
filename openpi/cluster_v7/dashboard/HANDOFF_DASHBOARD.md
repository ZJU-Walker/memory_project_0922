# Live training dashboard — how it was built (handoff for another session / project)

Built 2026-09-12 for the boba memory training (v7). Result: a private claude.ai Artifact page that shows run status,
loss curves with a metric switch, step timing, GPU/disk state, a run table and the tail of the training logs, and that
**updates itself every 10 minutes without a republish**. Everything here is plain Python 3 + hand-written HTML/SVG/JS;
no build tools, no chart library.

## 1. Pipeline (three files in this directory)

| file | role |
|---|---|
| `build_dashboard.py` | parses the training logs into one JSON object (`DATA`), writes `dash_data.json`, and prints the HTML page = `template.html` with `/*__DATA__*/` replaced by `const DATA = {...};` |
| `template.html` | the page: CSS tokens (light + dark), a small SVG line-chart function, `render(DATA)` that fills tiles / tables / charts / log panels, and a boot block that subscribes to the artifact database and re-renders on every push |
| `refresh.sh` | one refresh: ssh to the cluster for a GPU/job/disk snapshot (`gpu.json`), run the builder, leave `dash_data.json` + `boba_training_dashboard.html` next to the scripts |

Run once by hand:
```
bash refresh.sh                      # -> dash_data.json, boba_training_dashboard.html
```

## 2. What the builder parses (adapt these regexes to the new project's logs)

* `train_<exp>.log` — openpi `train.py` output:
  * `Step N: k=v, k=v, ...` every `log_interval` (100) updates; metric names may contain `/` (the builder keeps the last
    path component). Keys used: `ce_loss flow_loss loss grad_norm memory_grad_norm` plus the memory diagnostics
    (`diagnostic/v4_decision_ce_sum`, `v4_decision_count`, `v5_exact_decision_sum`, `v5_exact_evidence_sum`,
    `v5_evidence_count`, `v4_sem_commit_count`, `v5_prefill_sentence_count`, `v5_qk_cos_sum`, `v5_qk_count`, ...).
  * tqdm progress lines `HH:MM:SS.mmm [I] Progress on: 27.00it/251it rate:27.9s/it ...` -> wall clock per update
    (consecutive lines give seconds/update; gaps > 120 s are XLA compiles and are excluded from the ETA).
  * error signatures: `Traceback`, `error`, `CANCELLED`, `Killed`, `RESOURCE_EXHAUSTED`, `nan`.
* `train_<exp>_status.log` — written by `cluster_v7/run_train.sh`: `launch MM/DD HH:MM host=... job=... config=... exp=...
  batch=... accum=... mode=... code=<git sha>` and `exit=<code> MM/DD HH:MM`. Launch/exit pairs give the run state
  (queued / running / finished / stopped).
* checkpoint directories `<ckpt_base>/<config>/<exp>/<step>/` (numeric names = kept checkpoints).
* `gpu.json` from `refresh.sh`: `{"sampled", "h200":[{gpu,used_mib,total_mib,util}], "h100":[...], "disk_free_gb", "jobs"}`
  (nvidia-smi through `srun --jobid=<job> --overlap --gres=gpu:N`, `df`, `squeue`).

Runs are declared in `build_dashboard.py` as `(tag, config_name, exp_name, root)`; `total` = planned updates. Per-run
result: `{config, exp, steps:[{step, metric...}], progress:[{t, step, total}], status:[launch/exit events], ckpts:[...], total}`.
`compact_progress` subsamples long progress lists so the JSON stays well under the 256 KiB document limit (39 KB here).
`log_tail` produces the human-readable log panels (abbreviated Step lines, progress lines, errors, loader/sampler info).

## 3. The page (`template.html`)

* Sections: status sentence; tiles (stage, progress bar, seconds/update, ETA, free disk, GPU utilization); "runs at a
  glance" table; memory figure with a **metric switch** (buttons: CE / flow / total / exact match, choice kept in
  `localStorage`) drawing stage A and stage B on one axis; time-per-update chart; ablation charts; training-log panels;
  `<details>` folds for extra diagnostics and the recipe.
* Charts: `chart(series, opts)` returns an inline SVG (viewBox 640x280, "nice" ticks, grid, dashed reference lines,
  one dot at the latest value). Lines are stroke-only — **do not put `fill` on the series class for paths** (the first
  version painted a line as a filled polygon); the CSS has `path.ln{fill:none !important}` and fills only `circle`.
* `render(DATA)` is idempotent: it clears its containers first, so it can be called again with a new document.
* Theme: light palette on `:root`, dark tokens under `@media (prefers-color-scheme: dark)` guarded by
  `:root:not([data-theme="light"])` and again under `:root[data-theme="dark"]`; explicit `body` background.
* Fonts: IBM Plex Sans / Mono from Google Fonts (the only external resources; the Artifact CSP allows them).

## 4. Publishing and the live feed (Artifact tool of Claude Code)

1. First publish: `Artifact` with `file_path=<html>`, `favicon`, `description`, and `capabilities: {"db": {}}` (the
   page declares the artifact database; a page with `db` is organization-internal, not publicly shareable).
2. Seed / refresh the live document: `Artifact action=write_db, db_op=set, collection=dash, doc_id=data,
   file_path=dash_data.json` (the JSON's top-level object becomes the document).
3. The page boot block:
   ```js
   render(DATA);                                  // embedded snapshot first (works everywhere)
   const db = await window.claude.use('db');      // null outside the claude.ai viewer -> stay on the snapshot
   db.doc('dash/data').onSnapshot(snap => { if(snap.exists) render(snap.data()); }, err => {...});
   ```
   Every write to `dash/data` reaches open pages within seconds (realtime stream, 30 s polling fallback).
4. Cadence: a session cron (`CronCreate`, `4-59/10 * * * *`) runs `refresh.sh` and the `write_db` set every 10
   minutes, and republishes the HTML (same file path -> same URL) only at milestones. The cron is session-only and
   expires after 7 days; the page freezes at the last push when the session ends.
5. Republishing the same `file_path` keeps the URL; omit `capabilities` on a redeploy to keep the stored declaration.

## 5. Gotchas met on the way

* The workstation `python3` is old: keep `from __future__ import annotations` after the docstring.
* tqdm prints `27.00it`, so match progress with `[\d.]+it`, not `\d+it`.
* Never grep-poll with `tail -F` in a Monitor here (it never emitted); poll `wc -l` and print new lines.
* An openpi run's first log line after `Step 0` can be minutes late: three window-shape compiles (T60/T40/T20) each
  stall the GPUs for 5–15 min; the ETA must ignore those gaps.
* Kerberos: every ssh needs `KRB5CCNAME=FILE:/tmp/krb5cc_24706_<live cache>` plus `-i` key and `UserKnownHostsFile`;
  `refresh.sh` picks the newest cache file automatically.

## 6. Where the live copies are

Session scratchpad `…/scratchpad/dashboard/` holds the running copies used by the cron (same files); this directory is
the durable copy. Artifact: https://claude.ai/code/artifact/bcee575f-8b38-476d-b4cc-fa28db12d101 (owner: kewalk).

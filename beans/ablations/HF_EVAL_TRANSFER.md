# Transfer a saved checkpoint for offline evaluation

`upload_eval_checkpoint.py` uploads only a finalized checkpoint's Orbax `params/`,
its completion metadata and an `eval_manifest.json` provenance record to a
Hugging Face **model** repository. It does not upload optimizer state, datasets,
logs, source edits or credentials. It uses no GPU, stops no process and changes
no local checkpoint. Reading/hashing the weights does generate disk/network I/O.

Prefer a retained **B1000**, **B2000** or **B3000** checkpoint. A rolling B750
checkpoint may be deleted by training while an upload is still in progress.
Wait until Orbax has finalized the numeric step directory; temporary/incomplete
checkpoints are rejected. This bundle is for evaluation, not exact training resume.

From the training repository root, after pulling the helper from GitHub:

```bash
find beans/checkpoints -maxdepth 4 -type d -path '*_B/1000/params' -print

# Replace these two example values with the selected checkpoint and your HF repo.
EVAL_CKPT='beans/checkpoints/YOUR_CONFIG/YOUR_RUN_B/1000'
HF_EVAL_REPO='YOUR_HF_USERNAME/beans0922-eval'

openpi/.venv/bin/python -B beans/ablations/upload_eval_checkpoint.py \
  --checkpoint "$EVAL_CKPT" --repo-id "$HF_EVAL_REPO" --dry-run

# Hidden token entry: no token in the command arguments or shell history.
read -rsp 'Hugging Face WRITE token: ' HF_TOKEN
printf '\n'
export HF_TOKEN

openpi/.venv/bin/python -B beans/ablations/upload_eval_checkpoint.py \
  --checkpoint "$EVAL_CKPT" --repo-id "$HF_EVAL_REPO"
unset HF_TOKEN
```

The installed `openpi/.venv` already contains the needed `huggingface_hub` API.
The script can also use a cached HF login, or prompt for hidden input when no
token is available. Never paste the token into chat or put it in a tracked file.

The repository is private by default. An existing public repository is rejected
unless `--public` is explicitly supplied; that flag permits public release of
the weights. The receiving account must have read access to private repositories.

Remote layout:

```text
checkpoints/<config>/<experiment>/<step>/
  params/                 # complete Orbax parameter subtree
  _CHECKPOINT_METADATA
  eval_manifest.json      # names, step, code identifiers, recipe fields, file sizes
```

The uploader adds all files in one parent-pinned commit and verifies remote file
sizes at that commit. It refuses to overwrite an existing destination and never
deletes Hub files. A failed upload can be retried, but byte-level resumption is
not guaranteed. If the commit succeeded but subsequent verification was
interrupted, inspect the existing snapshot instead of deleting/replacing it.

After `UPLOAD_OK`, send these three printed lines to the evaluator:

```text
HF_REPO=...
REVISION=...  # immutable Hugging Face commit, not the training Git commit
SUBDIR=checkpoints/...
```

Download the exact revision and subtree, then pass `<download>/<SUBDIR>/params`
to the offline evaluator. `first_launch_code_from_log` records the first launch
header when available; `git_head_at_upload` is the checkout at upload time, which
can differ after `git pull`. Uncommitted source changes are not packaged, and
the script warns if the checkout has tracked edits.

Local tests (CPU only, no network or real HF repository writes):

```bash
openpi/.venv/bin/python -B -m unittest discover \
  -s beans/ablations -p upload_eval_checkpoint_test.py -v
```

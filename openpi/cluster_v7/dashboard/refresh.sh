#!/usr/bin/env bash
# Refresh the dashboard data: GPU/job snapshot -> build_dashboard.py -> dash_data.json (+ the HTML for republishing).
set -u
d=$(cd "$(dirname "$0")" && pwd)
export KRB5CCNAME=FILE:/tmp/krb5cc_24706_SKUlPR
cc=$(ls -t /tmp/krb5cc_24706_* 2>/dev/null | head -1); [ -n "$cc" ] && export KRB5CCNAME=FILE:$cc
timeout 120 ssh -o BatchMode=yes -o IdentitiesOnly=yes -i /iris/u/kewalk/.ssh/id_ed25519 -o UserKnownHostsFile=/iris/u/kewalk/.ssh/known_hosts -o ConnectTimeout=30 iris-hgx-1 'echo "{\"sampled\": \"$(date +%Y-%m-%dT%H:%M:%S)\", \"h200\": ["; srun --jobid=17403682 --overlap -n1 -N1 --gres=gpu:4 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F", " "{printf \"%s{\\\"gpu\\\":%s,\\\"used_mib\\\":%s,\\\"total_mib\\\":%s,\\\"util\\\":%s}\", (NR>1?\",\":\"\"), \$1,\$2,\$3,\$4}"; echo "], \"h100\": ["; srun --jobid=17356154 --overlap -n1 -N1 --gres=gpu:2 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F", " "{printf \"%s{\\\"gpu\\\":%s,\\\"used_mib\\\":%s,\\\"total_mib\\\":%s,\\\"util\\\":%s}\", (NR>1?\",\":\"\"), \$1,\$2,\$3,\$4}"; echo "], \"disk_free_gb\": $(df -BG /iris/u/kewalk | tail -1 | awk "{print \$4}" | tr -d G), \"jobs\": \"$(squeue -u kewalk -h -o "%i:%T:%N:%b:%L" | tr "\n" " ")\"}"' 2>/dev/null | grep -v "Could not create" > "$d/gpu.json.new"
python3 -c "import json,sys; json.load(open('$d/gpu.json.new'))" 2>/dev/null && mv "$d/gpu.json.new" "$d/gpu.json" || echo "gpu snapshot failed, keeping the previous one" >&2
cd "$d" && python3 build_dashboard.py gpu.json > boba_training_dashboard.html.new && mv boba_training_dashboard.html.new boba_training_dashboard.html
python3 -c "
import json; d=json.load(open('$d/dash_data.json')); a=d['runs']['memA']; p=a['progress'][-1] if a['progress'] else None
print('data', d['generated'], 'memA step', p['step'] if p else '-', 'of', a['total']-1, '| memB steps', len(d['runs']['memB']['steps']), '| h200 util', [g['util'] for g in d['gpu'].get('h200',[])], '| disk', d['gpu'].get('disk_free_gb'), 'GB | size', len(json.dumps(d))//1024, 'KB')"

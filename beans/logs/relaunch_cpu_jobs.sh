#!/usr/bin/env bash
# one-off (2026-09-22 00:15): move the ssh-launched CPU jobs into srun steps of job 17489557 (ssh shells land in a tiny cgroup)
pkill -u kewalk -f 'examples/yam/convert_yam_data_to_lerobot[.]py' ; pkill -u kewalk -f 'robomme/logs/probe_0920_data[.]py'
pkill -u kewalk -f 'pytest -q src/openpi/training/beans0922_test[.]py'; sleep 2
rm -rf /iris/u/kewalk/memory_project_v5/v5/data/lerobot/yam/bean_scoop_0905_v5
cd /iris/u/kewalk/memory_project_0920 && (setsid nohup srun --jobid=17489557 --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 bash beans/logs/convert_beans0905.sh > beans/logs/convert_beans0905.out 2>&1 < /dev/null &)
cd /iris/u/kewalk/memory_project_beans0922/openpi && (setsid nohup srun --jobid=17489557 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 env HOME=/iris/u/kewalk PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MEMORY_PROJECT_ROOT=/iris/u/kewalk/memory_project_beans0922 .venv/bin/python -m pytest -q src/openpi/training/beans0922_test.py src/openpi/training/beans_0920_test.py src/openpi/training/robomme_0920_test.py -p no:cacheprovider > /iris/u/kewalk/memory_project_beans0922/beans/logs/tests_cpu.out 2>&1 < /dev/null &)
sleep 40; ps -u kewalk -o pid,etime,time,pcpu,args | grep 'convert_yam_data_to_lerobo[t]\|pytest -[q]' | cut -c1-100
tail -c 400 /iris/u/kewalk/memory_project_0920/beans/logs/convert_beans0905.out | tr -d '\000' | tail -3 | cut -c1-160

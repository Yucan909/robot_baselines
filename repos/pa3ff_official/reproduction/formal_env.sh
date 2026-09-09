#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/feng/miniconda3/envs/articubot/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="/home/feng/miniconda3/envs/pa3ff/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="/home/feng/miniconda3/envs/pa3ff/lib:/home/feng/miniconda3/envs/pa3ff_formal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NUMEXPR_MAX_THREADS=8

exec /home/feng/miniconda3/envs/pa3ff_formal/bin/python "$@"

#!/usr/bin/env bash
set -euo pipefail

FINAL=/home/feng/robot_baselines/repos/where2act/code/v3_final
OFFLINE=/home/feng/robot_baselines/repos/where2act/logs/where2act_final_offline/final_metrics.json
SMOKE=/home/feng/robot_baselines/results/where2act/FINAL_NONFORMAL_SMOKE40/SMOKE40_SUMMARY.json

cd "$FINAL"

./run_model_final.sh

python - "$OFFLINE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit("offline final_metrics.json missing")
d = json.loads(p.read_text())
print("offline_gate_pass:", d.get("offline_gate_pass"))
if not d.get("offline_gate_pass", False):
    raise SystemExit(
        "FINAL OFFLINE GATE FAIL. Pre-formal pipeline stops here. "
        "Do not tune on formal shapes."
    )
PY

python patch_backend_final.py

python run_smoke40.py --workers 2

python - "$SMOKE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit("SMOKE40_SUMMARY.json missing")
d = json.loads(p.read_text())
print("engineering_smoke_pass:", d.get("engineering_smoke_pass"))
if not d.get("engineering_smoke_pass", False):
    raise SystemExit("SMOKE40 FAIL. Do not freeze or run formal benchmark.")
PY

python freeze_final.py

echo
echo "================================================================================"
echo "PREFORMAL PIPELINE PASS"
echo "================================================================================"
echo "Offline model gate: PASS"
echo "Non-formal physical smoke40: PASS"
echo "Freeze manifest created."
echo
echo "STOP HERE and inspect the reports before formal 1120."
echo "Then run:"
echo "  python $FINAL/run_formal_1120.py --workers 2"

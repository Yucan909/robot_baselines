#!/usr/bin/env python3
"""Watch/restart formal shards, then merge and independently aggregate them."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/feng/robot_baselines/results/articubot/unified_open")
SHARDS = ROOT / "formal_shards"
PROGRESS = ROOT / "progress.log"
PYTHON = "/home/feng/miniconda3/envs/articubot/bin/python"
CODE = "/home/feng/robot_baselines/repos/articubot/reproduction/unified_open"


def count_rows(path):
    if not path.is_file():
        return 0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def append_progress(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} supervisor {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def session_exists(name):
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def start_worker(shard_id):
    name = f"articubot_formal_{shard_id}"
    command = (
        f"env PYTHONUNBUFFERED=1 PYTHONPATH={CODE} {PYTHON} "
        f"{CODE}/run_articubot_unified_open.py --mode formal --device cuda:0 "
        f"--num-shards 4 --shard-id {shard_id} >> {SHARDS}/console_{shard_id}.log 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", name, command], check=True)
    append_progress(f"restarted shard={shard_id}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    last_counts = None
    stagnant = 0
    while True:
        counts = [count_rows(SHARDS / f"shard_{i}.jsonl") for i in range(4)]
        if counts == [280, 280, 280, 280]:
            break
        for shard_id, count in enumerate(counts):
            if count < 280 and not session_exists(f"articubot_formal_{shard_id}"):
                start_worker(shard_id)
        stagnant = stagnant + 1 if counts == last_counts else 0
        if counts != last_counts or stagnant % 5 == 0:
            append_progress(f"counts={counts} total={sum(counts)}/1120")
        last_counts = counts
        time.sleep(60)
    append_progress("all_shards_complete; merging")
    subprocess.run(
        [PYTHON, "-c", "from merge_formal_shards import main; main(4)"],
        cwd=CODE, env={**__import__('os').environ, "PYTHONPATH": CODE}, check=True,
    )
    append_progress("merge_complete; rerunning_official_cuda_forward_validation")
    subprocess.run(
        [PYTHON, f"{CODE}/run_articubot_unified_open.py", "--mode", "forward", "--device", "cuda:0"],
        cwd=CODE, env={**__import__('os').environ, "PYTHONPATH": CODE}, check=True,
    )
    append_progress("merge_complete; aggregating_and_auditing")
    subprocess.run(
        [PYTHON, f"{CODE}/aggregate_articubot_results.py"],
        cwd=CODE, env={**__import__('os').environ, "PYTHONPATH": CODE}, check=True,
    )
    (ROOT / "FULL_RUN_COMPLETE").write_text("PASS\n")
    append_progress("FULL_RUN_COMPLETE PASS")


if __name__ == "__main__":
    main()

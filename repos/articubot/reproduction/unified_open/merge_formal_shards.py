#!/usr/bin/env python3
"""Validate and deterministically merge resumable formal result shards."""

import json
import os
from pathlib import Path


ROOT = Path("/home/feng/robot_baselines/results/articubot/unified_open")
SHARDS = ROOT / "formal_shards"
OUTPUT = ROOT / "per_episode_results.jsonl"


def main(num_shards: int = 3):
    rows = []
    logs = []
    for shard_id in range(num_shards):
        path = SHARDS / f"shard_{shard_id}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
        log = SHARDS / f"shard_{shard_id}.log"
        if log.is_file():
            logs.append(log.read_text())
    rows.sort(key=lambda row: (int(row["case_index"]), int(row["repeat_id"])))
    keys = [(int(row["case_index"]), int(row["repeat_id"])) for row in rows]
    expected = [(case_index, repeat_id) for case_index in range(56) for repeat_id in range(20)]
    if keys != expected:
        raise RuntimeError(f"formal shards incomplete/duplicated: got={len(keys)} unique={len(set(keys))}")
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    (ROOT / "articubot_full_run.log").write_text("\n".join(logs))
    state = {
        "mode": "formal", "status": "complete", "completed": len(rows), "expected": 1120,
        "num_shards": num_shards,
    }
    temp_state = ROOT / "resume_state.json.tmp"
    temp_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temp_state, ROOT / "resume_state.json")
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()

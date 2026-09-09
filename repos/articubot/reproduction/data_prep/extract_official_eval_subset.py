#!/usr/bin/env python3
"""Materialize the exact first 25 official ArticuBot trials for one object."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from remote_zip_extract import RangeExtractor


TASK = "task_open_the_door_of_the_storagefurniture_by_its_handle"
EXPERIMENT = (
    "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-"
    "near-handle-300-demo-0.4-0.15-translation-first"
)
PRIMITIVE = "grasp_the_handle_of_the_storage_furniture_door_primitive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", default="40147")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    remote = RangeExtractor(args.index, args.url)
    task_root = f"open_the_door_{args.object_id}/{TASK}"
    experiment_root = f"{task_root}/experiment/{EXPERIMENT}"
    names = set(remote.archive.namelist())

    immediate = []
    for name in names:
        if not name.startswith(task_root + "/"):
            continue
        relative = name[len(task_root) + 1 :]
        if relative and "/" not in relative.rstrip("/") and not name.endswith("/"):
            immediate.append(name)
    for name in sorted(immediate):
        remote.extract(name, args.output_root)

    trial_pattern = re.compile(re.escape(experiment_root) + r"/([^/]+)/task_config\.yaml$")
    trial_names = sorted(
        match.group(1)
        for name in names
        if (match := trial_pattern.match(name)) is not None and "meta" not in match.group(1)
    )

    selected = []
    rejected = []
    for trial in trial_names:
        prefix = f"{experiment_root}/{trial}"
        required = [
            f"{prefix}/task_config.yaml",
            f"{prefix}/{PRIMITIVE}/states/state_0.pkl",
            f"{prefix}/{PRIMITIVE}/opened_angle.txt",
        ]
        missing = [name for name in required if name not in names]
        if missing:
            rejected.append({"trial": trial, "reason": "missing_required", "missing": missing})
            continue
        label_name = f"{prefix}/{PRIMITIVE}/label.json"
        if label_name in names:
            label = json.loads(remote.read_member(remote.archive.getinfo(label_name)))
            if not label.get("good_traj", False):
                rejected.append({"trial": trial, "reason": "label_good_traj_false"})
                continue
            required.append(label_name)
        for name in required:
            remote.extract(name, args.output_root)
        selected.append({"trial": trial, "files": required})
        if len(selected) == args.count:
            break

    if len(selected) != args.count:
        raise RuntimeError(f"selected only {len(selected)} of requested {args.count} trials")
    manifest = {
        "source": "official ArticuBot diverse_objects.zip",
        "object_id": args.object_id,
        "official_experiment": EXPERIMENT,
        "selection": "faithful prepare_env lexicographic order, good_traj gate, first N",
        "count": len(selected),
        "selected": selected,
        "rejected_before_cutoff": rejected,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

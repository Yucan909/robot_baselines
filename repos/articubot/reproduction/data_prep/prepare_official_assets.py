#!/usr/bin/env python3
"""Prepare PartNet objects for ArticuBot's official PyBullet loader.

The local benchmark copy contains ``mobility.urdf`` and visual meshes, but many
of its pre-existing ``mobility_vhacd.urdf`` files reference collision meshes
that are not present.  This script builds an isolated symlink overlay inside
ArticuBot and applies the repository's own VHACD operation to each *unique*
collision OBJ.  The upstream routine submits duplicate collision references;
deduplicating those references is the only behavioral fix here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ARTICUBOT_ROOT = Path("/home/feng/robot_baselines/repos/articubot")
SOURCE_ROOT = Path("/home/feng/robot_baselines/data/partnet-mobility")
OVERLAY_ROOT = ARTICUBOT_ROOT / "data/dataset"
CATALOG_ROOT = Path(
    "/home/feng/robot_baselines/results/where2act/four_task_noaff_v7_formal/"
    "20260905_082001/formal_catalogs"
)
RESULT_ROOT = Path("/home/feng/robot_baselines/results/articubot/official_idswap")
PROGRESS = RESULT_ROOT / "asset_preprocess.log"
FILENAME_RE = re.compile(r'filename="([^"]+\.obj)"')


def stamp(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def object_ids() -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for task in ("door_open", "drawer_open"):
        with (CATALOG_ROOT / f"{task}_formal_scene_catalog.jsonl").open() as handle:
            for line in handle:
                object_id = str(json.loads(line)["shape_id"])
                if object_id not in seen:
                    seen.add(object_id)
                    result.append(object_id)
    return result


def create_overlay(object_id: str) -> Path:
    source = SOURCE_ROOT / object_id
    target = OVERLAY_ROOT / object_id
    if not (source / "mobility.urdf").is_file():
        raise FileNotFoundError(source / "mobility.urdf")
    target.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(source):
        relative = Path(root).relative_to(source)
        target_dir = target / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (target_dir / directory).mkdir(exist_ok=True)
        for filename in files:
            # Never import the possibly broken pre-generated URDF.  The overlay
            # receives a locally verified one below.
            if relative == Path(".") and filename == "mobility_vhacd.urdf":
                continue
            destination = target_dir / filename
            if destination.exists() or destination.is_symlink():
                continue
            destination.symlink_to(Path(root) / filename)
    broken = target / "mobility_vhacd.urdf"
    if broken.is_symlink():
        broken.unlink()
    return target


def collision_meshes(urdf: Path) -> list[str]:
    inside_collision = False
    ordered: list[str] = []
    seen: set[str] = set()
    for line in urdf.read_text().splitlines(keepends=True):
        if "<collision" in line:
            inside_collision = True
        if inside_collision:
            match = FILENAME_RE.search(line)
            if match and match.group(1) not in seen:
                seen.add(match.group(1))
                ordered.append(match.group(1))
        if "</collision>" in line:
            inside_collision = False
    return ordered


def output_for(mesh: Path) -> Path:
    return mesh.with_name(mesh.stem + "_vhacd.obj")


def worker(input_path: Path, output_path: Path, log_path: Path) -> int:
    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        # Exact upstream call: no non-default VHACD arguments.
        p.vhacd(str(input_path), str(output_path), str(log_path))
    finally:
        p.disconnect(client)
    return 0 if output_path.is_file() and output_path.stat().st_size > 0 else 2


def launch_worker(job: tuple[str, Path, Path, Path], timeout: int) -> dict:
    object_id, input_path, output_path, log_path = job
    started = time.time()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(input_path),
        str(output_path),
        str(log_path),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        ok = result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0
        error = None if ok else f"worker_exit_{result.returncode}"
    except subprocess.TimeoutExpired:
        ok = False
        error = f"timeout_{timeout}s"
    return {
        "object_id": object_id,
        "input": str(input_path),
        "output": str(output_path),
        "ok": ok,
        "error": error,
        "runtime_sec": time.time() - started,
    }


def write_vhacd_urdf(object_dir: Path) -> dict:
    source = object_dir / "mobility.urdf"
    destination = object_dir / "mobility_vhacd.urdf"
    inside_collision = False
    lines: list[str] = []
    replacements = 0
    fallbacks = 0
    for line in source.read_text().splitlines(keepends=True):
        if "<collision" in line:
            inside_collision = True
        if inside_collision:
            match = FILENAME_RE.search(line)
            if match:
                relative = match.group(1)
                convex_relative = str(Path(relative).with_name(Path(relative).stem + "_vhacd.obj"))
                if (object_dir / convex_relative).is_file():
                    line = line.replace(relative, convex_relative)
                    replacements += 1
                else:
                    fallbacks += 1
        lines.append(line)
        if "</collision>" in line:
            inside_collision = False
    temporary = destination.with_suffix(".urdf.tmp")
    temporary.write_text("".join(lines))
    os.replace(temporary, destination)
    missing = []
    for relative in FILENAME_RE.findall(destination.read_text()):
        if not (object_dir / relative).is_file():
            missing.append(relative)
    return {
        "urdf": str(destination),
        "collision_replacements": replacements,
        "collision_fallbacks": fallbacks,
        "missing_references": sorted(set(missing)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=205)
    parser.add_argument("--worker", nargs=3, metavar=("INPUT", "OUTPUT", "LOG"))
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(*(Path(value) for value in args.worker)))

    selected = args.objects or object_ids()
    selected = list(dict.fromkeys(str(value) for value in selected))
    stamp(f"START objects={len(selected)} workers={args.workers} timeout={args.timeout}")
    object_dirs: dict[str, Path] = {}
    jobs: list[tuple[str, Path, Path, Path]] = []
    for object_id in selected:
        object_dir = create_overlay(object_id)
        object_dirs[object_id] = object_dir
        meshes = collision_meshes(object_dir / "mobility.urdf")
        for relative in meshes:
            input_path = object_dir / relative
            output_path = output_for(input_path)
            if not input_path.is_file():
                stamp(f"MISSING_INPUT object={object_id} mesh={relative}")
                continue
            if not output_path.is_file() or output_path.stat().st_size == 0:
                jobs.append((object_id, input_path, output_path, output_path.with_name(output_path.stem + "_log.txt")))
        stamp(f"OBJECT_QUEUED object={object_id} unique_meshes={len(meshes)}")
    stamp(f"VHACD_PENDING unique_jobs={len(jobs)}")

    statuses: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(launch_worker, job, args.timeout) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            status = future.result()
            statuses.append(status)
            if not status["ok"] or index % 25 == 0 or index == len(jobs):
                stamp(
                    f"VHACD_DONE {index}/{len(jobs)} object={status['object_id']} "
                    f"ok={status['ok']} runtime={status['runtime_sec']:.1f}s error={status['error']}"
                )

    audits = {}
    for object_id, object_dir in object_dirs.items():
        audits[object_id] = write_vhacd_urdf(object_dir)
    failures = [status for status in statuses if not status["ok"]]
    payload = {
        "schema": "articubot_official_vhacd_deduplicated_v1",
        "official_operation": "pybullet.vhacd(input, output, log) with default parameters",
        "interface_fix": "deduplicate repeated collision mesh references before VHACD",
        "objects": selected,
        "jobs": len(jobs),
        "failures": failures,
        "urdf_audit": audits,
    }
    audit_path = RESULT_ROOT / "asset_preprocess_audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    missing_refs = sum(len(item["missing_references"]) for item in audits.values())
    stamp(f"COMPLETE jobs={len(jobs)} failures={len(failures)} missing_urdf_refs={missing_refs}")
    if missing_refs:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

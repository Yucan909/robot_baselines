#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CATALOG = Path('/home/feng/robot_baselines/configs/pa3ff/reproduction_v1_formal/formal_episode_catalog.jsonl')
EXPECTED = {'door_open': 1120, 'door_close': 1120, 'drawer_open': 720, 'drawer_close': 680}


def result_path(root: Path, case: dict) -> Path:
    return (root / case['task'] / f"{int(case['target_index']):03d}_{case['shape_id']}_{case['target_link']}" /
            f"trial_{int(case['trial_index']):02d}_seed_{case['seed']}" / 'result.json')


def rate(n: int, d: int):
    return None if d == 0 else n / d


def pct(value) -> str:
    return 'N/A' if value is None else f'{100.0 * value:.3f}%'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.formal_root.resolve()
    manifest = json.loads((root / 'protocol_manifest.json').read_text(encoding='utf-8'))
    cases = [json.loads(x) for x in CATALOG.read_text(encoding='utf-8').splitlines() if x.strip()]
    rows = []
    incomplete = []
    for case in cases:
        p = result_path(root, case)
        try:
            row = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:
            incomplete.append({'case': case, 'path': str(p), 'reason': f'unreadable: {exc}'})
            continue
        if row.get('episode_status') != 'complete':
            incomplete.append({'case': case, 'path': str(p), 'reason': row.get('episode_status'),
                               'exception': row.get('exception')})
            continue
        if row.get('checkpoint_sha256') != manifest['checkpoint_sha256'] or row.get('catalog_sha256') != manifest['scene_catalog_sha256']:
            incomplete.append({'case': case, 'path': str(p), 'reason': 'provenance_mismatch'})
            continue
        rows.append(row)

    by_task = defaultdict(list)
    for row in rows: by_task[row['task']].append(row)
    metrics = {}
    failure = {}
    for task, expected in EXPECTED.items():
        task_rows = by_task[task]
        grasp = sum(bool(x['grasp_success']) for x in task_rows)
        final = sum(bool(x['final_success']) for x in task_rows)
        reached35 = sum(float(x.get('directional_task_progress') or 0.0) >= 0.35 for x in task_rows)
        reached = sum(bool(x['reached_target_40']) for x in task_rows)
        metrics[task] = {
            'expected_trials': expected, 'completed_trials': len(task_rows),
            'grasp_success': grasp, 'grasp_success_rate': rate(grasp, expected),
            'post_grasp_operation_success': final,
            'post_grasp_operation_success_rate': rate(final, grasp),
            'final_success': final, 'final_success_rate': rate(final, expected),
            'reached_target_35': reached35, 'reached_target_35_rate': rate(reached35, expected),
            'reached_target_40': reached, 'reached_target_40_rate': rate(reached, expected),
        }
        failure[task] = dict(Counter(x.get('failure_reason') or 'success' for x in task_rows))

    summary = {
        'status': 'PASS_COMPLETE' if len(rows) == 3640 and not incomplete else 'INCOMPLETE',
        'formal_root': str(root), 'checkpoint': manifest['checkpoint'],
        'checkpoint_sha256': manifest['checkpoint_sha256'],
        'completed_trials': len(rows), 'expected_trials': 3640,
        'tasks': metrics, 'incomplete_count': len(incomplete),
    }
    out = root / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'FINAL_METRICS.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'failure_breakdown.json').write_text(json.dumps({'failure_breakdown': failure, 'incomplete': incomplete}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    with (out / 'FINAL_METRICS.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'completed', 'expected', 'grasp_success', 'grasp_success_rate',
                    'post_grasp_operation_success', 'post_grasp_operation_success_rate',
                    'final_success', 'final_success_rate', 'reached_target_35', 'reached_target_35_rate',
                    'reached_target_40', 'reached_target_40_rate'])
        for task in EXPECTED:
            m = metrics[task]
            w.writerow([task, m['completed_trials'], m['expected_trials'], m['grasp_success'], m['grasp_success_rate'],
                        m['post_grasp_operation_success'], m['post_grasp_operation_success_rate'],
                        m['final_success'], m['final_success_rate'], m['reached_target_35'], m['reached_target_35_rate'],
                        m['reached_target_40'], m['reached_target_40_rate']])

    target_groups = defaultdict(list)
    for row in rows: target_groups[(row['task'], row['target_index'], row['shape_id'], row['target_link'])].append(row)
    with (out / 'per_target_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'target_index', 'shape_id', 'target_link', 'completed', 'expected',
                    'grasp_success', 'grasp_success_rate', 'final_success', 'final_success_rate',
                    'reached_target_35', 'reached_target_35_rate', 'reached_target_40', 'reached_target_40_rate'])
        for key in sorted(target_groups):
            rr = target_groups[key]; grasp = sum(x['grasp_success'] for x in rr); final = sum(x['final_success'] for x in rr)
            reached35 = sum(float(x.get('directional_task_progress') or 0.0) >= 0.35 for x in rr)
            reached = sum(x['reached_target_40'] for x in rr)
            w.writerow([*key, len(rr), 20, grasp, rate(grasp, 20), final, rate(final, 20),
                        reached35, rate(reached35, 20), reached, rate(reached, 20)])

    lines = [
        manifest.get('protocol_name', 'PA3FF REPRODUCTION') + ' — FINAL FORMAL METRICS',
        f'checkpoint: {manifest["checkpoint"]}',
        f'checkpoint_sha256: {manifest["checkpoint_sha256"]}',
        f'TOTAL: {len(rows)} / 3640', '',
    ]
    for task in EXPECTED:
        m = metrics[task]
        lines.extend([
            task,
            f'completed: {m["completed_trials"]} / {m["expected_trials"]}',
            f'grasp_success_rate: {m["grasp_success"]} / {m["expected_trials"]} = {pct(m["grasp_success_rate"])}',
            f'post_grasp_operation_success_rate: {m["post_grasp_operation_success"]} / {m["grasp_success"]} = {pct(m["post_grasp_operation_success_rate"])}',
            f'final_success_rate: {m["final_success"]} / {m["expected_trials"]} = {pct(m["final_success_rate"])}',
            f'reached_target_35: {m["reached_target_35"]} / {m["expected_trials"]} = {pct(m["reached_target_35_rate"])}',
            f'reached_target_40: {m["reached_target_40"]} / {m["expected_trials"]} = {pct(m["reached_target_40_rate"])}', '',
        ])
    (out / 'FINAL_METRICS.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    if summary['status'] != 'PASS_COMPLETE':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

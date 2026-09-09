#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVES="$ROOT/dataset_archives"

restore_one() {
  local prefix="$1"
  if ! compgen -G "$ARCHIVES/${prefix}.tar.zst.part-*" >/dev/null; then
    printf '缺少归档分卷: %s\n' "$prefix" >&2
    return 1
  fi
  cat "$ARCHIVES/${prefix}.tar.zst.part-"* | zstd -d --stdout | tar -xf - -C "$ROOT"
}

restore_one partnet-mobility
restore_one flowbot3d-custom
restore_one where2act-four-task
restore_one articubot-assets
restore_one pa3ff-compact-source

python "$ROOT/scripts/verify_bundle.py"

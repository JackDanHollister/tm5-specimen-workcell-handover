#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${ISAAC_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$task_python" "$task_root/scripts/isaac_workcell.py" "$@"

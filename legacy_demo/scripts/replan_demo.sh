#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUMOTION_ENV="${CUMOTION_ENV:-$HOME/isaac-work/envs/cumotion-1.1}"
DEMO_ASSETS_DIR="${DEMO_ASSETS_DIR:-$DEMO_ROOT/release_bundle}"
cd "$DEMO_ROOT"
python3 scripts/verify_demo_assets.py --bundle "$DEMO_ASSETS_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$CUMOTION_ENV/bin/python" -u scripts/plan_reconstructed_pin_demo.py \
  --reconstruction examples/seven_pins.json \
  --model-dir "$DEMO_ASSETS_DIR/assets/planner_model" \
  --output "outputs/plan-$(date -u +%Y%m%dT%H%M%S)-$$.json" "$@"

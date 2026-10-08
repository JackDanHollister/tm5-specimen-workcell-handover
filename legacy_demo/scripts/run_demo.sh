#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac-work/envs/isaac-sim-6.0}"
DEMO_ASSETS_DIR="${DEMO_ASSETS_DIR:-$DEMO_ROOT/../assets/legacy_demo}"
cd "$DEMO_ROOT"
python3 scripts/verify_demo_assets.py --bundle "$DEMO_ASSETS_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$ISAAC_ENV/bin/python" -u scripts/run_reconstructed_pin_demo.py \
  --plan "$DEMO_ASSETS_DIR/evidence/transfer-plan.json" \
  --usd "$DEMO_ASSETS_DIR/assets/isaac/tm5s_with_2fg7/tm5s_with_2fg7.usda" \
  --import-report "$DEMO_ASSETS_DIR/assets/isaac/import-report.json" \
  --report "outputs/run-$(date -u +%Y%m%dT%H%M%S)-$$.json" "$@"

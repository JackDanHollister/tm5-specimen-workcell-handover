# TM5 specimen workcell — colleague handover

A portable, offline NVIDIA Isaac workcell containing real drawer scans, four
robot attachment configurations, calibration evidence and the existing pin-axis
workflow. The development task is automatic specimen-relative viewing and access
planning for unfamiliar drawers. See [the mission](docs/MISSION.md).

The six drawer datasets were captured on **7 October 2026**. Each has one lit EIH
overhead photograph and three Photoneo scans at the centre and robot-base Y
offsets **−100 mm / +100 mm**, retaining each scan's actual flange pose. They are
six independent drawer arrangements swapped into the same work area, not six
drawers simultaneously present on the bench.

## Get the complete package

This repository and its release assets are **private**. The owner must grant
your GitHub account access before the commands below work. Use **Python 3.11+**
for the fetcher and offline tools.

```bash
gh auth login
gh repo clone JackDanHollister/tm5-specimen-workcell-handover
cd tm5-specimen-workcell-handover
python3 scripts/fetch_assets.py
python3 -m pip install -e '.[test]'
python3 scripts/verify_assets.py
```

Code/configuration is in git; larger payloads are checksum-verified private release
archives. `fetch_assets.py` downloads and installs them into `assets/`, `data/` and
`models/`. It joins split archives automatically. Use `--without-models` to omit
the neural-model inputs when only inspecting scenes and replaying saved fits.
Keep the local download archives for another installation, or remove `.downloads/`
after extraction to reclaim their disk space.

## Open a real drawer in Isaac

Tested with **Isaac Sim 6.0.1.0**. Isaac and NVIDIA binaries are installed
separately. Set `ISAAC_PYTHON` to your Isaac environment's Python executable.

```bash
export ISAAC_PYTHON=/path/to/isaac/python
./scripts/run_isaac.sh --drawer drawer_001 --robot scanner_no_gripper
./scripts/run_isaac.sh --drawer drawer_004 --robot no_scanner_gripper --replay
```

The camera and support follow the wrist. Registered drawer geometry stays fixed
in robot-base coordinates. `--replay` shows saved joint samples from the successful
survey; it is kinematic evidence playback, not a time-accurate dynamics model.
You can also open any of the 24 `assets/scenes/*.usda` combinations directly in
Isaac and use its articulation tools. The scripts connect to no robot, camera or ROS.

| Model name | Photoneo + side support | Whole 2FG7 | EIH / QC / circular adapter-cap |
|---|---|---|---|
| `scanner_gripper` | Present | Present | Retained |
| `no_scanner_gripper` | Removed | Present | Retained |
| `scanner_no_gripper` | Present | Removed | Retained |
| `no_scanner_no_gripper` | Removed | Removed | Retained |

Every variant includes USD, URDF, XRDF sphere geometry, frame paths and model
metadata. The measured **23.15 mm** adapter/cap extension is retained in all four
variants. [Model/frame details and limitations](docs/MODELS_AND_FRAMES.md).

## Reproduce the pin result and use its view patterns

```bash
python3 scripts/replay_pin_fit.py
python3 scripts/export_view_candidates.py \
  --target-m 0.46 -0.07 0.04 --reference-specimen 4 \
  --robot scanner_no_gripper --output outputs/example-candidates.json
```

The first command reproduces seven historical provisional shaft directions from
saved observations, without loading a neural model. The second translates the
successful specimen-relative viewing patterns to a new framing point. Its output
is candidate camera/flange transforms: **IK and complete collision/visibility
planning still need to be done**. Historical absolute joints are evidence, not
universal executable poses. [Pin toolkit instructions](docs/PIN_WORKFLOW.md).

The six newly scanned drawers have **no accepted per-specimen pin orientations**.
The historical fits belong to their original fixed-placement image groups and
must not be assigned to new specimens automatically.

## What is included

- Six selectable real drawer scenes, raw cloud/depth/normal/confidence/laser-texture
  arrays, reconstruction PLY/NPZ files, EIH photographs and pose records.
- 117 historical multiview images in seven fixed placements, cached DINO outputs,
  3D fit results, endpoint gallery and successful target-relative view patterns.
- Factory EIH data, Photoneo hand-eye estimate, supporting board captures and
  calibration diagnostics; original source records/hashes are retained.
- YOLO specimen detector, DINOv3 backbone plus the existing five-class head, and
  the saved-image pin toolkit. No new model training is performed.
- [Motion limits/profiles](docs/MOTION.md), [data inventory](docs/DATA.md),
  [calibration/coordinate conventions](docs/CALIBRATION.md) and tested examples.

The existing seven-pin illustrative transfer demo is also available under
`legacy_demo/`, with its data in `assets/legacy_demo/` and presentation video in
`data/pins/legacy-transfer.mp4`. It uses its own illustrative trays; those are
separate from the six real drawer scenes.

```bash
DEMO_ASSETS_DIR="$PWD/assets/legacy_demo" \
  ISAAC_ENV=/path/to/isaac/environment bash legacy_demo/scripts/run_demo.sh
```

## Checks

```bash
python3 -m pytest -q
"$ISAAC_PYTHON" scripts/verify_assets.py --usd
./scripts/run_isaac.sh --headless --self-test \
  --drawer drawer_001 --robot scanner_no_gripper --frames 20
```

See [validation](docs/VALIDATION.md) for what was checked. Estimated camera
registration, incomplete surfaces, assumed bracket envelopes and provisional
pin endpoints remain explicit. This package supports offline development and
does not qualify physical pickup or insertion.

Third-party code, assets and model inputs retain their licences; see
[third-party notices](THIRD_PARTY_NOTICES.md).

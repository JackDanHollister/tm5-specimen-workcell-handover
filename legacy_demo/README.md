# Reconstructed pin transfer — TM5 digital twin

An offline demonstration joining saved multiview pin reconstruction to a TM5S
robot and 2FG7 gripper in NVIDIA Isaac Sim.

Seven pins appear one at a time at their reconstructed positions. The gripper
aligns with each tilted shaft, approaches along it, grips, and withdraws along
the same axis. After clearing the source, it turns the specimen upright and
places it in the next destination slot. The arm returns home before the next
pin appears; placed specimens remain in the drawer.

[Download the presentation video and demo assets](https://github.com/JackDanHollister/tm5-reconstructed-pin-demo/releases/tag/v0.1.0).
Repository access is required because the project and release are private.

[New: pin-end and exposed-shaft image review (24 September)](reviews/pin-endpoints-2026-09-24/README.md)
compares 117 annotated saved views across seven pins. Endpoint and specimen-side
marks are exploratory, not approved grip measurements; the demo plan is unchanged.

## Run the demo

The video needs only an ordinary MP4 player. Interactive playback requires the
separately installed Isaac Sim **6.0.1.0** Python environment and a supported
NVIDIA GPU. Isaac Sim and cuMotion are not included in this repository.

```bash
gh repo clone JackDanHollister/tm5-reconstructed-pin-demo
cd tm5-reconstructed-pin-demo
gh release download v0.1.0 --pattern demo-assets.zip
unzip -n demo-assets.zip
./scripts/run_demo.sh
```

The window starts paused. Use **Play**, **Pause**, **Reset** and the speed slider
in the Reconstructed Pin Demo panel. Reset clears the destination, opens the
fingers and restores the arm and first source pin. Playback is kinematic;
there are no ROS, robot, camera or gripper connections.

The launcher defaults to `~/isaac-work/envs/isaac-sim-6.0`; set `ISAAC_ENV` to
your installed Isaac environment. Accept NVIDIA's applicable EULA when setting
up that environment.

## Change the grasp offset

The default is **5 mm below the provisional upper shaft endpoint**. Edit
`grasp_below_upper_endpoint_mm` in `config/reconstructed_pin_demo.json`, or pass
an override when generating a new plan:

```bash
./scripts/replan_demo.sh --grasp-offset-mm 7 --output outputs/grasp-7mm.json
./scripts/run_demo.sh --plan outputs/grasp-7mm.json
```

Replanning requires the separately installed **cuMotion 1.1.0** environment.
Set `CUMOTION_ENV` if it differs from `~/isaac-work/envs/cumotion-1.1`.
Changing configuration never silently changes an already validated plan.

## What is reconstructed

`examples/seven_pins.json` contains the exact geometric fields used from the
seven saved multiview fits: a point on each line, its direction, observed shaft
support endpoints and supporting-view information. Coordinates are metres in
the robot base frame. The original reconstruction's SHA-256 is retained in
`examples/provenance.json`. No raw photographs or model weights are included.

The **axis and observed span** come from the images. The upper support endpoint
is a provisional proxy for the pinhead; it has not been physically verified.
Foam height, buried length, diameter and the small coloured specimen bodies are
explicit simulation assumptions. The source tray is a fixed 250 × 180 mm
presentation tray; the destination drawer is 800 × 500 mm. Source pin lines
and their lateral locations are preserved without repositioning cases to solve
reachability.

This demonstrates geometric planning and rigid attach/release. It does not
simulate grasp forces, foam friction, slip, bending or actual specimen shapes,
and it is not a physical robot execution package.

## Validation and recording

The planner checks all seven complete home-to-home paths against the robot's
joint limits, self-collision and both trays. It checks the carried pin/body and
previously placed specimens, with 1 mm required clearance. Only intentional
shaft penetration into foam is excluded. Source approach/extraction and
destination insertion/retreat have separate line-following checks. The base
joint stays inside the inherited −60° to +150° sector.

The renderer checks the actual articulation and final placements. Published
validation reports accompany the video. Passing these checks applies to the
declared simulation geometry, not physical accuracy.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q

./scripts/run_demo.sh --headless --self-test --fps 15 --film-speed 1.5
./scripts/run_demo.sh --headless --fps 15 --film-speed 1.5 \
  --video outputs/my-demo.mp4
```

Recording needs FFmpeg with H.264 support (`--ffmpeg /path/to/ffmpeg` if needed).
It includes an overview, a moving pin/gripper close-up and stage captions.

## Source and licensing

The transfer geometry, collision checks and planning helpers are reused from
[Techman Isaac Jazzy Live Twin](https://github.com/JackDanHollister/techman-isaac-jazzy-live-twin).
The input fits were produced by the Viktor multiview workflow. This repository
contains only the offline Python dependency set needed for this demonstration.
See [third-party notices](THIRD_PARTY_NOTICES.md) for robot and gripper assets.

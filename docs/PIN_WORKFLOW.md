# Pin images, orientations and code

The seven historical fixed-placement groups contain117 saved images, including
the20-view sets4 and5. The preserved baseline used frozen DINOv3 ViT-H+/16 features
and an existing five-class head, then image-edge geometry and multiview plane
intersection. It produced seven provisional base-frame shaft lines.

`data/pins/observations.json` is the portable image/pose/geometry table. Its paths
are package-relative; original image/coordinate hashes, K, distortion, camera
matrices, joints and rejection results are preserved. `reconstruction.json`
contains the original fit diagnostics; `seven_pins.json` is the demonstration
input. `view_patterns.json` contains successful `T_target_eih_m` transforms, with
target axes parallel to robot-base axes and the origin at the historical framing
point. Translate these to a new target, then solve current IK and collisions.

Fast example (no neural inference):

```bash
python3 scripts/replay_pin_fit.py
python3 scripts/export_view_candidates.py --target-m .46 -.07 .04 \
  --reference-specimen 4 --robot scanner_no_gripper
```

Run geometry extraction again from the bundled cached predictions in a **fresh**
output directory. The original data are never overwritten:

```bash
python3 -m pip install -e '.[inference]'
python3 scripts/run_pin_toolkit.py --output outputs/new-cached-analysis
```

The upstream-style code and HTML templates live in `toolkit/pin_axis/`; its
reconstruction defaults have been made package-relative. Use the wrapper above
so manifests/caches are prepared correctly. `report.py` / `multiview_report.py`
consume the corresponding derived run directory and included templates.

For new photographs, the packaged DINO backbone and five-class head are in
`models/dino/`. `inference.py` points there, not to the original workstation.
`SavedDino(model_dir=..., device='cuda'/'cpu')` exposes the device. The bundled
YOLO detector provides specimen framing boxes:

```bash
python3 scripts/detect_specimens.py --drawer drawer_001 --device cpu
```

Detections are specimen framing points, not pin/grasp centres. Multiple handling
or mounting pins can be confused. Recovery must combine identity, visibility and
independent view diversity; optical-roll duplicates are not independent views.

The endpoint gallery is at `data/pins/endpoint-gallery/index.html`. Outer endpoint
clusters showed development-set agreement in67/71 eligible views; this is not
accuracy. Head underside, true specimen/card boundary and usable gripping length
remain uncertain. The old demo's configurable5mm upper-end offset is a simulation
assumption, not a qualified physical grip.

The six Oct7 drawers have no accepted pin orientations. The later Oct6 obscured-pin
experiment is bundled separately in `data/recent_pin_trial/` and remained uncertain;
historical success does not label new specimens.
No new training, live acquisition, gripper actuation or physical transfer is part
of this package.

# Data inventory

`configs/drawers.json` is the portable drawer index. Each independent drawer has
`eih/image.png`, its original acquisition/coordinates record, three raw scans,
original registered reconstruction arrays/PLY and the successful survey feedback.
Raw scanner output is millimetres; registered reconstruction is robot-base metres.

| Drawer | Original registered observations (including overlap) |
|---|---:|
| drawer_001 | 1,415,791 |
| drawer_002 | 1,494,092 |
| drawer_003 | 1,474,599 |
| drawer_004 | 1,756,661 |
| drawer_005 | 1,865,460 |
| drawer_006 | 1,832,917 |
| Total | 9,839,520 |

There are six overhead photographs and eighteen Oct7 raw scans. Counts include
overlapping observations, not unique surface points or specimen counts. Per-view
meshes retain neighbouring-pixel triangles and missing areas; they are combined
without completing holes, smoothing, ICP or connecting triangles across views.
The first original observation per display voxel is retained, capped at450,000
display points. Full arrays retain the original registered observation count.

The visual texture is the scanner's **laser greyscale**, not colour from the EIH
photograph. The RGB photograph is retained separately. The floor, rim, wings and
labels can be observed, but unseen walls/undersides and thin reflective pins are
not assumed complete. Surface/camera/frame provenance stays with each view.

Drawer002's successful source is the second side-sweep attempt. Its first attempt
was blocked by a local calibration command filter before movement; it is not
silently counted as an extra scan or successful trajectory. The data copy selects
the complete source from the final verified Oct7 session index.

Historical data live separately in `data/pins/`:117 images in seven fixed-placement
groups, cached predictions, exact saved observations, fit diagnostics, endpoint
gallery and the illustrative transfer video. Group labels are placement IDs, not
independently verified biological/specimen identities for held-out training.

`data/recent_pin_trial/` includes14 additional lit RGB/depth pairs from Oct6, with
their own actual camera/flange poses and focus/exposure metadata. This separate
obscured-pin case remains **uncertain** and supplies no accepted pin axis. These
extra depth scans are not included in the eighteen-scan Oct7 drawer total.

`data/calibration/` preserves the working Photoneo fit, factory EIH information,
translation/tilt supporting captures and diagnostics. Raw original JSON can record
source workstation paths; packaged runnable manifests use relative locations.
Original hashes remain meaningful because original photos, scans and metadata are
copied without changing their contents.

`configs/payload_files.json` lists every distributed payload file, size and SHA256.
`scripts/verify_assets.py` checks these identities plus the six/eighteen Oct7 count.
Large-file release archives have separate checksums and are joined/verified by
the fetcher before extraction.

The v0.2.0 [experiment supplement](EXPERIMENTS.md) adds original development
records/source and further close-view photographs. Its43 session records include
offline benchmark replays, rather than43 independent hardware runs. Added files
are listed in `configs/experiments.json` and the combined payload manifest;
reused original payloads and omitted dense records are explicit.

# Calibration and coordinate conventions

Working transforms are in `configs/calibration.json`. Factory EIH intrinsics and
focus-specific data are in `data/calibration/eih_factory_intrinsics.json`.
The Photoneo draft, board correspondences, diagnostics and supporting captures are
under `data/calibration/`; original records remain unchanged as provenance.

For homogeneous column vectors in millimetres:

```
p_base_mm = T_base_flange_mm @ T_flange_camera_mm @ p_camera_mm
```

Raw Photoneo NPZ clouds use **CameraSpace / PrimaryCamera**, in **millimetres**.
Use each capture's actual flange pose. Registered reconstruction arrays already
use **robot-base metres**; do not apply the camera transform a second time.
EIH pixel coordinates have +x right, +y down. The optical +z axis points forward;
USD viewing cameras use a different convention. Intrinsics must match actual
image size and focus; no unexplained pixel scaling or hand-eye inversion.

The optical housing/body uses the original vendor mesh coordinates. Its bbox
centre is not a mounting hole, TCP or centre of mass. The estimate preserves the
camera's unusual mounting angle.

The preferred Photoneo draft combines eight translation views and seven usable
board-bearing tilt views, with separate board poses for the two sessions. Final
training agreement was about0.332mm RMS over2104 points; the largest leave-one-out
mapped-point shift was2.390mm RMS. These are consistency/sensitivity diagnostics,
not absolute accuracy or certified error bounds. Board pitch, scanner scale,
mount stability and robot absolute calibration remain separate issues.

The final all-data draft includes retrospective check views; their check residuals
are not an independent test of that final estimate. Two +/-8-degree Y tilt captures
showed an empty drawer and are excluded from board fitting. Missing original
combined-angle validation was not invented. The original diagnostics preserve this.

Historical pin images use their recorded factory EIH calibration; the raw historical
metadata is included separately at `data/pins/historical_eih_metadata.json`.
`data/pins/observations.json` holds the exact camera matrices, K, distortion and
flange/joint records used in the existing fits. Those specimens were confirmed
fixed within each historical group. New drawers have no such pin-axis labels.

Raw source JSON can contain original machine paths as documentary provenance.
Runnable entrypoints use package-relative manifests and bundled files, and do not
open those original workspace paths. Original images/raw scans retain their hashes.

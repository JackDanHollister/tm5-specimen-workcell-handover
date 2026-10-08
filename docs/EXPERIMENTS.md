# Previous implementations and experiment evidence

The handover includes our own October development work, including failures. Start
with the examples below, then inspect the original records and source. This is
useful starting code and evidence for the general planner; it is not an already
finished arbitrary-drawer workflow.

For an existing checkout, install only the new supplement:

```bash
git pull
python3 scripts/fetch_assets.py --experiments-only
python3 scripts/replay_experiment_evidence.py
```

Normal `fetch_assets.py` also installs it. The original large data/model bundles
remain on v0.1.0; the additional archive is on v0.2.0. Existing downloads can be
reused. No original release asset is replaced.

## What we already implemented

| Component | Starting evidence | What it demonstrates |
|---|---|---|
| YOLO image centre to scan-supported 3D framing point | [Example](../examples/prior-specimen-framing.json); original `20261006-pin-orbit-continuation-v9/target.json` | One historical detection maps to base XYZ approximately (459.98, −66.90, 37.28) mm using407 supporting points. This is an inspection target, not a detected pin or grasp. |
| Drawer lining, walls, free-area and depth extraction | `20261001-drawer-usable-areas/analysis-v1/` and `20261001-two-drawer-reconstruction/measurements/` | Fitted outline/lining and observed coverage, with missing areas retained. It is previous placement evidence, not labels for the six new drawers. |
| Close EIH views around a framing target | `20261005-close-pivots-isaac-v2/trajectory.json` | Ten photographic stations, one descent and a constant232.559mm lens-to-target distance, reaching42° in feasible directions. Preview sampling is distinct from controller commands. |
| Actual ten-view acquisition | `20261006-close-pivots-live-v1/run_summary.json` | Ten lit EIH/Photoneo pairs completed, but the motion/capture session took398.47s and the pin fit was insufficient. Complete acquisition did not imply useful orientation. |
| Historical20-view comparison and target-relative angles | `20261006-historical-pin-view-study-v4/` | Original joint/camera poses, view-angle comparison, preferred views and rejected candidates. Historical specimen4/5 images are already in `data/pins/`. |
| Rapid joint-PVT route compilation | [Compact programme](../examples/prior-rapid-programme.json); original `rapid-batch-benchmarks/motion-home-release/` | Eleven photo stations in three banks,18 controller moves,35.75s planned motion at45%. Dense clearance samples are checks rather than transmitted stops. Physical capture throughput was not measured for this prototype. |
| Resident model, cached observations and adaptive image banks | `rapid-batch-benchmarks/20261006-v3/` and `historical-v2/` | Recent14 images remained uncertain:6.09s model load,3.94s cold replay and0.86s cached replay. Historical data produced a stable candidate after10 views, but semantic pin identity remained unconfirmed. These are saved-image timings. |
| Fast drawer survey | Oct7 side-sweep sessions; existing `data/drawers/*/recorded_route.json` | Successful centre/±100mm survey,2400mm/s nominal transit,300ms ramp and45% override. Retains the first slow profile and the failed drawer002 command-filter attempt alongside the successful retry. |

Paths in this table are relative to `data/experiments/records/`. The normalised
examples contain package-relative provenance; original records retain their
workstation paths and historical hashes. Use `configs/experiments.json` to resolve
an original source into its copied file or an existing base payload. The framing
example's photograph, coordinates and registered points are supplied.

## Reference code to reuse

`data/experiments/reference_source/workcell/scripts/` is a byte-identical snapshot
of the existing implementation. Matching tests are beside it. Original frozen
snapshots inside the run folders show older executed versions; the current source
must not be assumed to be the version that produced every earlier failure.

| File | Relevant work |
|---|---|
| `pin_observation_pilot.py`, `prepare_pin_view_trial.py`, `run_depth_pin_views.py` | Detection/framing preparation, calibration/scene inputs and scan-supported target construction. |
| `analyse_drawer_areas.py`, `reconstruct_drawer_scans.py` | Measured lining/wall fits, observed free-area maps and scan registration. |
| `depth_pin_views.py` | `view_pattern`, `flange_for_view`, endpoint reachability and the early retreat/arc/approach design. The retreat design was unsuitable for the desired consistently close images. |
| `prepare_close_pivot_preview.py`, `prepare_live_close_views.py` | Close constant-distance preview and subsequent controller qualification. Their discretisation mismatch is documented below. |
| `prepare_standard_pin_views.py`, `compare_historical_pin_views.py` | Reuse of historical views after removal of the whole gripper. |
| `compile_rapid_pin_programme.py`, `rapid_pin_motion.py` | Known-station graph search, wire rounding, cubic/PVT timing and interval clearance. The retiming dependency is also preserved in `reference_source/retiming/`. |
| `rapid_pin_batches.py`, `rapid_pin_analysis.py` | Deterministic capture-bank/analysis loop, one resident model, cache/provenance and uncertain versus fault outcomes. |
| `rapid_pin_live.py`, `rapid_pin_capture_worker.py` | Original controller/camera adapter and persistent acquisition worker as reference code. |
| `pin_observation_batch.py`, `automatic_calibration_io.py` | Ordered route progress and complete-frame protocol repairs; compare frozen source with the later corrected version. |
| `scan_drawer_sides.py`, `drawer_motion_profile.py` | Successfully attended Oct7 survey implementation and correct transit profile. |

The source snapshot is for reading and porting. It retains sibling-workspace
imports, native SDK dependencies, frozen absolute paths and original controller
entrypoints. It is outside the installed Python package and is not invoked by
the handover tools. The supported offline entrypoint is the replay command above;
it never imports these controller/camera modules. The general planner should use
the four packaged models and current scene rather than replay old absolute poses.

## Failure evidence and lessons

[The detailed audit](RUNNER_FAILURE_AUDIT.md) preserves the October5 diagnosis,
timing table and repair proposal. Read its date: some issues were subsequently
fixed, and some remain. The original audit inventory/catalogue are also included.

| Finding | Evidence | Later outcome / design implication |
|---|---|---|
| False stall after reaching the endpoint | Oct5 `close_01_chunk_02.json`:114 feedback samples; final error0.002863mm/0.000152° while old route index stayed2 instead of3 | Ordered progress repair was replayed on412 samples/eight batches; Oct6 subsequently completed all10 captures. Keep endpoint and queue completion separate. |
| Speed override omitted from deadlines | Oct1 calibration timing repair | Use actual project override and ramp/communication time. Extending arbitrary global timeouts is not a repair. |
| Tight nominal FK/branch thresholds rejected motion | Oct1 long survey and Oct5 depth-view attempts | Represent model/registration uncertainty and verify branch/clearance coherently. A residual is not itself collision evidence. |
| Binary/text feedback dispatch failed | Pilot v2 traceback and subsequent protocol repair | Parse complete frames using response mode. An incomplete feedback packet must not be treated as valid pose. |
| Repeated coordinates reused transaction identity | Oct5 two-drawer rescan:11 pairs including the separate home pair, final non-capture return rejected | Give each visit/command its own identity. Preserve capture completeness independently of final-return status. |
| Requalification and precise stops dominated elapsed time | Oct6 complete run:32 batches;129.85s command records,76.68s approximate workers,191.94s unallocated orchestration | Compile once, stop at photo stations and retain only live-dependent checks in the loop. Unallocated time cannot all be called safety checking. |
| Close Photoneo scan had zero useful depth | Close reference approximately262mm from scanner, versus recorded366–558mm range | Use survey depth for close EIH planning; retain per-view depth validity. Do not force the EIH camera away just to obtain another depth frame. |
| Thin/occluded pin could not be fitted | Recent14-view replay remains uncertain; historical candidate identity pending | Separate negative vision results from control/acquisition faults. Add prepared angles automatically only for uncertainty. |
| Survey reused the calibration speed profile | First Oct7 side sweep94.19s | Later GUI transit removes the unintended low-speed profile; successful side sweep44.58s includes acquisitions and checks. |
| Authorised speed-limit command rejected by calibration filter | First drawer002 side session | Narrow existing-transport correction, then successful retry. A successful retry does not erase the first error. |
| User cancellation / shutdown | STOP files and stopped session records | Preserve these as deliberate stops, not algorithm failures or resumable jobs. |

## Coverage, provenance and remaining work

`configs/experiments.json` inventories copied records,43 saved session records,
reused existing payloads and omitted dense acquisition metadata. Session records
include offline cold/warm benchmarks and partial continuation roots; they are
not43 independent physical experiments. Original files are copied byte-for-byte
and individually hashed. More close-view/exposure/pilot photos are provided; the
existing117 historical images and14 recent pairs are reused where identical.

This is a diagnostic supplement, not a second full raw-scan backup. Large dense
metadata are explicitly listed as omitted; older full depth arrays, original
robot assets and every historic image are not all supplied. Six Oct7 drawers and
the14 recent pairs retain the raw scans already delivered. Retained errors can
repeat the same event; optional unavailable-camera-setting warnings are not robot
faults. Missing/overwritten logs limit reconstruction of the full lab day.

The next integration step is to turn the existing framing/angle code into a
portable arbitrary-target planner: calibrated image/scan association, current IK,
full route clearance and visibility for all four models, then one compiled mission
used for preview and eventual deterministic execution. Benchmarks should measure
visible-pin success, rejected/unreachable targets and end-to-end time across centre,
edge and crowded specimens. Current data provide examples and counterexamples,
not a qualified final product or physical pickup workflow.

> Historical audit written on5October2026. Later repairs and successful captures
> are listed in [EXPERIMENTS.md](EXPERIMENTS.md). This copy adapts links and
> operational wording; numerical results and historical conclusions are retained.
> Some referenced large scans/assets are outside the diagnostic supplement.

# Robot runner error and timing audit — 5 October 2026

The recent interruptions include several confirmed software faults and false
stops. The records support the user's complaint: increasing the number of
checks did not make this workflow reliable, and repeated qualification between
small movements made it unnecessarily slow. Today's final stop happened after
the arm had already reached the commanded endpoint. It was not evidence that
the arm could not perform the move.

This is an offline review requested after the user stopped for the day. No
robot/camera connection, movement, capture, simulation, recovery, commit or push
was made during the audit. Runtime control behavior and thresholds have not
been changed by this review. Proposed fixes below are not installed fixes.

## What the user actually asked the workflow to do

For one specimen: use the existing detection as a framing point, descend once,
stay close, pivot around that point through ten useful viewing directions, stop
only long enough to capture a lit EIH image and Photoneo scan at each station,
then run the existing pin-orientation analysis. The pattern should be a reusable
automatic program, translated to other specimen positions later. It should not
need manual approval or execution of each waypoint.

The most recent approved Isaac draft keeps the EIH lens approximately
232.559 mm from the target, with angles up to 42 degrees where the mounted
assembly allows it. It retains the user-selected calibration estimate, all
fresh drawer scan geometry, a 48.843 mm rim ceiling and the existing 35 mm model
reserve. These are the recorded development assumptions, not a newly verified
physical accuracy bound. No specimen pickup was part of this run.

## Evidence coverage and limits

The final inventory is
[inventory.json](../data/experiments/records/20261005-robot-runner-audit-v4/inventory.json).
It hashes all 1,706 retained JSON/log files under
`/home/ai-user/Desktop/SpecimenWorkcell`, excluding the audit's own output roots.
It covers 21 session records, 25 terminal exception lines and 29 structured
failure entries. The 54 failure manifestations include repeated representations
of the same event, user cancellation and a diagnostic description; they are
**not 54 distinct robot faults**. A separate 134 entries record optional camera
settings reported as unavailable. No JSON parse failure occurred.

[error_catalogue.tsv](../data/experiments/records/20261005-robot-runner-audit-v4/error_catalogue.tsv)
gives every extracted entry's source, log line or JSON pointer, and error text.
The full sources remain intact. The inventory was produced by the filesystem-only
[audit utility](../data/experiments/reference_source/workcell/scripts/audit_saved_runs.py), which does not import device
control code. The earlier v1–v3 inventories are superseded: v1 missed failed
command intervals; v2's recursive search incorrectly counted schema descriptions
as failures; v3 omitted settings warnings and the old stop timeout. Those audit
errors were corrected before using these results.

Coverage is of retained files, not every terminal command or every exception
that ever appeared in chat. Preparation logs were sometimes overwritten, and
some old motion failures omit their end timestamp. There are no complete phase
timings for all setup/qualification calls. A preparation `FileNotFoundError`
for the missing V7 EIH calibration path is recorded in the session history but
not the surviving successful preparation log. These gaps prevent a claim that
every historical error or every minute of the lab day is accounted for.

## Today's final false stall: demonstrated cause

Source:
[close_01_chunk_02.json](../data/experiments/records/20261005-close-pivots-live-v1/batches/close_01_chunk_02.json),
with the traceback in
[execution.log](../data/experiments/records/20261005-close-pivots-live-v1/execution.log).
The earlier `close_01_chunk_01` completed successfully; it is not the failed batch.

The failed batch commanded four legs totalling 12.4874 mm. Feedback reports
12.4870 mm start-to-end displacement and 2.5753 degrees of orientation change.
The final pose met the runner's 0.1 mm / 0.05 degree arrival tolerance after
2.5244 seconds of recorded feedback; three qualifying consecutive samples were
available by 2.7429 seconds. Ninety-one samples remained within that tolerance.
The last reported error was only **0.002863 mm / 0.000152 degree**.

The last leg was 0.25293 mm / 0.05216 degree long. Its endpoint also falls inside
the preceding leg's looser 0.5 mm / 0.3 degree path acceptance region.
[BatchGuard](../data/experiments/reference_source/workcell/scripts/pin_observation_batch.py) searches forward and accepts the
**first** matching leg. Consequently it stayed on the previous leg even while
the arm was at the final endpoint. Its arrival condition also requires the
final leg index, so its settled counter stayed at zero. Progress was measured
against the obsolete leg endpoint; the watchdog fired 10.0601 seconds after its
last progress checkpoint.

An independent
[saved-feedback replay](../data/experiments/records/20261005-robot-runner-audit-v3/latest_failure_replay.json)
reproduced the stop at zero-based sample index 113 of 114; its
[reproduction script](../data/experiments/records/20261005-robot-runner-audit-v3/reproduce_latest_failure.py)
verifies the six relevant source hashes against the frozen physical run and
blocks network/process activity.
Maximum feedback interval was 0.11394 seconds and maximum acquisition duration
was 0.09668 seconds. These are within the existing feedback limits; this record
does not show a feedback starvation problem. Geometry checks were stubbed for
the replay, so it demonstrates the progress/arrival defect, not new collision
qualification.

The command acknowledgement completed. `QueueTag` completion was **not queried**:
the code only reaches that query after `guard.done`, which never became true.
The evidence establishes reported endpoint arrival, not successful queue
completion or completion of the photographic view. Stop acknowledgement,
stationarity and light restoration to zero were recorded. Only the
`close_reference` image/scan pair was captured.

The appropriate correction is ordered route advancement that handles
overlapping acceptance regions, including short final legs, followed by final
pose settling and controller queue completion. A last-match search alone could
skip forward on a reversing or self-crossing route; an endpoint-only success
test could miss incomplete queued motion. Neither is an adequate general fix.
Keep the watchdog's current thresholds while correcting the state it measures.

## Failure ledger

Run/source names below are abbreviated. The linked catalogue contains the full
paths, all duplicated log/session/motion manifestations and exact line numbers.
The disposition column distinguishes an already recorded correction from a
proposed one; success on one short move is not proof of a full workflow.

| Event and retained source | Finding | Disposition |
| --- | --- | --- |
| 29 Sep, `load_update_mass_only/record.json`: compensatory payload differs | The write set TCP mass to 2.55732 kg and the reported additional payload stayed 0. The first interpretation expected a different relationship between these fields. This was configuration interpretation, not a failed motion. | Later reconciliation exists; use the established mass setter/readback rather than recreate this experiment. Centre of mass remains an estimate. |
| 29 Sep, `calibration_tilts/motions/xy_m5_m5.json`: incomplete TMSVR frame; subsequent stop timeout | A protective-state read could not parse a complete frame. This record also contains `stop_error=TimeoutError('timed out')`; it does not prove a stop ACK. | Transport/control fault unresolved by this record. Do not bypass missing protective feedback. Later sessions' verified stops are separate evidence. |
| 29 Sep, `isaac_mount_preview_v2.log`: static preview contains physics | An imported cylinder retained a physics schema despite the visual-only preview requirement. | Offline asset-composition issue. No robot stop or new physical prerequisite follows from this. |
| 30 Sep, `photoneo-reconstruction/render.log`: flange axes differ by 0.569862 degrees | Nominal URDF axes and reported controller axes differed. A strict visual check rejected the initial render. | Offline model-registration issue; the discrepancy must be represented as model uncertainty, not proof of a bad physical branch. Later renders exist. |
| 30 Sep, `photoneo-reconstruction/render_v3.log`: viewport capture failed | The screenshot/export call failed at `wrist-clock-top`. | Offline rendering/API issue, not arm failure. |
| 30 Sep, `moving-camera/audit-v1.log`: missing scipy | Audit launched in an environment lacking the dependency. | Environment selection error; use the existing appropriate Python environment. |
| 30 Sep, `moving-camera/audit-v2.log`: incompatible `within_cspace_limits` arguments | Incorrect SDK API argument types in the offline audit. | API integration error; later v3/v4 audit files exist. No hardware fault implied. |
| 30 Sep, `automatic-calibration/frozen-worker.log`: string has no `read_bytes` | A path serialized into worker JSON was treated as a Path object. | Recorded worker-path conversion fix; isolated worker validation was added. |
| 30 Sep, `automatic-calibration/rehearsal-v2.log` and `rehearsal-v3.log`: self-collision proxy rejects link_0/link_2 | Full interval enclosure was too coarse for some proposed paths. A rejection does not establish a physical collision. | Offline proposal rejection; refine sampling/geometry with the same clearance, or choose another route. Do not delete the collision check. |
| 1 Oct, `automatic-calibration-runs/20261001T110650/session.json`: generic Listen/base/TCP/protection failure | Saved preflight shows RobotEndFlange mass 0 after reset; the established assembly is 2.55732 kg. Protective fields shown were clear. The generic error obscured the actionable field. | Valid configuration mismatch, poor diagnostic. Known load restoration is established and should be a single startup step. |
| Same startup, `restore-known-load.json`: non-allowlisted calibration command | `ChangeTCP` was rejected by a helper intended for calibration motion. No motion was sent. | Wrong helper for an authorised configuration action; later dedicated setter used. Keep command scoping, choose the proper setter automatically. |
| 1 Oct, `20261001T110901/operations/0003_analyse/worker.log`: invalid association/hash/EIH settings | Origin EIH shutter read 10323 while frozen acquisition expected 5000. The generic validation message combined several possible causes. | Relevant capture/configuration mismatch, not a motion fault. Record the exact differing setting; restore once before acquisition. |
| 1 Oct, `20261001T111035/session.json`: board identification failed | Stationary image quality/recognition failed; focus readback did not itself guarantee a sharp frame. | Calibration measurement failure; focus was reapplied and later boards recognized. It must not gate unrelated drawer motion. |
| 1 Oct, `20261001T111241/motions/0034.json`: completion deadline | A 175.36 mm move at command 5 mm/s and override 45% had actual nominal duration 77.94 s. The old 47.07 s deadline assumed the unscaled speed. | Confirmed false timeout. Override-aware timing was installed and saved replay plus attended recovery passed. Do not fix this by increasing physical speed. |
| 1 Oct, `20261001T114445/motions/0038.json`: unexpected joint branch/path | Maximum saved J3 nominal residual 0.251747 degrees slightly exceeded the old 0.25 degree bound; reported endpoint was approximately 0.0017 mm from target. | False inference from nominal-model error. J3 bound changed to 0.5 degrees in both feedback and qualification; other bounds retained. This did not solve all later model drift. |
| 1 Oct, `20261001-recovery-path-01/motions/0001.json`: joint branch/path | Feedback was acquired by sequential reads. In the last sample, Tool and Flange orientation differ by approximately 0.103 degrees over a 0.248 s acquisition, with joints read earlier. Asynchronous feedback is evidenced and is a suspected cause; stationary model error is not established as this stop's cause. | Separate failed continuation retained. Later `calibration-path-fix/batch-read-diagnostic.json` demonstrates a combined response with matching Tool/Flange values. Coherent feedback acquisition is necessary; the earlier timeout repair alone was incomplete. |
| 1 Oct, `calibration-path-fix/qualification.log` and `qualification-v2.log` | Three link_0/link_2 enclosure rejections: gap/enclosure 8.253/9.059, 7.194/6.503, 7.438/6.740 mm. Last two leave less than the 1 mm required net clearance. | Subsequent route/enclosure corrections passed full qualification without reducing the clearance reserve. The successful source also includes joint-specific bounds and corrected reversed-origin fallbacks; success cannot be attributed to sampling refinement alone. These were not three physical collisions. |
| 1 Oct, `second-drawer-survey/motions/0001.json`: joint branch/path | A long route used one home alignment. J3 residual reached approximately -0.500306 degrees; nominal/model registration varied with configuration. | Saved local-anchor replay supports short stationary-anchored sections. Continuation passed 1,166 saved feedback samples and captured three more views. Nominal model needs a better contract for efficient longer routes. |
| 1 Oct, `pin-observation-pilot/qualification_attempt_1/diagnosis.json`: outside origin envelope | Invented extra-Z allowance was 5 mm, but the authorised 6-degree camera-centred arc requires approximately 10.07 mm extra flange descent because of the camera offset. | Task-envelope specification bug; derive the bounded envelope from the requested camera motion and mounted assembly. Not a controller limit. Duplicate `prepare.log` records describe this same attempt. |
| 1 Oct, `pin-observation-pilot/qualification_attempt_2/prepare.log` and duplicate `prepare-v2.log` | Link_0/link_2 gap 7.900 minus enclosure 6.907 mm gives 0.993 mm, just below the required 1 mm. | Offline conservative enclosure rejection; bounded sampling refinement, not tolerance erosion. |
| 1 Oct, `pin-observation-pilot/motions/0000.json`: Cartesian path | A 57.66 mm translation with only about 0.001416 degrees rotation was classified as rotational progress. Tiny angular noise distorted path fraction although measured lateral deviation was about 0.0103 mm. | Confirmed progress-parameter bug. Translation progress now selected when rotation is below arrival resolution; angular checks remain. |
| 1 Oct, `pin-observation-pilot-v2/motions/0008.json`: Unicode decode | Unsolicited binary mode-1 status was decoded as requested mode-12 text. | Confirmed packet-dispatch bug. Mode-aware parsing installed; checksum, matching transaction and deadline retained. |
| 5 Oct, `one-specimen-lit-v1`: Cancelled | User requested changing the speed; the current run was deliberately stopped and two captures retained. | Expected cancellation, not a spurious protective trip. STOP must not be removed to make the statistics look better. |
| 5 Oct, `one-specimen-depth10-v5/batches/near_reference.json`: misleading Cartesian error | Last sample on the actual leg had only 0.0000993 mm path error, but J1 residual 0.2500268 degrees exceeded 0.25. The old search replaced this with the last unrelated future-leg rejection. | Two software issues: nominal tolerance/model mismatch and lost diagnostic cause. All candidate rejections are now retained; V7 short anchors replaced the unsafe proposed adaptive-offset approach. Latest short-leg completion bug still remains. |
| 5 Oct, `one-specimen-depth10-v7`: expected waypoint mismatch | User rejected the distant/retreating view pattern, stopped it and sent the arm home. The saved runner then found a pose different from its expected waypoint. | User interruption/task correction, not evidence that an otherwise intended move failed. Never auto-recover this root. |
| 5 Oct, `two-drawer-rescan-v1/session.json`: never resend batch | All eleven camera pairs were captured. The final noncapture centering leg reused a pose-derived label belonging to an earlier command; collision detected before sending it. | Confirmed bookkeeping bug. Future scan labels include a unique motion index, verified for all 23 planned identities. No physical rerun of that fix yet. |
| Same scan, `processing.log`: capture stopped/no continuation | Offline reconstruction treated the motion root's failure status as invalidating its complete capture set. | Acquisition/control status coupling. A separate hash-bound capture-completion review enabled offline processing and preserved the original motion error. Both drawers reconstructed; no robot continuation. |
| 5 Oct, `close-pivots-live-v1/batches/close_01_chunk_02.json`: no net progress | Reported final pose reached in 2.52 s, but overlapping path tolerances kept the guard on the previous leg. | Confirmed false stall, still unresolved in runtime. See replay explanation above; fix first. |
| Optional camera settings, 67 JSON records / 134 entries | `InterreflectionsFilterStrength` and `InterreflectionsFiltering` reported `not available`. These are retained setting queries, not 134 capture failures. | Nonfatal device-version/capability warnings. Discover supported controls once; do not promote optional unsupported settings into motion prerequisites. |

Additional implementation findings that lack their own retained terminal log:

- The light helper had the same first-packet binary/text dispatch problem as
  the pose reader; corrected on 5 October before the lit runs. Consolidate
  protocol handling rather than copying independent fragile parsers.
- Independent review caught nested EIH operation locking and child cancellation
  problems before the ten-view physical attempt. Parent-held camera access must
  pass its ownership to the worker; workers must receive parent stop promptly.
- A feasible endpoint did not guarantee a feasible tenth transition. Continuous
  transition feasibility was subsequently included. This is valid route rejection,
  not a reason to discard IK/clearance checks.
- V6's proposed adaptive joint offsets passed a saved-feedback comparison but
  could leave the originally qualified collision tube. Review rejected it; V6
  was never executed. A passing residual replay alone was inadequate evidence.
- The early ten-view planner increased camera distance and returned upward
  between views to satisfy the combined geometry/Photoneo viewing objective.
  That did not match the user's close EIH observation request. The constant
  distance Isaac draft corrected task design; it did not automatically correct
  the physical runner's execution semantics.
- The close Photoneo frame at the earlier reference contained zero valid depth
  points. The target was approximately 262 mm from Photoneo, closer than the
  recorded 366–558 mm device range. A saved image/scan can legitimately have no
  useful depth. Do not force EIH farther away to make Photoneo produce depth;
  use the prior survey for geometry and record depth validity per close view.
- The live close-view preparer initially referenced an absent V7 calibration
  file. It was changed to the retained V5 source. The overwritten log limits
  reproducibility of that exception, which should be prevented by input checks
  before presenting the route as ready.

## Where the time went

All times are seconds. Session wall time is its saved start/end interval and
does not include earlier design, preview, route preparation, analysis after the
session or conversation gaps. Command intervals include monitoring, ACK/queue
handling and ramp/stop time; they are not pure mechanical travel. `>=` adds the
known portion of failed records whose end timestamp is missing. Worker time is
an approximate request-file to result/log-file modification interval. The
remaining column is unallocated orchestration time, or an approximate upper
bound on it where a command end is missing. It cannot all be called safety checks.

| Session | Wall | Command intervals | Workers approx. | Unallocated approx. | Captures in saved state |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 Oct startup 110650 | 1.34 | 0 | 0 | 1.34 | 0 |
| Startup 110901 | 15.01 | 0 | 9.51 | 5.50 | 0 accepted |
| Startup 111035 | 39.65 | 0 | 31.34 | 8.31 | 0 accepted |
| Calibration 111241 | 607.46 | >=168.78 | 362.53 | <=76.14 | 6 |
| Recovery timing 01 | 79.78 | 53.37 | 7.50 | 18.92 | 1 |
| Calibration 114445 | 668.20 | >=232.70 | 345.11 | <=90.40 | 6 |
| Recovery path 01 | 31.46 | >=3.61 | 7.19 | <=20.67 | 1 |
| First drawer Y sweep | 336.90 | 285.76 | 36.39 | 14.75 | 6 |
| Second drawer survey | 142.53 | >=128.94 | 6.05 | <=7.54 | 1 |
| Stationary capture after stop | 8.76 | 0 | 6.02 | 2.74 | 1 |
| Second drawer continuation v2 | 366.98 | 268.25 | 17.99 | 80.74 | 3 |
| Pin pilot v1 | 41.45 | >=0.22 | 0 | <=41.23 | 0 |
| Pin pilot v2 | 292.43 | >=151.15 | 4.64 | <=136.65 | 2 |
| Pin pilot v3 | 231.85 | 101.53 | 7.07 | 123.24 | 5 total; 2 retained |
| 5 Oct lit v1 | 204.15 | >=108.81 | 4.77 | <=90.57 | 2; user paused |
| Lit fast v2 | 194.51 | 69.61 | 7.17 | 117.73 | 5 total; 2 retained |
| Batch check v3 | 23.16 | 6.08 | 2.38 | 14.71 | 6 total; 5 retained |
| Depth10 v5 | 28.53 | >=12.27 | 0 | <=16.27 | 0 |
| Depth10 v7 | 301.90 | 110.77 | 40.51 | 150.62 | 2; user stopped |
| Two-drawer rescan v1 | 321.80 | 118.24 | 67.76 | 135.79 | 11 pairs: home + 10 in list |
| Close pivots live v1 | 127.03 | >=45.47 | 20.30 | <=61.26 | 1 |

The saved sessions total 4,064.89 seconds, or 67.75 minutes across days. This is
not the complete time the user spent waiting. Large preparation and conversation
gaps are outside these session clocks.

Specific avoidable contributors are visible in the code and records:

1. **Repeated setup between micro-moves.** The current
   [qualify_chunk](../data/experiments/reference_source/workcell/scripts/pin_observation_batch.py) reruns full stationary
   preflight, reanchors and recomputes checked-leg IK/clearance before each short
   batch. Runtime view loops also rehash the frozen dependencies. The earlier
   pilot repeated similar work per approximately one-degree leg. This explains
   why raising the speed did not remove long pauses, although the logs cannot
   allocate every setup second to an individual call.
2. **Observed pauses during the latest descent.** Six approximately 50 mm
   sections took about 4.75 seconds each, separated by gaps of 4.34–4.86 seconds,
   with one gap of 15.69 seconds. They were automatic, but visibly stepwise.
   Source records contain this repeated stationary qualification explicitly.
3. **Precise-stop path discretization.** Every emitted `Line("CAP",...)` uses
   blend 0. The close preview becomes 148 fine legs and 32 short batches. Fine
   geometry samples were turned into physical precise-stop commands, so a
   smooth-looking animation did not imply a smooth controller trajectory.
4. **Real override and ramp cost.** Command 30 mm/s at 45% gives at most
   13.5 mm/s commanded scale; observed fast travel was about 13.53 mm/s.
   Earlier command 5 mm/s at 45% gave 2.25 mm/s. Each precise stop adds ramps.
   This cost must be estimated, not treated as a mysterious arm delay.
5. **Acquisition has its own cost.** The latest reference EIH worker took
   approximately 16.82 seconds including autofocus/setup; Photoneo took 3.48
   seconds. Other basic EIH capture workers took roughly 2–3 seconds. Per-view
   autofocus, process/SDK startup and file writing deserve measurement and reuse.
   They cannot currently be assumed to take a fraction of a second.
6. **Preparing the physical route too late.** After the user approved the Isaac
   animation, live command qualification took another 116.58 seconds. The draft
   animation lasts 17.2 seconds, but the generated plan estimates 56.23 seconds
   nominal motion and 359.03 seconds total with capture/overhead assumptions.
   These are different quantities. The animation was an unhelpful predictor of
   live elapsed time. This qualification should have been completed alongside
   the preview, before calling it ready to enact.
7. **Large, unnecessary calibration route.** The original plan used hundreds of
   segments and long returns/orbits. A later compact calibration reduced travel
   to 505 mm and 202 segments. That earlier lesson was not carried through to
   the new close-view controller path.

## Which checks to retain, replace or stop repeating

This is a proposed disposition, not a new set of barriers. Existing working
rules already say to apply only checks relevant to the requested action.

| Existing behavior | Disposition | Reason / replacement |
| --- | --- | --- |
| User/pendant stop, controller protection and exclusive motion ownership | Retain | Actual control boundary; never reinterpret a user stop as a transient failure to retry. |
| Known tool/load, TCP/base readback | Retain once at startup; verify changes when relevant | The reset-to-zero mass is real. Restore the already authorised known profile automatically through the established setter. Do not repeatedly ask for the same mass. |
| Lost, malformed, stale or nonfinite motion feedback | Retain; improve protocol implementation | A real incomplete-frame incident exists. Use mode-aware complete frames and coherent pose/joint acquisition. One combined response is not proof of hardware timestamps being synchronized. |
| Watchdog and override-aware deadlines | Retain; correct their progress state | Today's false stall comes from wrong segment attribution. Removing or stretching the timer would conceal that defect. |
| First-match segment assignment | Replace first | Cannot distinguish the short terminal leg from the previous acceptance region. Use ordered progress plus completion evidence, tested on reversing/overlapping routes. |
| Tight nominal-joint residual used as proof of wrong branch | Replace or properly qualify | Its small thresholds repeatedly stopped valid-looking Cartesian motion. Model discrepancy, real branch identity and collision bounds need an explicit consistent contract. Do not simply delete this test while relying on the same nominal tube for clearance. |
| Full preflight/replanning at every tiny section | Remove duplicate work where unchanged; preserve needed anchor/state checks | Precompile stable geometry and command structure once. Current local reanchor affects IK and collision qualification, so it cannot be dropped until that dependency is resolved. Use a bounded measured alignment and lightweight checks; requalify only what actually changed. |
| Fine geometry sample equals precise-stop physical command | Replace command generation | Controller-supported continuous paths/blends need their swept envelope qualified. Stop at photographic stations, not at every visualization/sample point. Do not blindly change blend from zero. |
| Full dependency set rehashed per view | Move to immutable mission validation/startup where suitable | Preserve source/plan identity and prevent mid-run mutation. A stable executable bundle avoids repeatedly checking irrelevant renderer files. |
| Pose-derived transaction labels | Replace identity generation | Distinct visits to the same coordinate are distinct commands. Assign immutable mission/command IDs before execution; retain anti-resend handling. |
| Motion success required for offline use of complete capture set | Separate states | Capture completeness, final holding/return outcome and pin-analysis quality are different results. An unexecuted final centering move does not invalidate eleven saved pairs. |
| Board detection, calibration fit and required camera settings | Apply to that measurement only | Needed for calibration acceptance, not unrelated drawer motion. Unsupported optional filters are warnings. |
| Fresh scan validity used for clearance | Retain where geometry is needed | The old survey can support close observations within its uncertainty; a zero-depth close frame must not silently be treated as free space. |
| Generic 120 s batch and two-hour session caps | No logged trigger; review their concrete purpose during runner repair | They did not cause the diagnosed stops. Do not invent larger scopes or remove them as an unexplained cure. The relevant timing change is actual motion/worker deadlines. |

## Proposed repair order for tomorrow

This is a repair agenda for discussion and implementation. It does not require
the user to approve every subsequent waypoint, and it does not authorize an
overnight hardware run.

1. **Fix the reproduced completion state machine first.** Replay the exact
   failed short terminal leg; also exercise overlap, reversals/self-crossings,
   a genuine intermediate stall, invalid/lost feedback and an incomplete queue
   tag. Preserve the existing geometric and watchdog thresholds. Acceptance:
   reported final settling is recognized, queue completion is required, and real
   control faults still stop. No full calibration recollection is needed.
2. **Separate durable command identity and result states.** Preassign unique
   command IDs including repeated coordinates; record every intended/sent/ACK/
   queue-complete/settled/captured transition with timestamps. Save capture
   completeness independently of the optional end movement. Validate the retained
   scan collision case offline; never resend an ambiguous prior transaction.
3. **Resolve the model/anchor contract before reducing batch boundaries.**
   Replay the existing long-drawer and V5 false positives. Compare controller
   joint/pose consistency, branch continuity and modeled collision clearance
   with quantified registration error. Qualify any changed bounds in both
   monitoring and collision envelopes. Keep adaptive-offset V6 rejected unless
   a new consistent clearance argument is supplied.
4. **Produce one compiled mission for preview and execution.** It should contain
   the target-relative stations, transitions, controller commands, clearance
   assumptions, source identity and realistic timing. Include measured override,
   ramps, stationary capture and required residual anchor checks. Generate it
   before the lab run; do not regenerate the route after the user likes a video.
   The controller executes station transitions; the runner handles arrival and
   capture automatically, with brief state/progress messages.
5. **Reduce execution and camera overhead.** Keep immutable scene/model objects,
   combine appropriate readbacks and reuse camera connections/focus where valid.
   Plan smooth transitions with the controller's supported path semantics and
   qualified blending envelope, stopping only at the ten capture stations.
   Add phase timings before claiming a new total duration. Start with the
   established speed profile; speed tuning is separate from software reliability.
6. **Prove the affected change on a short attended route, then the full approved
   sequence once.** Hardware validation needs a fresh lab start because the user
   ended the lab session and said they were turning the arm off. Use a fresh
   evidence root and the approved
   close-view geometry, not an automatic restart of a stopped root. The meaningful
   completion criterion is ten actual image/scan records plus a truthful pin-fit
   result, not merely a process exit or attractive Isaac animation.

The first repair is narrow and diagnosable offline. Broader changes to collision
bounds, blending or anchor assumptions require the existing proportionate design
review because they change what geometry the runner actually guarantees. That
review belongs before implementation, not between photographic waypoints.

## Audit validation

All 1,706 inventoried source hashes were rechecked against the retained files;
all matched. Session intervals, failed-record lower bounds and entry counts
passed consistency checks. Fourteen top-level JSON arrays were also searched
and contained no missed structured errors. The reproduction's input hashes and
six frozen runtime source bindings matched. The offline audit utility was copied
into the v4 evidence directory, binding it to the inventory's auditor hash.

Independent review agreed with the reproduced completion defect, timing
definitions and proposed check dispositions. It corrected the recovery-feedback
and historical qualification wording above. Neither the audit nor its review
implements or validates replacement control behavior.

## Handoff state

- The user stopped live work and said they are turning the arm off. The last
  verified state before shutdown was stop ACK, stationary feedback, light 0 and
  runner PID 0. This audit made no later hardware query.
- `20261005-close-pivots-live-v1/STOP` and
  `SESSION_STOPPED_BY_USER.json` remain. Ten-view run incomplete; one pair saved;
  no pin solver result, accepted 3D pin axis or pickup from it.
- Fresh drawer survey remains usable: eleven pairs, 5,025,072 valid observations,
  registered cloud (older large asset not included in this supplement: `20261005-two-drawer-rescan-v1/scan-v1/registered_full.npz`)
  and Isaac drawers (older large asset not included in this supplement: `20261005-two-drawer-rescan-v1/wide-v1/two_drawers.usda`).
- Approved close-view draft remains at
  [trajectory.json](../data/experiments/records/20261005-close-pivots-isaac-v2/trajectory.json)
  and Isaac animation (older large asset not included in this supplement: `20261005-close-pivots-isaac-v2/isaac-v1/close_pivots.usda`).
  It is a geometry preview, not a proven end-to-end physical workflow.
- Tomorrow's starting point is the demonstrated completion bug and this repair
  agenda. The execution requirements already prohibit redundant approval loops
  and nominal discrepancies being called collisions without evidence. The
  remaining work is to correct the implementation; adding another rules document
  alone will not make these faults disappear.

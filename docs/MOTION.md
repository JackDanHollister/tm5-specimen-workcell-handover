# Motion information

Each robot variant contains its source joint limits in `robot.urdf`, with planning
geometry and cspace limits in `robot.xrdf`. Read these files rather than copying a
single historical pose range to every task. Existing demo workcell restrictions
(for example its base-joint sector) are task constraints, not extra robot joints.

The Oct7 repeated drawer survey is a high, fixed-height/orientation route:
centre -> base-Y -100mm -> centre -> base-Y +100mm -> centre. One lit EIH photo
and a centre Photoneo scan precede movement; the two sides each take another scan.
Each new drawer is placed by the operator after the prior sequence returns.

Validated attended transit setting: GUI nominal **2400mm/s**, **300ms** ramp and
the operator's **45%** project override. Short moves do not reach a constant
1080mm/s cruise; the four recorded motion transactions took roughly2s each,
including feedback/acknowledgement overhead. One successful two-side sweep took
44.58s including per-move geometry checks and two camera acquisitions. Camera and
validation time must be included in end-to-end estimates.

The earlier scanner runner incorrectly used30mm/s plus an additional TCP limit,
yielding13.5mm/s at45%. That limiter was disabled for GUI transit. Do not silently
reintroduce the close/calibration speed profile for an overhead drawer survey.
Close specimen viewing is a different geometry/speed task.

The recorded `data/drawers/*/recorded_route.json` retains actual joint feedback,
sample times and controller command text. Commands use flange/base coordinates,
positions in millimetres, XYZ Euler angles in degrees (Rz Ry Rx convention), and
explicit controller velocity/ramp parameters. `Line("CAP",...)` honours project
override. These are reference specifications; this repository has no live sender.

The old pin-view planner used a35mm development scene/rim reserve (the requested
30mm clearance plus5mm allowance), original joint tracking/model tolerances and
coherent feedback. Those assumptions are retained as context, not promoted to
universal certified uncertainty. Box sizes/heights, attached tool geometry,
occluded contents and full swept paths must determine new inspection routes.

Routine execution should be deterministic move -> settled capture -> next pose,
with model/inference loaded once and batch analysis. A negative fit may advance
to another prepared view batch; control/acquisition/provenance faults terminate.
Use actual override, ramps and communication latency for deadlines. Avoid repeated
qualification of identical immutable inputs; current pose/scene-dependent checks
remain relevant.

Physical handoff must preserve exclusive control, correct load/TCP, controller
protections, meaningful continuous feedback and immediate operator stop. STOP
cancels the sequence; no automatic return/restart after it. Simulator success and
historical camera alignment do not qualify grasping, pickup or insertion.

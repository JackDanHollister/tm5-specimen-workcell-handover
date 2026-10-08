# Development mission

Build a reusable planner that accepts a registered drawer scan/photograph and one
of the four hardware configurations, then works out useful, efficient routes for
observing and approaching specimens throughout that drawer.

The first deliverable is **camera-positioning and pin observation**. Gripper
variants should also expose whether an approach along a recovered shaft direction
is feasible. Contact/grasp/insertion qualification is subsequent work.

## Input and output contract

Inputs: drawer geometry and its base-frame registration, EIH image/calibration,
specimen framing detections, selected robot/tool model, current joint pose and
movement constraints. A scannerless configuration can consume a previously
registered scan; measuring an unfamiliar drawer using EIH alone is a separate
perception problem.

Outputs: ordered view/approach candidates, solved joint/Cartesian waypoints,
declared units/frames, capture stops, complete swept-path checks, estimated timing
and a reasoned report for every specimen. Distinguish reachable, uncertain because
of missing/occluded measurements, and geometrically unreachable targets.

## Suggested work

1. Find specimens in the EIH image; associate observations across views without
   confusing a specimen centre with the actual handling-pin line.
2. Register depth into EIH pixels and estimate drawer boundaries/rim/lining and
   useful framing heights. Preserve unseen regions.
3. Instantiate historical view patterns relative to each framing point. Choose
   camera roll, side, inclination and supported distance for the selected tool.
   Successful historical pin views were often 60–80 degrees from straight down;
   those angles are examples, not a requirement for every specimen.
4. Solve IK and check the entire arm, attached camera/support/QC/gripper, drawer
   and neighbouring contents along the full route. Consider sensor field of view
   and visibility, not only endpoint reachability.
5. Rank useful view batches by travel and acquisition time. Collect a batch,
   analyse once, and choose another prepared batch only when evidence is uncertain.
6. Preview and export a deterministic movement/capture specification. Include
   target/model identities, units, speed/ramp settings, and unfinished checks.

## Evaluate across all cases

Exercise all six drawers and all four configurations, including edge/corner
specimens, varied apparent heights, occlusion and neighbouring specimens. Report
coverage, useful independent views, rejected routes/reasons and timing. Use
negative cases and a held-out placement/drawer when evaluating generalisation.
Small residuals on development images are not independent accuracy.

The historical 20-view sets and successful seven-axis fits provide initial evidence.
The Oct6 close-view library was uncertain; the Oct7 drawers provide fresh scenes
and overhead images. The general planner is the new work, not a finished feature
claimed by this handover.

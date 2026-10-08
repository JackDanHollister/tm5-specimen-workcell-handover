# Handover validation

The package is tested offline; no robot/camera/controller is contacted.

- All24 drawer/model USD compositions open with their dependencies inside the kit.
- All URDF mesh references resolve inside the kit. XRDF spheres, default joints,
  tool frames and self-collision exclusions refer to retained links/joints.
- Removed scanner/support and gripper geometry, physics joints and relationship
  targets are removed together. All variants retain QC/EIH and the23.15mm stack.
- Zero-pose USD/URDF FK agreement is below9e-8 in maximum matrix element error.
- Four configurations were imported into Isaac Sim6.0.1.0 and checked at the
  recorded home and two nonzero poses. Maximum live-articulation FK discrepancies
  were below4e-7 in matrix element magnitude. Camera/mount/QC followed the flange;
  registered drawer geometry remained fixed.
- The saved-observation example reproduces all seven provisional historical shaft
  fits; maximum direction change is below0.001degree.
- The relocated checkout installs/verifies all1,500 payload files. All24 USD
  combinations resolve inside that checkout and the seven-axis replay passes.
  A Python file-access audit blocks the original data/source paths during replay.
  Private release hashes are checked separately before publication.

Normal CPU CI tests cover frame conventions, attachment motion and invalid trees.
GitHub-hosted CI does not run proprietary Isaac/GPU software or live hardware.
Detailed local reports and example screenshots are included with the repository.

The viewer is kinematic pose playback with paused dynamics; replay timing is not
the controller's timing. `recorded_route.json` separately retains actual sample
times and command settings. Contact forces, foam friction, slip, bending,
grasp-length metrology, cable envelopes and absolute physical calibration are not
qualified by these tests. Historical inferred pin axes are development evidence,
not verified pinhead/tip/grasp landmarks or labels for the six new drawers.

The relocated DINO backbone/head loaded from bundled files under OS network
isolation and produced finite normalized five-class probabilities on one saved
image. The bundled YOLO detector produced11 specimen framing boxes for drawer001;
that is an example output, not a verified specimen inventory or recall estimate.
Cached geometry extraction was rerun for117 historical images and produced all
seven provisional fits. Original data/caches remain unchanged.

## Experiment supplement, v0.2.0

The additional archive contains2,609 byte-identical original records/source files,
including31 further EIH photographs, and indexes43 saved session records. It was
installed into a second relocated directory through `fetch_assets.py
--experiments-only`; every added file was hash-checked and the evidence replay
passed there. The combined payload manifest now lists4,109 files; the original
1,500-file base payload and its published archive identities are unchanged.

The replay verifies18 connected moves/11 photo stations/three banks and35.7534s
planned motion. It recalculates the historical false stall from114 saved samples:
91 endpoint samples met the original tolerance; three consecutive ones occurred
by2.74294s; final error was0.002863mm/0.000152°. This is pose-only interpretation,
not controller queue completion or newly qualified collision geometry.

Seven CPU tests pass, including deliberately broken route order, override-blind
timing and changed command provenance. The supplemental source is a reference
snapshot, not imported by the supported replay. No hardware, model inference,
new calibration or new Isaac qualification is part of this supplement.

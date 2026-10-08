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
- Final publication validation also checks the relocated/fresh checkout and
  private release file hashes. Machine paths in immutable original evidence do
  not become runtime dependencies.

Normal CPU CI tests cover frame conventions, attachment motion and invalid trees.
GitHub-hosted CI does not run proprietary Isaac/GPU software or live hardware.
Detailed local reports and example screenshots are included with the repository.

The viewer is kinematic pose playback with paused dynamics; replay timing is not
the controller's timing. `recorded_route.json` separately retains actual sample
times and command settings. Contact forces, foam friction, slip, bending,
grasp-length metrology, cable envelopes and absolute physical calibration are not
qualified by these tests. Historical inferred pin axes are development evidence,
not verified pinhead/tip/grasp landmarks or labels for the six new drawers.

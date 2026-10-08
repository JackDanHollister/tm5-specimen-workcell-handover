# Robot configurations and frames

The supplied existing twin is derived from **TM5S-900-labelled** Techman assets.
Keep this source identity visible. Physical robot identity/revision and its factory
kinematics must be reconciled with the model before hardware-critical use.

All variants retain the built-in EIH camera, robot-side Quick Changer and circular
adapter/cap stack. The adapter shifts the QC/fitted gripper chain by **23.15 mm**.
Scannerless variants remove the Photoneo and projecting side-support model. The
complete 2FG7 body/fingers/tool frames are removed in gripperless variants.

These changes are made in USD geometry/physics relationships and URDF/XRDF
kinematics, spheres, tool frames and exclusions, rather than only visibility.
`configs/robots.json` records retained/removed parts and packaged frame paths.

Units: URDF/USD transforms in metres and radians; six robot joints `joint_1` through
`joint_6`. USD has Z-up and one metre per unit. The base coincides with robot base.
`eih_camera_optical` is a fixed flange-relative optical frame in every variant.
The `photoneo_camera` frame retains the native PrimaryCamera optical/CAD origin;
its housing centre is a different point.

The full gripper's inherited `pin_grasp_tcp` is a nominal development frame, not a
measured pinch/contact calibration. Whole-tool totals are nominal **2.55732 kg**
with scanner/gripper and **1.41732 kg** with scanner/QC only. Scannerless masses
are left unknown because removal of camera/support needs a defined weight split.
Centre of mass/inertia and physical contact dynamics are not validated.

The camera mesh is vendor geometry. Camera collision envelope uses its native CAD
bounds with development padding. Adapter radius and projecting bracket bounds are
assumptions; flexible cables and fastener engagement are not modelled reliably.
Source robot/gripper sphere enclosure is inherited and not certified mechanical
metrology. Self-collision exclusions are inherited/pruned by retained links.

Drawer surfaces retain measured per-view triangles, including overlap and missing
regions. Optional triangle-mesh CollisionAPI is authored but disabled by default
for fast viewing. Enabling it supplies observed surfaces only, not unseen walls,
thin shiny pins, foam deformation or complete free-space evidence.

The viewer pauses dynamics and writes articulation joint positions for evidence
playback. Camera/support/QC must follow the flange, while registered scans remain
fixed. The tests compare those transforms against URDF FK at nonzero poses.

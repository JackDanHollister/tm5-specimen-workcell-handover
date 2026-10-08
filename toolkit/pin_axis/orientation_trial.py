"""Pure geometry for a clearance-only pin-axis orientation preview; no robot IO.

The twin's convention is gripper local +Z pointing down the shaft. A line has
no measured roll or grasp point, so retain the closest wrist rotation and the
current pinch-centre XY. This is an orientation trial, not a grasp target.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def orientation_trial(axis_base, flange_base_m, tcp_offset_flange_m, *, lift_m=.030):
    axis = np.asarray(axis_base, dtype=float)
    flange = np.asarray(flange_base_m, dtype=float)
    offset = np.asarray(tcp_offset_flange_m, dtype=float)
    if (axis.shape != (3,) or flange.shape != (4, 4) or offset.shape != (3,)
            or not all(np.isfinite(x).all() for x in (axis, flange, offset))
            or not np.isfinite(lift_m) or not 0 <= lift_m <= .1):
        raise ValueError("Invalid finite geometry or lift outside 0..100 mm")
    if np.linalg.norm(axis) < 1e-9:
        raise ValueError("Zero pin axis")
    r0 = flange[:3, :3]
    if (not np.allclose(r0.T @ r0, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(r0), 1, atol=1e-7)
            or not np.allclose(flange[3], [0, 0, 0, 1])):
        raise ValueError("Expected a rigid flange transform")
    axis = axis / np.linalg.norm(axis)
    if axis[2] < 0:
        axis = -axis
    if axis[2] < .25:
        raise ValueError("Near-horizontal shaft is outside this downward trial")
    desired_z = -axis
    cross = np.cross(r0[:, 2], desired_z)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(r0[:, 2] @ desired_z, -1, 1))
    if cosine < -1 + 1e-8:
        raise ValueError("Antiparallel approach has ambiguous rotation")
    angle = float(np.arctan2(sine, cosine))
    correction = (Rotation.from_rotvec(cross / sine * angle).as_matrix()
                  if sine > 1e-10 else np.eye(3))
    lifted = flange.copy()
    lifted[2, 3] += lift_m
    pinch = lifted[:3, 3] + r0 @ offset
    aligned = lifted.copy()
    aligned[:3, :3] = correction @ r0
    aligned[:3, 3] = pinch - aligned[:3, :3] @ offset
    return {
        "motion_authority": False,
        "scope": "orientation_only_at_clearance_not_collinear_grasp",
        "frame": "robot_base_metres",
        "axis_base_up": axis.tolist(),
        "tool_z_target_base": desired_z.tolist(),
        "rotation_change_deg": float(np.rad2deg(angle)),
        "pinch_center_target_base_m": pinch.tolist(),
        "tcp_offset_flange_m": offset.tolist(),
        "lift_m": lift_m,
        "lifted_flange_base_m": lifted.tolist(),
        "aligned_flange_base_m": aligned.tolist(),
        "roll_policy": "minimum rotation from current wrist; shaft roll unmeasured",
    }

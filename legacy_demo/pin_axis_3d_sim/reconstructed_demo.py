"""Offline geometry for sequential transfers of saved multiview shaft lines.

Upper support endpoints are provisional display landmarks, not verified heads.
This module has no robot, ROS, camera, or network interfaces.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .drawer_planning_scene import CollisionPrimitive, transform_collision_primitive
from .geometry import rotation_matrix_from_z_axis


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(rotation=None, position=None):
    value = np.eye(4)
    if rotation is not None:
        value[:3, :3] = rotation
    if position is not None:
        value[:3, 3] = position
    return value


def make_scene(reconstruction: dict, config: dict) -> dict:
    cfg = dict(config)
    positive = (
        "grasp_below_upper_endpoint_mm", "approach_distance_m", "extraction_distance_m",
        "transport_clearance_m", "pin_radius_m", "head_radius_m", "required_clearance_m",
        "destination_slot_spacing_m", "axis_tolerance_m", "axis_angle_tolerance_deg",
    )
    for key in positive:
        if not math.isfinite(float(cfg[key])) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    offset = cfg["grasp_below_upper_endpoint_mm"] / 1000.0
    cases, ids = [], set()
    for group in reconstruction["specimens"]:
        item = group["primary"]
        if item["status"] != "provisional_consensus":
            raise ValueError(f"specimen {group['specimen_id']} has no accepted line")
        sid = str(group["specimen_id"])
        if sid in ids:
            raise ValueError("duplicate specimen ID")
        ids.add(sid)
        axis = np.asarray(item["axis_base"], dtype=float)
        support = np.asarray(item["observed_support_extent_base_m"], dtype=float)
        if axis.shape != (3,) or support.shape != (2, 3) or not np.isfinite(support).all():
            raise ValueError("invalid reconstructed line")
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("invalid shaft direction")
        axis /= np.linalg.norm(axis)
        if axis[2] < 0:
            axis = -axis
        if axis[2] < 0.25:
            raise ValueError("shaft too close to horizontal for this foam demo")
        point = np.asarray(item["point_on_axis_base_m"], dtype=float)
        along = (support - point) @ axis
        if np.max(np.linalg.norm(support - point - along[:, None] * axis, axis=1)) > 1e-5:
            raise ValueError("support endpoints do not lie on the reconstructed line")
        upper = point + float(max(along)) * axis
        length = (upper[2] - cfg["source_foam_z_m"]) / axis[2]
        if length <= offset + cfg["head_radius_m"] or length + cfg["buried_pin_length_m"] >= cfg["extraction_distance_m"]:
            raise ValueError("grasp/extraction dimensions do not fit this pin")
        entry = upper - length * axis
        grasp = upper - offset * axis
        slot = np.array(cfg["destination_first_slot_m"], dtype=float)
        index = len(cases)
        slot[0] += (index % cfg["destination_columns"]) * cfg["destination_slot_spacing_m"]
        slot[1] += (index // cfg["destination_columns"]) * cfg["destination_slot_spacing_m"]
        slot += cfg["destination_drawer_center_m"]
        cases.append({
            "id": sid, "axis_up": axis.tolist(), "upper_endpoint_m": upper.tolist(),
            "source_entry_m": entry.tolist(), "grasp_m": grasp.tolist(),
            "destination_entry_m": slot.tolist(), "pin_length_above_foam_m": float(length),
            "observed_support_span_m": float(np.ptp(along)),
            "tilt_deg": float(np.degrees(np.arccos(axis[2]))),
            "supporting_views": len(item.get("inlier_ids", [])),
            "median_reprojection_error_px": item.get("median_inlier_residual_px"),
            "upper_endpoint_status": "observed_support_upper_end_not_verified_head",
            "physical_pick_allowed": False,
        })
    if not cases:
        raise ValueError("no reconstructed pins")
    return {"format_version": 1, "mode": "offline_reconstructed_pin_demo", "config": cfg,
            "cases": cases, "physical_pick_allowed": False,
            "assumptions": [
                "Saved factory camera calibration is not physically verified.",
                "Pin upper endpoints use observed support limits; they are not verified heads.",
                "Foam height, buried length, diameter and specimen bodies are simulation assumptions.",
                "Separate acquisition placements appear sequentially; axes and lateral locations are unchanged.",
                "Kinematic attach/release does not simulate grasp forces or foam contact."]}


def payload_primitives(case: dict, config: dict) -> tuple[CollisionPrimitive, ...]:
    """Geometry in the pin frame: entry at origin, shaft along +Z."""
    length = case["pin_length_above_foam_m"]
    buried = config["buried_pin_length_m"]
    identity = np.eye(3)
    return (
        CollisionPrimitive("cuboid", "shaft", np.array([0, 0, (length - buried) / 2]),
                           identity, np.array([2 * config["pin_radius_m"]] * 2 + [length + buried])),
        CollisionPrimitive("sphere", "head", np.array([0, 0, length]), identity,
                           radius_m=config["head_radius_m"]),
        CollisionPrimitive("cuboid", "body", np.array([0, 0, length * config["illustrative_body_fraction"]]),
                           identity, np.array(config["illustrative_body_size_m"])),
    )


def source_payload_pose(case: dict) -> np.ndarray:
    return transform(rotation_matrix_from_z_axis(np.array(case["axis_up"])), case["source_entry_m"])


def transformed_payload(case: dict, config: dict, pose: np.ndarray):
    return tuple(transform_collision_primitive(p, pose) for p in payload_primitives(case, config))


def validate_plan(plan: dict) -> None:
    if plan.get("mode") != "offline_reconstructed_pin_transfer" or plan.get("status") != "passed":
        raise ValueError("not a completed reconstructed-pin plan")
    if plan.get("physical_pick_allowed") is not False:
        raise ValueError("offline plan cannot grant physical authority")
    cases = plan["scene"]["cases"]
    if len(plan["sequence"]) != len(cases):
        raise ValueError("incomplete sequence")
    ready = np.asarray(plan["scene"]["config"]["ready_joints_rad"])
    for case, sequence in zip(cases, plan["sequence"]):
        if case["id"] != sequence["id"] or sequence["audit"]["passed"] is not True:
            raise ValueError("unverified case")
        stages = sequence["stages"]
        names = [s["name"] for s in stages]
        expected = ["approach", "grasp", "extract", "clearance", "upright", "transport", "preplace", "place", "retreat", "home"]
        if names != expected:
            raise ValueError("stage sequence invalid")
        previous = ready
        for stage in stages:
            q = np.asarray(stage["samples"], dtype=float)
            if q.ndim != 2 or q.shape[1] != 6 or not np.isfinite(q).all():
                raise ValueError("invalid joint samples")
            if np.max(abs(q[0] - previous)) > 1e-6 or np.max(abs(np.diff(q, axis=0))) > 0.008001:
                raise ValueError("discontinuous or unbounded joint path")
            if np.any(q[:, 0] < -math.pi / 3) or np.any(q[:, 0] > 5 * math.pi / 6):
                raise ValueError("base joint outside permitted sector")
            previous = q[-1]
        if np.max(abs(previous - ready)) > 1e-5:
            raise ValueError("pin sequence does not return home")


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def presentation_frames(plan: dict, fps: int = 30, film_speed: float = 1.0) -> list[dict]:
    """Deterministic presentation clock; preserve every grasp/release boundary."""
    validate_plan(plan)
    if not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")
    if not math.isfinite(film_speed) or film_speed <= 0:
        raise ValueError("film speed must be finite and positive")
    durations = {"approach": 1.8, "grasp": 1.5, "extract": 1.8, "clearance": .8,
                 "upright": 1.0, "transport": 2.0, "preplace": .7, "place": 1.3,
                 "retreat": .8, "home": 1.7}
    frames = []
    for i, item in enumerate(plan["sequence"]):
        ready = plan["scene"]["config"]["ready_joints_rad"]
        for _ in range(fps):
            frames.append(dict(case=i, phase="spawn", joints=ready, grip=0., attached=False, placed=False))
        attached, placed = False, False
        for stage in item["stages"]:
            name = stage["name"]
            q = np.asarray(stage["samples"])
            count = max(2, round(durations[name] * fps / film_speed))
            for t in np.linspace(0, len(q) - 1, count):
                low = int(t)
                value = q[low] + (q[min(low + 1, len(q) - 1)] - q[low]) * (t - low)
                frames.append(dict(case=i, phase=name, joints=value.tolist(), grip=1. if attached else 0., attached=attached, placed=placed))
            if name == "grasp":
                for grip in np.linspace(0, 1, max(2, fps // 2)):
                    frames.append(dict(case=i, phase="close", joints=q[-1].tolist(), grip=float(grip), attached=False, placed=False))
                attached = True
            if name == "place":
                for grip in np.linspace(1, 0, max(2, fps // 2)):
                    frames.append(dict(case=i, phase="release", joints=q[-1].tolist(), grip=float(grip), attached=True, placed=False))
                attached, placed = False, True
        for _ in range(fps // 2):
            frames.append(dict(case=i, phase="home_pause", joints=ready, grip=0., attached=False, placed=True))
    return frames

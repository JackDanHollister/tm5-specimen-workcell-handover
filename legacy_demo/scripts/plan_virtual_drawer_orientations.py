#!/usr/bin/env python3
"""Assess virtual drawer grasp orientations with standalone cuMotion 1.1."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cumotion
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pin_axis_3d_sim.drawer_planning_scene import (
    CollisionPrimitive,
    DrawerCollisionConfig,
    build_drawer_collision_primitives,
    candidate_to_specimen_map,
    collision_scene_metadata,
)
from pin_axis_3d_sim.all_specimen_transfer import canonicalize_periodic_joint_path
from pin_axis_3d_sim.pin_planning_bridge import (
    OrientationPlanAssessment,
    select_best_orientation,
    validate_rigid_transform,
)
from plan_synthetic_pick import (
    EXPECTED_CUMOTION_VERSION,
    finite_vector,
    float64_sha256,
    quaternion_xyzw,
    sha256_file,
)


DEFAULT_RESULT = ROOT / "outputs/pin_perception/virtual_realistic_drawer/result.json"
DEFAULT_TARGETS = (
    ROOT / "outputs/pin_perception/virtual_realistic_drawer/planning_targets.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/pin_perception/virtual_realistic_drawer/cumotion_orientation_assessment.json"
)
DEFAULT_MODEL_DIR = ROOT / "generated/tool_profiles/watson_qc_nominal/cumotion"
DEFAULT_URDF = DEFAULT_MODEL_DIR / "tm5s_with_2fg7.urdf"
DEFAULT_XRDF = DEFAULT_MODEL_DIR / "tm5s_with_2fg7.xrdf"
DEFAULT_ASSET_MANIFEST = DEFAULT_MODEL_DIR / "asset_manifest.json"
DEFAULT_PLANNER_CONFIG = ROOT / "config/virtual_drawer_cumotion_planner.yaml"
DEFAULT_TASK_CONFIG = ROOT / "config/virtual_drawer_cumotion_planning.yaml"
STAGE_POSITION_KEYS = {
    "pregrasp": "pregrasp_position_xyz_m",
    "grasp": "grasp_position_xyz_m",
    "lift": "lift_position_xyz_m",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--planning-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--xrdf", type=Path, default=DEFAULT_XRDF)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--maximum-pins",
        type=int,
        default=None,
        help="diagnostic prefix limit; omit for the complete drawer",
    )
    parser.add_argument(
        "--removed-specimen-id",
        type=int,
        action="append",
        default=None,
        help="specimen already transferred and omitted from collision worlds; repeatable",
    )
    return parser


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_transform(targets: dict[str, Any]) -> np.ndarray:
    record = targets.get("source_to_target_transform", {})
    matrix = np.asarray(record.get("matrix_row_major"), dtype=float)
    if matrix.shape != (16,):
        raise ValueError("planning target transform must contain 16 values")
    if record.get("registration_status") != (
        "provisional_virtual_scene_placement_not_physical_registration"
    ):
        raise ValueError("drawer transform must remain explicitly provisional")
    if record.get("physical_motion_allowed") is not False:
        raise ValueError("drawer transform must not allow physical motion")
    return validate_rigid_transform(matrix.reshape(4, 4))


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]:
    input_paths = (
        args.result_json,
        args.planning_targets,
        args.urdf,
        args.xrdf,
        args.asset_manifest,
        args.planner_config,
        args.task_config,
    )
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing virtual drawer planning inputs:\n  " + "\n  ".join(missing))
    if getattr(cumotion, "__version__", None) != EXPECTED_CUMOTION_VERSION:
        raise RuntimeError(
            f"expected cuMotion {EXPECTED_CUMOTION_VERSION}; found "
            f"{getattr(cumotion, '__version__', 'unknown')}"
        )
    if args.maximum_pins is not None and args.maximum_pins < 1:
        raise ValueError("maximum-pins must be positive")

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    targets = json.loads(args.planning_targets.read_text(encoding="utf-8"))
    task = read_yaml(args.task_config)
    if result.get("mode") != "offline_photo_informed_virtual_drawer_pin_consensus":
        raise ValueError("unexpected realistic drawer source mode")
    if targets.get("mode") != "offline_virtual_consensus_to_unscored_planning_candidates":
        raise ValueError("unexpected planning-target mode")
    if targets.get("source", {}).get("sha256") != sha256_file(args.result_json):
        raise ValueError("planning targets do not match the realistic drawer result hash")
    for source, fields in (
        (
            result.get("scope", {}),
            (
                "isaac_launched",
                "ros_used",
                "watson_connected",
                "real_robot_commanded",
                "physical_pick_allowed",
            ),
        ),
        (
            targets.get("scope", {}),
            (
                "isaac_launched",
                "cumotion_run",
                "ros_used",
                "watson_connected",
                "real_robot_commanded",
                "physical_pick_allowed",
            ),
        ),
        (
            task.get("scope", {}),
            (
                "isaac_launched",
                "ros_used",
                "watson_connected",
                "real_robot_commanded",
                "physical_pick_allowed",
            ),
        ),
    ):
        for field in fields:
            if source.get(field) is not False:
                raise ValueError(f"input scope field {field} must be false")
    if task.get("format_version") != 1:
        raise ValueError("planning config must use format_version: 1")
    if targets.get("counts", {}).get("planner_assessment_count") != 0:
        raise ValueError("input orientations must still be unassessed")
    if targets.get("counts", {}).get("selected_orientation_count") != 0:
        raise ValueError("input orientations must not contain a selection")
    for target in targets.get("targets", []):
        if target.get("simulation_planning_allowed") is not True:
            raise ValueError("every requested target must pass simulation gates")
        if target.get("physical_pick_allowed") is not False:
            raise ValueError("target attempted to grant physical authority")
        for orientation in target.get("orientations", []):
            if orientation.get("physical_pick_allowed") is not False:
                raise ValueError("orientation attempted to grant physical authority")
            if orientation.get("simulation_planning_status") != (
                "unscored_requires_isaac_cumotion"
            ):
                raise ValueError("orientation is not in the expected unscored state")
    return result, targets, task, source_transform(targets)


def collision_config(task: dict[str, Any]) -> DrawerCollisionConfig:
    values = task.get("collision_scene")
    if not isinstance(values, dict):
        raise ValueError("collision_scene configuration is missing")
    return DrawerCollisionConfig(**values)


def add_world_primitive(world: Any, primitive: CollisionPrimitive) -> None:
    if primitive.primitive_type == "cuboid":
        obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, primitive.side_lengths)
        pose = cumotion.Pose3(
            cumotion.Rotation3.from_matrix(primitive.rotation_matrix),
            primitive.position,
        )
    elif primitive.primitive_type == "sphere":
        obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
        obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, primitive.radius_m)
        pose = cumotion.Pose3.from_translation(primitive.position)
    else:
        raise ValueError(f"unsupported collision primitive: {primitive.primitive_type}")
    world.add_obstacle(obstacle, pose)


def orientation_pose(orientation: dict[str, Any], stage: str) -> Any:
    position = finite_vector(
        orientation[STAGE_POSITION_KEYS[stage]], 3, f"{stage} position"
    )
    qx, qy, qz, qw = finite_vector(
        orientation["quaternion_xyzw"], 4, "orientation quaternion"
    )
    if not math.isclose(float(np.linalg.norm([qx, qy, qz, qw])), 1.0, abs_tol=2.0e-3):
        raise ValueError("orientation quaternion is not unit length")
    return cumotion.Pose3(cumotion.Rotation3(qw, qx, qy, qz), position)


def joint_limits(kinematics: Any) -> list[tuple[float, float]]:
    return [
        (
            float(kinematics.cspace_coord_limits(index).lower),
            float(kinematics.cspace_coord_limits(index).upper),
        )
        for index in range(kinematics.num_cspace_coords())
    ]


def assess_trajectory(
    trajectory: Any,
    *,
    inspector: Any,
    non_target_inspector: Any,
    selected_specimen_inspector: Any,
    limits: list[tuple[float, float]],
    validation_dt_seconds: float,
) -> dict[str, Any]:
    duration = float(trajectory.domain().span())
    count = max(1, int(math.ceil(duration / validation_dt_seconds)))
    times = np.linspace(0.0, duration, count + 1)
    upper = float(trajectory.domain().upper)
    positions: list[np.ndarray] = []
    minimum_clearance = math.inf
    minimum_non_target_clearance = math.inf
    minimum_selected_specimen_clearance = math.inf
    minimum_joint_margin = math.inf
    sampled_self_collision = False
    for sample_time in times:
        configuration = np.asarray(
            trajectory.eval(min(upper, float(sample_time)), 0), dtype=np.float64
        )
        positions.append(configuration)
        sampled_self_collision = sampled_self_collision or bool(
            inspector.in_self_collision(configuration)
        )
        minimum_clearance = min(
            minimum_clearance,
            float(inspector.min_distance_to_obstacle(configuration)),
        )
        minimum_non_target_clearance = min(
            minimum_non_target_clearance,
            float(non_target_inspector.min_distance_to_obstacle(configuration)),
        )
        minimum_selected_specimen_clearance = min(
            minimum_selected_specimen_clearance,
            float(
                selected_specimen_inspector.min_distance_to_obstacle(configuration)
            ),
        )
        minimum_joint_margin = min(
            minimum_joint_margin,
            *(
                min(value - lower, upper_limit - value)
                for value, (lower, upper_limit) in zip(configuration, limits)
            ),
        )
    path_length = sum(
        float(np.linalg.norm(second - first))
        for first, second in zip(positions, positions[1:])
    )
    stacked_positions = np.asarray(positions, dtype=np.float64)
    return {
        "duration_seconds": duration,
        "sample_count": len(positions),
        "minimum_sampled_sphere_clearance_m": minimum_clearance,
        "minimum_non_target_clearance_m": minimum_non_target_clearance,
        "minimum_selected_specimen_clearance_m": (
            minimum_selected_specimen_clearance
        ),
        "sampled_world_collision": minimum_clearance < 0.0,
        "sampled_self_collision": sampled_self_collision,
        "minimum_joint_limit_margin_rad": minimum_joint_margin,
        "joint_space_path_length_rad": path_length,
        "minimum_sampled_joint_positions_rad": np.min(
            stacked_positions, axis=0
        ).tolist(),
        "maximum_sampled_joint_positions_rad": np.max(
            stacked_positions, axis=0
        ).tolist(),
        "sampled_positions_float64_sha256": float64_sha256(positions),
    }


def plan_stage(
    *,
    stage_name: str,
    planner: Any,
    trajectory_generator: Any,
    inspector: Any,
    non_target_inspector: Any,
    selected_specimen_inspector: Any,
    kinematics: Any,
    limits: list[tuple[float, float]],
    tool_frame: str,
    current: np.ndarray,
    target_pose: Any,
    planning: dict[str, Any],
    include_trajectory_data: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    planner.reset()
    started = time.perf_counter_ns()
    planning_result = planner.plan_to_pose_target(current, target_pose, True)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if not planning_result.path_found:
        return (
            {
                "name": stage_name,
                "path_found": False,
                "accepted": False,
                "planning_latency_ms": latency_ms,
                "rejection_reasons": ["planner_found_no_path"],
            },
            current,
        )
    raw_path = [np.asarray(point, dtype=np.float64) for point in planning_result.path]
    if len(raw_path) < 2:
        return (
            {
                "name": stage_name,
                "path_found": True,
                "accepted": False,
                "planning_latency_ms": latency_ms,
                "raw_knots": len(raw_path),
                "rejection_reasons": ["planner_returned_short_path"],
            },
            current,
        )
    raw_path[0] = current.copy()
    canonicalized_base_knot_count = 0
    base_minimum = planning.get("required_base_joint_minimum_rad")
    base_maximum = planning.get("required_base_joint_maximum_rad")
    if (base_minimum is None) != (base_maximum is None):
        raise ValueError("base-joint clock-sector bounds must be supplied together")
    if base_minimum is not None:
        prior_base = [float(point[0]) for point in raw_path]
        try:
            raw_path = list(
                canonicalize_periodic_joint_path(
                    raw_path,
                    joint_index=0,
                    minimum_rad=float(base_minimum),
                    maximum_rad=float(base_maximum),
                )
            )
        except ValueError as exc:
            return (
                {
                    "name": stage_name,
                    "path_found": True,
                    "accepted": False,
                    "planning_latency_ms": latency_ms,
                    "raw_knots": len(raw_path),
                    "rejection_reasons": [
                        "base_joint_clock_sector_no_allowed_equivalent"
                    ],
                    "base_joint_clock_sector_error": str(exc),
                },
                current,
            )
        canonicalized_base_knot_count = sum(
            not math.isclose(before, float(after[0]), abs_tol=1.0e-12)
            for before, after in zip(prior_base, raw_path)
        )
    final = raw_path[-1].copy()
    trajectory = trajectory_generator.generate_trajectory(raw_path)
    if trajectory is None:
        return (
            {
                "name": stage_name,
                "path_found": True,
                "accepted": False,
                "planning_latency_ms": latency_ms,
                "raw_knots": len(raw_path),
                "rejection_reasons": ["trajectory_generation_failed"],
            },
            current,
        )
    trajectory_result = assess_trajectory(
        trajectory,
        inspector=inspector,
        non_target_inspector=non_target_inspector,
        selected_specimen_inspector=selected_specimen_inspector,
        limits=limits,
        validation_dt_seconds=float(planning["collision_validation_dt_seconds"]),
    )
    actual_pose = kinematics.pose(final, tool_frame)
    translation_error = float(
        np.linalg.norm(actual_pose.translation - target_pose.translation)
    )
    orientation_error = float(
        cumotion.Rotation3.distance(actual_pose.rotation, target_pose.rotation)
    )
    goal_tolerance_met = bool(
        translation_error <= float(planning["goal_translation_tolerance_m"])
        and orientation_error <= float(planning["goal_orientation_tolerance_rad"])
    )
    required_clearance = float(planning["required_sampled_sphere_clearance_m"])
    selected_specimen_clearance = float(
        planning["required_selected_specimen_clearance_m"]
    )
    reasons: list[str] = []
    if not goal_tolerance_met:
        reasons.append("goal_tolerance_not_met")
    if trajectory_result["sampled_self_collision"]:
        reasons.append("sampled_self_collision")
    if trajectory_result["minimum_non_target_clearance_m"] < required_clearance:
        reasons.append("non_target_clearance_below_required")
    if (
        trajectory_result["minimum_selected_specimen_clearance_m"]
        < selected_specimen_clearance
    ):
        reasons.append("selected_specimen_collision")
    if trajectory_result["minimum_joint_limit_margin_rad"] < 0.0:
        reasons.append("joint_limit_exceeded")
    if base_minimum is not None:
        sampled_minimum = float(
            trajectory_result["minimum_sampled_joint_positions_rad"][0]
        )
        sampled_maximum = float(
            trajectory_result["maximum_sampled_joint_positions_rad"][0]
        )
        if (
            sampled_minimum < float(base_minimum) - 1.0e-9
            or sampled_maximum > float(base_maximum) + 1.0e-9
        ):
            reasons.append("base_joint_clock_sector_exceeded")
    stage_result = {
        "name": stage_name,
        "path_found": True,
        "accepted": not reasons,
        "planning_latency_ms": latency_ms,
        "raw_knots": len(raw_path),
        "raw_path_float64_sha256": float64_sha256(raw_path),
        "goal_translation_error_m": translation_error,
        "goal_orientation_error_rad": orientation_error,
        "goal_tolerance_met": goal_tolerance_met,
        "trajectory_assessment": trajectory_result,
        "canonicalized_base_knot_count": canonicalized_base_knot_count,
        "rejection_reasons": reasons,
    }
    if include_trajectory_data:
        stage_result.update(
            {
                "start_joint_positions_rad": current.tolist(),
                "end_joint_positions_rad": final.tolist(),
                "raw_path": [point.tolist() for point in raw_path],
            }
        )
    return stage_result, final


def assess_orientation(
    *,
    candidate_id: int,
    orientation: dict[str, Any],
    planner: Any,
    trajectory_generator: Any,
    inspector: Any,
    non_target_inspector: Any,
    selected_specimen_inspector: Any,
    kinematics: Any,
    limits: list[tuple[float, float]],
    ready: np.ndarray,
    tool_frame: str,
    planning: dict[str, Any],
    include_trajectory_data: bool = False,
) -> tuple[dict[str, Any], OrientationPlanAssessment]:
    current = ready.copy()
    stages: list[dict[str, Any]] = []
    for stage_name in planning["stages"]:
        stage, next_configuration = plan_stage(
            stage_name=stage_name,
            planner=planner,
            trajectory_generator=trajectory_generator,
            inspector=inspector,
            non_target_inspector=non_target_inspector,
            selected_specimen_inspector=selected_specimen_inspector,
            kinematics=kinematics,
            limits=limits,
            tool_frame=tool_frame,
            current=current,
            target_pose=orientation_pose(orientation, stage_name),
            planning=planning,
            include_trajectory_data=include_trajectory_data,
        )
        stages.append(stage)
        if not stage["accepted"]:
            break
        current = next_configuration

    trajectory_results = [
        stage["trajectory_assessment"]
        for stage in stages
        if "trajectory_assessment" in stage
    ]
    complete_path = bool(
        len(stages) == len(planning["stages"])
        and all(stage.get("path_found") for stage in stages)
    )
    minimum_clearance = (
        min(item["minimum_non_target_clearance_m"] for item in trajectory_results)
        if trajectory_results
        else None
    )
    minimum_joint_margin = (
        min(item["minimum_joint_limit_margin_rad"] for item in trajectory_results)
        if trajectory_results
        else None
    )
    path_length = (
        sum(item["joint_space_path_length_rad"] for item in trajectory_results)
        if trajectory_results
        else None
    )
    self_collision = any(
        item["sampled_self_collision"] for item in trajectory_results
    )
    goal_tolerance_met = bool(
        len(stages) == len(planning["stages"])
        and all(stage.get("goal_tolerance_met", False) for stage in stages)
    )
    assessment = OrientationPlanAssessment(
        candidate_id=candidate_id,
        orientation_id=int(orientation["orientation_id"]),
        path_found=complete_path,
        sampled_self_collision=self_collision,
        minimum_clearance_m=minimum_clearance,
        joint_limit_margin_rad=minimum_joint_margin,
        path_length_rad=path_length,
        goal_tolerance_met=goal_tolerance_met,
    )
    stage_reasons = list(
        dict.fromkeys(
            reason
            for stage in stages
            for reason in stage.get("rejection_reasons", [])
        )
    )
    if not stage_reasons and len(stages) != len(planning["stages"]):
        stage_reasons.append("incomplete_stage_sequence")
    assessment_record = assessment.to_dict(
        minimum_clearance_m=float(planning["required_sampled_sphere_clearance_m"])
    )
    assessment_record["accepted"] = not stage_reasons
    assessment_record["rejection_reasons"] = stage_reasons
    return (
        {
            "candidate_id": candidate_id,
            "orientation_id": int(orientation["orientation_id"]),
            "roll_about_pin_axis_deg": float(orientation["roll_about_pin_axis_deg"]),
            "accepted": not stage_reasons,
            "rejection_reasons": stage_reasons,
            "assessment": assessment_record,
            "stages": stages,
            "physical_pick_allowed": False,
        },
        assessment,
    )


def assess_target(
    *,
    target: dict[str, Any],
    specimen_id: int,
    result: dict[str, Any],
    transform: np.ndarray,
    collision_cfg: DrawerCollisionConfig,
    robot_description: Any,
    kinematics: Any,
    trajectory_generator: Any,
    planner_config_path: Path,
    ready: np.ndarray,
    tool_frame: str,
    planning: dict[str, Any],
    include_trajectory_data: bool = False,
    removed_specimen_ids: set[int] | None = None,
) -> dict[str, Any]:
    candidate_id = int(target["candidate_id"])
    primitives = build_drawer_collision_primitives(
        result,
        source_to_target=transform,
        selected_specimen_id=specimen_id,
        config=collision_cfg,
        removed_specimen_ids=removed_specimen_ids,
    )
    world = cumotion.create_world()
    for primitive in primitives:
        add_world_primitive(world, primitive)
    world_view = world.add_world_view()
    non_target_world = cumotion.create_world()
    selected_specimen_world = cumotion.create_world()
    for primitive in primitives:
        if primitive.source_specimen_id == specimen_id:
            add_world_primitive(selected_specimen_world, primitive)
        else:
            add_world_primitive(non_target_world, primitive)
    non_target_world_view = non_target_world.add_world_view()
    selected_specimen_world_view = selected_specimen_world.add_world_view()
    planner_config = cumotion.create_motion_planner_config_from_file(
        str(planner_config_path), robot_description, tool_frame, world_view
    )
    planner_config.set_param("enable_self_collision_checking", True)
    planner = cumotion.create_motion_planner(planner_config)
    inspector = cumotion.create_robot_world_inspector(robot_description, world_view)
    non_target_inspector = cumotion.create_robot_world_inspector(
        robot_description, non_target_world_view
    )
    selected_specimen_inspector = cumotion.create_robot_world_inspector(
        robot_description, selected_specimen_world_view
    )
    ready_self_collision = bool(inspector.in_self_collision(ready))
    ready_world_collision = bool(inspector.in_collision_with_obstacle(ready))
    ready_clearance = float(inspector.min_distance_to_obstacle(ready))
    if ready_self_collision or ready_world_collision:
        return {
            "candidate_id": candidate_id,
            "source_specimen_id": specimen_id,
            "orientation_count": len(target["orientations"]),
            "assessed_orientation_count": 0,
            "accepted_orientation_count": 0,
            "selected_orientation_id": None,
            "ready_configuration": {
                "sampled_self_collision": ready_self_collision,
                "world_collision": ready_world_collision,
                "minimum_clearance_m": ready_clearance,
            },
            "collision_scene": collision_scene_metadata(primitives, collision_cfg),
            "orientations": [],
            "rejection_reasons": ["ready_configuration_invalid_for_target_world"],
            "physical_pick_allowed": False,
        }

    limits = joint_limits(kinematics)
    orientation_results: list[dict[str, Any]] = []
    assessments: list[OrientationPlanAssessment] = []
    planner_exceptions: list[str] = []
    for orientation in target["orientations"]:
        try:
            orientation_result, assessment = assess_orientation(
                candidate_id=candidate_id,
                orientation=orientation,
                planner=planner,
                trajectory_generator=trajectory_generator,
                inspector=inspector,
                non_target_inspector=non_target_inspector,
                selected_specimen_inspector=selected_specimen_inspector,
                kinematics=kinematics,
                limits=limits,
                ready=ready,
                tool_frame=tool_frame,
                planning=planning,
                include_trajectory_data=include_trajectory_data,
            )
            orientation_results.append(orientation_result)
            if orientation_result["accepted"]:
                assessments.append(assessment)
        except Exception as exc:  # cuMotion errors must be recorded, never promoted.
            planner_exceptions.append(type(exc).__name__)
            orientation_results.append(
                {
                    "candidate_id": candidate_id,
                    "orientation_id": int(orientation["orientation_id"]),
                    "roll_about_pin_axis_deg": float(
                        orientation["roll_about_pin_axis_deg"]
                    ),
                    "accepted": False,
                    "rejection_reasons": ["planner_exception"],
                    "planner_exception_type": type(exc).__name__,
                    "physical_pick_allowed": False,
                }
            )

    required_clearance = float(planning["required_sampled_sphere_clearance_m"])
    best = select_best_orientation(
        assessments, minimum_clearance_m=required_clearance
    )
    accepted_count = sum(item["accepted"] for item in orientation_results)
    selected_orientation = None
    if best is not None:
        pose = next(
            item
            for item in target["orientations"]
            if int(item["orientation_id"]) == best.orientation_id
        )
        selected_orientation = {
            "orientation_id": best.orientation_id,
            "roll_about_pin_axis_deg": float(pose["roll_about_pin_axis_deg"]),
            "pose_candidate": pose,
            "assessment": best.to_dict(minimum_clearance_m=required_clearance),
            "physical_pick_allowed": False,
        }
    return {
        "candidate_id": candidate_id,
        "source_specimen_id": specimen_id,
        "orientation_count": len(target["orientations"]),
        "assessed_orientation_count": len(orientation_results),
        "accepted_orientation_count": accepted_count,
        "selected_orientation_id": None if best is None else best.orientation_id,
        "selected_orientation": selected_orientation,
        "ready_configuration": {
            "sampled_self_collision": ready_self_collision,
            "world_collision": ready_world_collision,
            "minimum_clearance_m": ready_clearance,
        },
        "collision_scene": collision_scene_metadata(primitives, collision_cfg),
        "orientations": orientation_results,
        "planner_exception_types": sorted(set(planner_exceptions)),
        "rejection_reasons": (
            ["no_orientation_passed_all_planning_gates"] if best is None else []
        ),
        "physical_pick_allowed": False,
    }


def report_text(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    lines = [
        "# Virtual Drawer cuMotion Orientation Assessment",
        "",
        "Standalone cuMotion assessed the virtual pin-grasp candidates against",
        "conservative drawer, specimen, surface-clutter, and non-target-pin primitives.",
        "",
        f"- Pins assessed: `{counts['pin_count_assessed']}`.",
        f"- Orientations assessed: `{counts['orientation_count_assessed']}`.",
        f"- Accepted orientations: `{counts['accepted_orientation_count']}`.",
        f"- Pins with a selected orientation: `{counts['pin_count_with_selection']}`.",
        f"- Pins without a selection: `{counts['pin_count_without_selection']}`.",
        f"- Planner exceptions: `{counts['planner_exception_count']}`.",
        "- Isaac launched: `false`.",
        "- Physical pick allowed: `false`.",
        "",
        "| Candidate | Specimen | Accepted rolls | Selected roll | Non-target clearance |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in payload["targets"]:
        selected = target.get("selected_orientation")
        roll = "-" if selected is None else f"{selected['roll_about_pin_axis_deg']:.1f} deg"
        clearance = (
            "-"
            if selected is None
            else f"{1000.0 * selected['assessment']['minimum_clearance_m']:.2f} mm"
        )
        lines.append(
            f"| {target['candidate_id']} | {target['source_specimen_id']} | "
            f"{target['accepted_orientation_count']}/{target['orientation_count']} | "
            f"{roll} | {clearance} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The drawer transform is a provisional virtual placement. The collision",
            "world uses conservative primitives, not the rendered point mesh. The QC/2FG7",
            "profile and pin-grasp TCP are not physically commissioned. Selection is",
            "simulation evidence only and cannot authorize a robot command.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    for field in (
        "result_json",
        "planning_targets",
        "urdf",
        "xrdf",
        "asset_manifest",
        "planner_config",
        "task_config",
        "output",
    ):
        setattr(args, field, resolved(getattr(args, field)))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite orientation assessment: {args.output}")
    result, target_payload, task, transform = validate_inputs(args)
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    robot_description = cumotion.load_robot_from_file(str(args.xrdf), str(args.urdf))
    if robot_description.num_cspace_coords() != 6:
        raise RuntimeError("virtual drawer planner requires a six-axis robot")
    kinematics = robot_description.kinematics()
    expected_joint_names = list(task["robot"]["joint_names"])
    actual_joint_names = [
        kinematics.cspace_coord_name(index)
        for index in range(kinematics.num_cspace_coords())
    ]
    if actual_joint_names != expected_joint_names:
        raise RuntimeError(f"unexpected joint order: {actual_joint_names}")
    tool_frame = str(task["robot"]["planning_tool_frame"])
    if tool_frame not in robot_description.tool_frame_names():
        raise RuntimeError(f"planning tool frame is unavailable: {tool_frame}")
    ready = finite_vector(
        task["robot"]["ready_joint_positions"], 6, "ready joint positions"
    )
    planning = task["planning"]
    if planning.get("stages") != ["pregrasp", "grasp", "lift"]:
        raise ValueError("virtual drawer planning stages must be pregrasp, grasp, lift")
    for field in (
        "required_sampled_sphere_clearance_m",
        "goal_translation_tolerance_m",
        "goal_orientation_tolerance_rad",
        "collision_validation_dt_seconds",
    ):
        value = float(planning[field])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"planning field {field} must be finite and positive")
    selected_clearance = float(planning["required_selected_specimen_clearance_m"])
    if not math.isfinite(selected_clearance) or selected_clearance < 0.0:
        raise ValueError(
            "required_selected_specimen_clearance_m must be finite and non-negative"
        )

    mapping = candidate_to_specimen_map(result)
    specimen_ids = {
        int(item["specimen_id"]) for item in result["drawer"]["specimens"]
    }
    removed_specimen_ids = set(args.removed_specimen_id or [])
    unknown_removed = sorted(removed_specimen_ids - specimen_ids)
    if unknown_removed:
        raise ValueError(f"removed specimen IDs are missing: {unknown_removed}")
    targets = list(target_payload["targets"])
    if args.maximum_pins is not None:
        targets = targets[: args.maximum_pins]
    if not targets:
        raise ValueError("no virtual drawer targets were selected")
    missing_matches = [
        int(target["candidate_id"])
        for target in targets
        if int(target["candidate_id"]) not in mapping
    ]
    if missing_matches:
        raise ValueError(f"targets lack virtual specimen matches: {missing_matches}")
    already_removed_targets = [
        int(target["candidate_id"])
        for target in targets
        if mapping[int(target["candidate_id"])] in removed_specimen_ids
    ]
    if already_removed_targets:
        raise ValueError(
            f"requested candidates are already removed: {already_removed_targets}"
        )

    trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)
    collision_cfg = collision_config(task)
    results: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        candidate_id = int(target["candidate_id"])
        target_result = assess_target(
            target=target,
            specimen_id=mapping[candidate_id],
            result=result,
            transform=transform,
            collision_cfg=collision_cfg,
            robot_description=robot_description,
            kinematics=kinematics,
            trajectory_generator=trajectory_generator,
            planner_config_path=args.planner_config,
            ready=ready,
            tool_frame=tool_frame,
            planning=planning,
            removed_specimen_ids=removed_specimen_ids,
        )
        results.append(target_result)
        print(
            f"[{index:02d}/{len(targets):02d}] candidate {candidate_id}: "
            f"{target_result['accepted_orientation_count']}/"
            f"{target_result['orientation_count']} orientations accepted"
        )

    orientation_count = sum(item["orientation_count"] for item in results)
    assessed_count = sum(item["assessed_orientation_count"] for item in results)
    accepted_count = sum(item["accepted_orientation_count"] for item in results)
    selected_count = sum(item["selected_orientation_id"] is not None for item in results)
    exception_count = sum(
        sum(
            orientation.get("rejection_reasons") == ["planner_exception"]
            for orientation in item["orientations"]
        )
        for item in results
    )
    input_paths = {
        "realistic_drawer_result": args.result_json,
        "planning_targets": args.planning_targets,
        "urdf": args.urdf,
        "xrdf": args.xrdf,
        "asset_manifest": args.asset_manifest,
        "planner_config": args.planner_config,
        "task_config": args.task_config,
    }
    payload = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_virtual_drawer_cumotion_orientation_assessment",
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cumotion_version": cumotion.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "scope": {
            "virtual_source_only": True,
            "cumotion_run": True,
            "isaac_launched": False,
            "ros_used": False,
            "watson_connected": False,
            "real_robot_commanded": False,
            "physical_pick_allowed": False,
        },
        "model": {
            "planning_tool_frame": tool_frame,
            "tool_profile": task["robot"]["tool_profile"],
            "status": (
                "simulation-only Watson QC nominal candidate; TCP and mount are not "
                "physically commissioned"
            ),
        },
        "source_to_target_transform": target_payload["source_to_target_transform"],
        "planning": planning,
        "removed_source_specimen_ids": sorted(removed_specimen_ids),
        "input_artifacts": {
            name: artifact_record(path) for name, path in input_paths.items()
        },
        "requested_full_drawer": args.maximum_pins is None,
        "counts": {
            "pin_count_available": len(target_payload["targets"]),
            "pin_count_assessed": len(results),
            "orientation_count_requested": orientation_count,
            "orientation_count_assessed": assessed_count,
            "accepted_orientation_count": accepted_count,
            "rejected_orientation_count": assessed_count - accepted_count,
            "pin_count_with_selection": selected_count,
            "pin_count_without_selection": len(results) - selected_count,
            "planner_exception_count": exception_count,
        },
        "all_requested_orientations_assessed": assessed_count == orientation_count,
        "all_assessed_pins_have_selection": selected_count == len(results),
        "selection_status": (
            "all_simulation_orientations_selected"
            if selected_count == len(results)
            else (
                "partial_simulation_orientation_selection"
                if selected_count
                else "no_orientation_passed_all_planning_gates"
            )
        ),
        "rejection_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for target in results
                    for orientation in target["orientations"]
                    for reason in orientation.get("rejection_reasons", [])
                ).items()
            )
        ),
        "targets": results,
        "physical_pick_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = args.output.with_name("cumotion_orientation_report.md")
    report_path.write_text(report_text(payload), encoding="utf-8")
    print(f"Output: {args.output}")
    print(f"Report: {report_path}")
    print(f"Selected pins: {selected_count}/{len(results)}")
    print("Isaac launched: false")
    print("Physical pick allowed: false")
    return 0 if selected_count and exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

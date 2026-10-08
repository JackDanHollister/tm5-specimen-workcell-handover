#!/usr/bin/env python3
"""Replan and validate an indexed two-drawer transfer for every detected specimen."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cumotion
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pin_axis_3d_sim.all_specimen_transfer import (
    DestinationGridConfig,
    destination_slots,
    indexed_drawer_transform,
    rigid_payload_transform,
    transfer_geometry_record,
)
from pin_axis_3d_sim.drawer_planning_scene import (
    build_drawer_collision_primitives,
    build_empty_drawer_collision_primitives,
    candidate_to_specimen_map,
)
from pin_axis_3d_sim.geometry import normalize, rotation_matrix_from_z_axis
from plan_synthetic_pick import (
    EXPECTED_CUMOTION_VERSION,
    finite_vector,
    float64_sha256,
    sha256_file,
)
from plan_virtual_drawer_orientations import (
    add_world_primitive,
    assess_target,
    assess_trajectory,
    collision_config,
    joint_limits,
    plan_stage,
    source_transform,
)


DEFAULT_CONFIG = ROOT / "config/virtual_drawer_all_specimen_sequence.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/pin_perception/virtual_two_drawer_all_specimens"
    / "cumotion_all_specimen_transfer_sequence.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def resolved_config_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checked_file(config: dict[str, Any], key: str) -> Path:
    path = resolved_config_path(str(config[key]))
    if not path.is_file():
        raise FileNotFoundError(f"missing all-specimen input {key}: {path}")
    return path


def make_planner_and_inspector(
    *,
    primitives: tuple[Any, ...],
    robot_description: Any,
    planner_config_path: Path,
    tool_frame: str,
) -> tuple[Any, Any]:
    world = cumotion.create_world()
    for primitive in primitives:
        add_world_primitive(world, primitive)
    view = world.add_world_view()
    planner_config = cumotion.create_motion_planner_config_from_file(
        str(planner_config_path), robot_description, tool_frame, view
    )
    planner_config.set_param("enable_self_collision_checking", True)
    return (
        cumotion.create_motion_planner(planner_config),
        cumotion.create_robot_world_inspector(robot_description, view),
    )


def target_pose(position: np.ndarray, rotation: np.ndarray) -> Any:
    return cumotion.Pose3(
        cumotion.Rotation3.from_matrix(rotation),
        np.asarray(position, dtype=float),
    )


def plan_return_ready(
    *,
    planner: Any,
    trajectory_generator: Any,
    inspector: Any,
    limits: list[tuple[float, float]],
    current: np.ndarray,
    ready: np.ndarray,
    planning: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    planner.reset()
    started = time.perf_counter_ns()
    result = planner.plan_to_cspace_target(current, ready, True)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if not result.path_found:
        return (
            {
                "name": "return_ready",
                "path_found": False,
                "accepted": False,
                "planning_latency_ms": latency_ms,
                "rejection_reasons": ["planner_found_no_path"],
            },
            current,
        )
    raw_path = [np.asarray(point, dtype=np.float64) for point in result.path]
    if len(raw_path) < 2:
        return (
            {
                "name": "return_ready",
                "path_found": True,
                "accepted": False,
                "planning_latency_ms": latency_ms,
                "rejection_reasons": ["planner_returned_short_path"],
            },
            current,
        )
    raw_path[0] = current.copy()
    final = raw_path[-1].copy()
    trajectory = trajectory_generator.generate_trajectory(raw_path)
    if trajectory is None:
        return (
            {
                "name": "return_ready",
                "path_found": True,
                "accepted": False,
                "planning_latency_ms": latency_ms,
                "rejection_reasons": ["trajectory_generation_failed"],
            },
            current,
        )
    assessed = assess_trajectory(
        trajectory,
        inspector=inspector,
        non_target_inspector=inspector,
        selected_specimen_inspector=inspector,
        limits=limits,
        validation_dt_seconds=float(planning["collision_validation_dt_seconds"]),
    )
    reasons: list[str] = []
    if float(np.max(np.abs(final - ready))) > 1.0e-4:
        reasons.append("ready_goal_tolerance_not_met")
    if assessed["sampled_self_collision"]:
        reasons.append("sampled_self_collision")
    if assessed["minimum_non_target_clearance_m"] < float(
        planning["required_sampled_sphere_clearance_m"]
    ):
        reasons.append("non_target_clearance_below_required")
    if assessed["minimum_joint_limit_margin_rad"] < 0.0:
        reasons.append("joint_limit_exceeded")
    return (
        {
            "name": "return_ready",
            "path_found": True,
            "accepted": not reasons,
            "planning_latency_ms": latency_ms,
            "raw_knots": len(raw_path),
            "raw_path_float64_sha256": float64_sha256(raw_path),
            "start_joint_positions_rad": current.tolist(),
            "end_joint_positions_rad": final.tolist(),
            "raw_path": [point.tolist() for point in raw_path],
            "trajectory_assessment": assessed,
            "rejection_reasons": reasons,
        },
        final,
    )


def assessment_targets(
    assessment_path: Path,
    *,
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, set[int]]:
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if assessment.get("mode") != "offline_virtual_drawer_cumotion_orientation_assessment":
        raise ValueError(f"unexpected assessment mode: {assessment_path}")
    if assessment.get("all_requested_orientations_assessed") is not True:
        raise ValueError(f"incomplete assessment: {assessment_path}")
    if assessment.get("counts", {}).get("planner_exception_count") != 0:
        raise ValueError(f"planner exceptions in assessment: {assessment_path}")
    if assessment.get("physical_pick_allowed") is not False:
        raise ValueError("assessment attempted to grant physical authority")
    records = assessment.get("input_artifacts", {})
    if records.get("realistic_drawer_result", {}).get("sha256") != sha256_file(
        result_path
    ):
        raise ValueError(f"assessment detector hash mismatch: {assessment_path}")
    target_path = Path(records["planning_targets"]["path"]).resolve()
    if not target_path.is_file() or sha256_file(target_path) != records[
        "planning_targets"
    ]["sha256"]:
        raise ValueError(f"assessment planning-target artifact mismatch: {assessment_path}")
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    selected = [
        item for item in assessment.get("targets", [])
        if item.get("selected_orientation_id") is not None
    ]
    if not selected:
        raise ValueError(f"assessment has no selected targets: {assessment_path}")
    return (
        assessment,
        target_payload,
        source_transform(target_payload),
        set(int(item) for item in assessment.get("removed_source_specimen_ids", [])),
    )


def main() -> int:
    args = build_parser().parse_args()
    args.config = args.config.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite all-specimen plan: {args.output}")
    if getattr(cumotion, "__version__", None) != EXPECTED_CUMOTION_VERSION:
        raise RuntimeError(
            f"expected cuMotion {EXPECTED_CUMOTION_VERSION}; found "
            f"{getattr(cumotion, '__version__', 'unknown')}"
        )
    config = read_yaml(args.config)
    if config.get("format_version") != 1:
        raise ValueError("unexpected all-specimen config version")
    if config.get("scope", {}).get("physical_pick_allowed") is not False:
        raise ValueError("all-specimen config attempted to grant physical authority")
    result_path = checked_file(config, "result_json")
    planner_config_path = checked_file(config, "planner_config")
    task_config_path = checked_file(config, "task_config")
    model = config.get("precision_model", {})
    urdf_path = checked_file(model, "urdf")
    xrdf_path = checked_file(model, "xrdf")
    manifest_path = checked_file(model, "asset_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for link in ("onrobot_2fg7_left_finger_link", "onrobot_2fg7_right_finger_link"):
        audit = manifest.get("sphere_coverage_audit", {}).get(link, {})
        if int(audit.get("sphere_count", 0)) < 128:
            raise ValueError("precision model does not retain 128-sphere fingertips")
        if float(audit.get("audited_max_uncovered_gap_m", math.inf)) > -0.001 + 1e-9:
            raise ValueError("precision fingertip model failed its 1 mm coverage audit")

    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    task = read_yaml(task_config_path)
    mapping = candidate_to_specimen_map(result_payload)
    if len(mapping) != int(result_payload["drawer"]["specimen_count"]):
        raise ValueError("detector does not geometrically associate every specimen")

    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    robot_description = cumotion.load_robot_from_file(str(xrdf_path), str(urdf_path))
    kinematics = robot_description.kinematics()
    if [kinematics.cspace_coord_name(index) for index in range(6)] != list(
        task["robot"]["joint_names"]
    ):
        raise RuntimeError("unexpected robot joint order")
    tool_frame = str(task["robot"]["planning_tool_frame"])
    ready = finite_vector(task["robot"]["ready_joint_positions"], 6, "ready")
    planning = task["planning"]
    cfg = collision_config(task)
    trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)
    limits = joint_limits(kinematics)
    empty_view = cumotion.create_world().add_world_view()
    empty_inspector = cumotion.create_robot_world_inspector(
        robot_description, empty_view
    )

    destination = config["destination"]
    drop_base = finite_vector(
        destination["fixed_drop_base_world_m"], 3, "destination drop base"
    )
    grid = DestinationGridConfig(**destination["grid"])
    slots = destination_slots(grid)
    assessment_paths = [resolved_config_path(item) for item in config["assessments"]]
    if any(not path.is_file() for path in assessment_paths):
        missing = [str(path) for path in assessment_paths if not path.is_file()]
        raise FileNotFoundError("missing sequence assessments: " + ", ".join(missing))

    selected_records: list[tuple[Path, dict[str, Any], dict[str, Any], np.ndarray, set[int]]] = []
    seen_candidates: set[int] = set()
    seen_specimens: set[int] = set()
    for assessment_path in assessment_paths:
        assessment, target_payload, transform, removed = assessment_targets(
            assessment_path,
            result_path=result_path,
        )
        for selected in assessment["targets"]:
            if selected.get("selected_orientation_id") is None:
                continue
            candidate_id = int(selected["candidate_id"])
            specimen_id = int(selected["source_specimen_id"])
            if candidate_id in seen_candidates or specimen_id in seen_specimens:
                raise ValueError("all-specimen assessment selection is not unique")
            if mapping.get(candidate_id) != specimen_id:
                raise ValueError("selected target disagrees with geometry association")
            seen_candidates.add(candidate_id)
            seen_specimens.add(specimen_id)
            selected_records.append(
                (assessment_path, selected, target_payload, transform, removed)
            )
    specimen_count = int(result_payload["drawer"]["specimen_count"])
    if len(selected_records) != specimen_count or len(slots) < specimen_count:
        raise ValueError(
            f"sequence does not cover every specimen: {len(selected_records)}/{specimen_count}"
        )

    destination_rotation = rotation_matrix_from_z_axis(
        np.array([0.0, 0.0, -1.0]),
        x_hint=np.array([0.0, 1.0, 0.0]),
    )
    sequence: list[dict[str, Any]] = []
    all_complete = True
    for sequence_index, record in enumerate(selected_records):
        assessment_path, selected, target_payload, transform, removed = record
        candidate_id = int(selected["candidate_id"])
        specimen_id = int(selected["source_specimen_id"])
        orientation_id = int(selected["selected_orientation_id"])
        source_target = next(
            item for item in target_payload["targets"]
            if int(item["candidate_id"]) == candidate_id
        )
        orientation = next(
            item for item in source_target["orientations"]
            if int(item["orientation_id"]) == orientation_id
        )
        one_orientation_target = {**source_target, "orientations": [orientation]}
        replanned = assess_target(
            target=one_orientation_target,
            specimen_id=specimen_id,
            result=result_payload,
            transform=transform,
            collision_cfg=cfg,
            robot_description=robot_description,
            kinematics=kinematics,
            trajectory_generator=trajectory_generator,
            planner_config_path=planner_config_path,
            ready=ready,
            tool_frame=tool_frame,
            planning=planning,
            include_trajectory_data=True,
            removed_specimen_ids=removed,
        )
        source_orientation = replanned["orientations"][0]
        source_stages = source_orientation.get("stages", [])
        if (
            replanned.get("accepted_orientation_count") != 1
            or len(source_stages) != 3
            or not all(stage.get("accepted") for stage in source_stages)
        ):
            raise RuntimeError(f"precision source replan failed for candidate {candidate_id}")
        current = np.asarray(source_stages[-1]["raw_path"][-1], dtype=np.float64)
        source_rotation = np.asarray(
            orientation["rotation_matrix_row_major"], dtype=float
        ).reshape(3, 3)
        source_axis = normalize(np.asarray(source_target["pin_axis_up"], dtype=float))
        source_grasp = finite_vector(
            orientation["grasp_position_xyz_m"], 3, "source grasp"
        )
        source_entry = finite_vector(
            source_target["foam_entry_xyz_m"], 3, "source foam entry"
        )
        source_head = finite_vector(
            source_target["head_center_xyz_m"], 3, "source head"
        )
        detected_length = float(np.dot(source_head - source_entry, source_axis))
        grasp_below_head = float(orientation["grasp_below_head_m"])

        source_primitives = tuple(
            item
            for item in build_drawer_collision_primitives(
                result_payload,
                source_to_target=transform,
                selected_specimen_id=specimen_id,
                config=cfg,
                removed_specimen_ids=removed,
            )
            if item.source_specimen_id != specimen_id
        )
        source_planner, source_inspector = make_planner_and_inspector(
            primitives=source_primitives,
            robot_description=robot_description,
            planner_config_path=planner_config_path,
            tool_frame=tool_frame,
        )
        transfer_planning = dict(planning)
        transfer_planning["required_selected_specimen_clearance_m"] = 0.0
        source_axis_clearance_position = source_grasp + (
            float(destination["source_axis_clearance_m"]) * source_axis
        )
        source_axis_clearance, current = plan_stage(
            stage_name="source_axis_clearance",
            planner=source_planner,
            trajectory_generator=trajectory_generator,
            inspector=source_inspector,
            non_target_inspector=source_inspector,
            selected_specimen_inspector=empty_inspector,
            kinematics=kinematics,
            limits=limits,
            tool_frame=tool_frame,
            current=current,
            target_pose=target_pose(source_axis_clearance_position, source_rotation),
            planning=transfer_planning,
            include_trajectory_data=True,
        )
        source_clearance_position = source_grasp + np.array(
            [0.0, 0.0, float(destination["transport_clearance_m"])]
        )
        source_clearance: dict[str, Any]
        if source_axis_clearance.get("accepted"):
            source_clearance, current = plan_stage(
                stage_name="source_clearance",
                planner=source_planner,
                trajectory_generator=trajectory_generator,
                inspector=source_inspector,
                non_target_inspector=source_inspector,
                selected_specimen_inspector=empty_inspector,
                kinematics=kinematics,
                limits=limits,
                tool_frame=tool_frame,
                current=current,
                target_pose=target_pose(source_clearance_position, source_rotation),
                planning=transfer_planning,
                include_trajectory_data=True,
            )
        else:
            source_clearance = {
                "name": "source_clearance",
                "accepted": False,
                "path_found": False,
                "rejection_reasons": ["prior_source_axis_clearance_failed"],
            }

        slot = slots[sequence_index]
        destination_transform = indexed_drawer_transform(
            local_slot_xyz_m=slot,
            world_drop_xyz_m=drop_base,
            yaw_deg=float(destination["drawer_yaw_deg"]),
        )
        destination_primitives = build_empty_drawer_collision_primitives(
            drawer_width_m=float(result_payload["drawer"]["drawer_width_m"]),
            drawer_height_m=float(result_payload["drawer"]["drawer_height_m"]),
            source_to_target=destination_transform,
            config=cfg,
        )
        destination_planner, destination_inspector = make_planner_and_inspector(
            primitives=destination_primitives,
            robot_description=robot_description,
            planner_config_path=planner_config_path,
            tool_frame=tool_frame,
        )
        payload_transform = rigid_payload_transform(
            source_tool_rotation=source_rotation,
            destination_tool_rotation=destination_rotation,
            source_foam_entry_world_m=source_entry,
            destination_pin_base_world_m=drop_base,
        )
        destination_grasp = drop_base + np.array(
            [0.0, 0.0, detected_length - grasp_below_head]
        )
        stage_specs = [
            ("source_verticalize", source_clearance_position),
            (
                "transport",
                destination_grasp
                + np.array([0.0, 0.0, float(destination["transport_clearance_m"])]),
            ),
            (
                "destination_preplace",
                destination_grasp
                + np.array([0.0, 0.0, float(destination["preplace_clearance_m"])]),
            ),
            ("destination_place", destination_grasp),
            (
                "destination_retreat",
                destination_grasp
                + np.array([0.0, 0.0, float(destination["retreat_clearance_m"])]),
            ),
        ]
        transfer_stages = [source_axis_clearance, source_clearance]
        if source_axis_clearance.get("accepted") and source_clearance.get("accepted"):
            for stage_name, position in stage_specs:
                stage_planning = dict(transfer_planning)
                if stage_name in {"destination_place", "destination_retreat"}:
                    stage_planning["required_sampled_sphere_clearance_m"] = 0.0
                stage, next_configuration = plan_stage(
                    stage_name=stage_name,
                    planner=destination_planner,
                    trajectory_generator=trajectory_generator,
                    inspector=destination_inspector,
                    non_target_inspector=destination_inspector,
                    selected_specimen_inspector=empty_inspector,
                    kinematics=kinematics,
                    limits=limits,
                    tool_frame=tool_frame,
                    current=current,
                    target_pose=target_pose(position, destination_rotation),
                    planning=stage_planning,
                    include_trajectory_data=True,
                )
                transfer_stages.append(stage)
                if not stage.get("accepted"):
                    break
                current = next_configuration
        if len(transfer_stages) == 7 and all(
            stage.get("accepted") for stage in transfer_stages
        ):
            return_stage, current = plan_return_ready(
                planner=destination_planner,
                trajectory_generator=trajectory_generator,
                inspector=destination_inspector,
                limits=limits,
                current=current,
                ready=ready,
                planning=planning,
            )
            transfer_stages.append(return_stage)
        complete = bool(
            len(transfer_stages) == 8
            and all(stage.get("accepted") for stage in transfer_stages)
        )
        all_complete = all_complete and complete
        sequence.append(
            {
                "sequence_index": sequence_index,
                "candidate_id": candidate_id,
                "source_specimen_id": specimen_id,
                "source_assessment": artifact(assessment_path),
                "removed_source_specimen_ids_before_pick": sorted(removed),
                "source_drawer_transform_row_major": transform.reshape(-1).tolist(),
                "orientation_id": orientation_id,
                "roll_about_source_pin_axis_deg": float(
                    orientation["roll_about_pin_axis_deg"]
                ),
                "grasp_below_head_m": grasp_below_head,
                "source_pose_candidate": orientation,
                "source_stages": source_stages,
                "transfer_stages": transfer_stages,
                "transfer_geometry": transfer_geometry_record(
                    slot=slot,
                    destination_drawer_transform=destination_transform,
                    payload_transform=payload_transform,
                    source_axis_world=source_axis,
                ),
                "complete_ready_to_ready_transfer": complete,
                "physical_pick_allowed": False,
            }
        )
        status = "accepted" if complete else next(
            (
                ",".join(stage.get("rejection_reasons", []))
                for stage in transfer_stages
                if not stage.get("accepted")
            ),
            "incomplete",
        )
        print(
            f"[{sequence_index + 1:02d}/{specimen_count:02d}] candidate "
            f"{candidate_id}, specimen {specimen_id}: {status}"
        )
        if not complete:
            break

    completed_count = sum(item["complete_ready_to_ready_transfer"] for item in sequence)
    payload = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_virtual_all_specimen_indexed_cumotion_transfer_sequence",
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cumotion_version": cumotion.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "scope": {
            "detector_geometry_only": True,
            "indexed_source_and_destination_drawers": True,
            "precision_collision_sphere_model": True,
            "carried_payload_collision_geometry_in_cumotion": False,
            "placed_payload_collision_geometry_in_cumotion": False,
            "isaac_launched": False,
            "ros_used": False,
            "watson_connected": False,
            "real_robot_commanded": False,
            "physical_pick_allowed": False,
        },
        "inputs": {
            "config": artifact(args.config),
            "detector_result": artifact(result_path),
            "urdf": artifact(urdf_path),
            "xrdf": artifact(xrdf_path),
            "asset_manifest": artifact(manifest_path),
            "planner_config": artifact(planner_config_path),
            "task_config": artifact(task_config_path),
        },
        "destination": destination,
        "counts": {
            "detected_specimen_count": len(mapping),
            "requested_transfer_count": specimen_count,
            "planned_transfer_count": len(sequence),
            "complete_ready_to_ready_transfer_count": completed_count,
        },
        "all_specimens_have_complete_transfer": bool(
            all_complete and completed_count == specimen_count
        ),
        "static_collision_limit": (
            "robot checked against current source or indexed empty destination drawer; "
            "carried and previously placed specimen bodies remain visual-only"
        ),
        "sequence": sequence,
        "physical_pick_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = args.output.with_name("cumotion_all_specimen_transfer_report.md")
    report_path.write_text(
        "\n".join(
            [
                "# All-specimen Indexed cuMotion Transfer",
                "",
                f"- Detected specimens: `{len(mapping)}/{specimen_count}`.",
                f"- Complete ready-to-ready transfers: `{completed_count}/{specimen_count}`.",
                f"- All specimens complete: `{str(payload['all_specimens_have_complete_transfer']).lower()}`.",
                "- Source and destination drawers: `indexed between transfers`.",
                "- Precision fingertip collision spheres: `128 per finger, 1 mm sampled enclosure margin`.",
                "- Carried and previously placed payload collision: `not yet modeled by cuMotion`.",
                "- Isaac launched: `false`.",
                "- Physical pick allowed: `false`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Output: {args.output}")
    print(f"Complete transfers: {completed_count}/{specimen_count}")
    print("Physical pick allowed: false")
    return 0 if payload["all_specimens_have_complete_transfer"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

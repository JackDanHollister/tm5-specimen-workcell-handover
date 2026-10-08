#!/usr/bin/env python3
"""Plan and audit the saved multiview pins without any hardware interfaces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cumotion
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pin_axis_3d_sim.reconstructed_demo import (
    digest, make_scene, transform, source_payload_pose, transformed_payload, validate_plan,
)
from pin_axis_3d_sim.geometry import rotation_matrix_from_z_axis
from pin_axis_3d_sim.drawer_planning_scene import (
    DrawerCollisionConfig, build_empty_drawer_collision_primitives,
    collision_primitives_intersect, transform_collision_primitive,
)
from plan_virtual_all_specimen_transfers import make_planner_and_inspector, target_pose, plan_return_ready
from plan_virtual_drawer_orientations import add_world_primitive, joint_limits, plan_stage


def make_inspector(robot, primitives):
    world = cumotion.create_world()
    for primitive in primitives:
        add_world_primitive(world, primitive)
    return cumotion.create_robot_world_inspector(robot, world.add_world_view())


def sampled_stage(stage, generator):
    trajectory = generator.generate_trajectory([np.array(q) for q in stage["raw_path"]])
    if trajectory is None:
        raise ValueError("trajectory could not be reproduced")
    duration = float(trajectory.domain().span())
    samples = [np.array(stage["start_joint_positions_rad"])]
    for t in np.linspace(0, duration, max(2, math.ceil(duration / .01) + 1))[1:]:
        q = np.asarray(trajectory.eval(min(t, float(trajectory.domain().upper)), 0))
        start = samples[-1]
        count = max(1, math.ceil(float(np.max(abs(q - start))) / .006))
        samples.extend(start + (q - start) * i / count for i in range(1, count + 1))
    return np.array(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/reconstructed_pin_demo.json")
    parser.add_argument("--grasp-offset-mm", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "generated/tool_profiles/watson_qc_fixed_workcell_cumotion")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if args.grasp_offset_mm is not None:
        config["grasp_below_upper_endpoint_mm"] = args.grasp_offset_mm
    reconstruction = json.loads(args.reconstruction.read_text())
    if args.limit:
        reconstruction["specimens"] = reconstruction["specimens"][:args.limit]
    scene = make_scene(reconstruction, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = args.model_dir
    urdf, xrdf = model / "tm5s_with_2fg7.urdf", model / "tm5s_with_2fg7.xrdf"
    if cumotion.__version__ != "1.1.0":
        raise RuntimeError(f"Unexpected cuMotion: {cumotion.__version__}")
    cumotion.set_log_level(cumotion.LogLevel.ERROR)
    # Constrain planning itself to the same base sector used by the final audit;
    # rejecting an unconstrained long-way-around path after planning is insufficient.
    bounded_tree = ET.parse(urdf)
    limit = bounded_tree.find("./joint[@name='joint_1']/limit")
    limit.set("lower", str(-math.pi / 3))
    limit.set("upper", str(5 * math.pi / 6))
    for mesh in bounded_tree.findall(".//mesh"):
        mesh.set("filename", str((urdf.parent / mesh.get("filename")).resolve()))
    bounded_urdf = args.output.with_suffix(".bounded.urdf")
    bounded_tree.write(bounded_urdf, encoding="unicode")
    robot = cumotion.load_robot_from_file(str(xrdf), str(bounded_urdf))
    kin = robot.kinematics()
    generator = cumotion.create_cspace_trajectory_generator(kin)
    limits = joint_limits(kin)
    task = yaml.safe_load((ROOT / "config/virtual_drawer_cumotion_planning.yaml").read_text())
    settings = dict(task["planning"])
    settings.update(required_sampled_sphere_clearance_m=config["required_clearance_m"],
                    required_base_joint_minimum_rad=-math.pi / 3,
                    required_base_joint_maximum_rad=5 * math.pi / 6)
    ready = np.array(config["ready_joints_rad"])
    tool = "pin_grasp_tcp"
    width, height = config["drawer_size_m"]
    yaw = math.radians(config["source_drawer_yaw_deg"])
    source_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    source = transform(source_rotation, config["source_drawer_center_m"])
    destination = transform(position=config["destination_drawer_center_m"])
    shells = tuple(build_empty_drawer_collision_primitives(
        drawer_width_m=size[0], drawer_height_m=size[1], source_to_target=frame,
        config=DrawerCollisionConfig()) for frame, size in
        ((source, config["source_drawer_size_m"]), (destination, [width, height])))
    static = (*shells[0], *shells[1])
    placed = []
    empty = make_inspector(robot, ())
    plan = {"format_version": 1, "mode": "offline_reconstructed_pin_transfer", "status": "planning",
            "scene": scene, "sequence": [], "physical_pick_allowed": False,
            "provenance": {"reconstruction_sha256": digest(args.reconstruction),
                           "config_sha256": digest(args.config), "urdf_sha256": digest(urdf),
                           "xrdf_sha256": digest(xrdf), "cumotion_version": cumotion.__version__,
                           "base_joint_bounds_rad": [-math.pi / 3, 5 * math.pi / 6]},
            "scope": {"ros_used": False, "robot_connected": False, "physical_pick_allowed": False}}

    def checkpoint():
        args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")

    for case in scene["cases"]:
        print(f"Planning pin {case['id']} tilt={case['tilt_deg']:.2f} deg", flush=True)
        axis = np.array(case["axis_up"])
        grasp_position = np.array(case["grasp_m"])
        pin_frame = source_payload_pose(case)
        payload_world = transformed_payload(case, config, pin_frame)
        body = tuple(p for p in payload_world if p.role == "body")
        obstacle_world = (*static, *placed)
        planner_path = ROOT / "config/virtual_drawer_cumotion_planner.yaml"
        transit_planner, transit_inspector = make_planner_and_inspector(
            primitives=obstacle_world, robot_description=robot, planner_config_path=planner_path, tool_frame=tool)
        pick_planner, pick_inspector = make_planner_and_inspector(
            primitives=(*obstacle_world, *body), robot_description=robot, planner_config_path=planner_path, tool_frame=tool)
        attempts = []
        accepted = None
        for roll in range(0, 360, 30):
            r = math.radians(roll)
            roll_matrix = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
            rotation = rotation_matrix_from_z_axis(-axis) @ roll_matrix
            grasp = transform(rotation, grasp_position)
            payload_in_tool = np.linalg.inv(grasp) @ pin_frame
            # A minimal rotation aligns the shaft with destination +Z, preserving
            # the chosen jaw roll and rigid attachment of the payload.
            from pin_axis_3d_sim.all_specimen_transfer import verticalizing_payload_rotation
            alignment = verticalizing_payload_rotation(source_pin_axis_world=axis,
                source_body_axis_world=pin_frame[:3, 0], destination_body_axis_world=np.array([1., 0., 0.]))
            dest_rotation = alignment @ rotation
            dest_grasp_position = np.array(case["destination_entry_m"]) + np.array([0, 0, case["pin_length_above_foam_m"] - config["grasp_below_upper_endpoint_mm"] / 1000])
            destination_grasp = transform(dest_rotation, dest_grasp_position)
            targets = [
                ("approach", transform(rotation, grasp_position + axis * config["approach_distance_m"])),
                ("grasp", grasp),
                ("extract", transform(rotation, grasp_position + axis * config["extraction_distance_m"])),
                ("clearance", transform(rotation, grasp_position + [0, 0, config["transport_clearance_m"]])),
                ("upright", transform(dest_rotation, grasp_position + [0, 0, config["transport_clearance_m"]])),
                ("transport", transform(dest_rotation, dest_grasp_position + [0, 0, config["transport_clearance_m"]])),
                ("preplace", transform(dest_rotation, dest_grasp_position + [0, 0, .06])),
                ("place", destination_grasp),
                ("retreat", transform(dest_rotation, dest_grasp_position + [0, 0, .08])),
            ]
            current = ready.copy()
            stages = []
            rejected = None
            metrics = {"max_axis_lateral_error_m": 0., "max_axis_error_deg": 0.,
                       "max_destination_lateral_error_m": 0., "max_destination_axis_error_deg": 0.,
                       "minimum_arm_clearance_m": 1000., "samples_checked": 0,
                       "payload_pair_checks": 0, "passed": True}
            for name, target in targets:
                planner, inspector = (pick_planner, pick_inspector) if name in {"approach", "grasp"} else (transit_planner, transit_inspector)
                # Short Cartesian waypoints keep the insertion/extraction on-axis;
                # the complete resulting sampled path is independently checked.
                axis_move = name in {"grasp", "extract", "preplace", "place", "retreat"}
                start_pose = kin.pose(current, tool)
                start_position = np.asarray(start_pose.translation)
                spacing = .015 if axis_move else .06
                pieces = max(1, math.ceil(np.linalg.norm(target[:3, 3] - start_position) / spacing)) if axis_move or name == "transport" else 1
                if name == "transport":
                    pieces = max(pieces, 12)
                samples = [current.copy()]
                for piece in range(1, pieces + 1):
                    endpoint = target.copy()
                    fraction = piece / pieces
                    endpoint[:3, 3] = start_position + (target[:3, 3] - start_position) * fraction
                    if name == "transport":
                        start_angle = math.atan2(start_position[1], start_position[0])
                        end_angle = math.atan2(target[1, 3], target[0, 3])
                        angle = start_angle + (end_angle - start_angle) * fraction
                        radius = np.linalg.norm(start_position[:2]) * (1-fraction) + np.linalg.norm(target[:2,3]) * fraction
                        endpoint[:2,3] = radius * np.array([math.cos(angle), math.sin(angle)])
                    record, next_q = plan_stage(stage_name=name, planner=planner, trajectory_generator=generator,
                        inspector=inspector, non_target_inspector=inspector, selected_specimen_inspector=empty,
                        kinematics=kin, limits=limits, tool_frame=tool, current=current,
                        target_pose=target_pose(endpoint[:3, 3], endpoint[:3, :3]), planning=settings,
                        include_trajectory_data=True)
                    if not record["accepted"]:
                        rejected = {"stage": name, "reasons": record["rejection_reasons"],
                                    "target": endpoint[:3,3].tolist(), "start_joints": current.tolist()}
                        break
                    dense = sampled_stage(record, generator)
                    samples.extend(dense[1:])
                    current = next_q
                if rejected:
                    break
                q_values = np.array(samples)
                # Audit the exact densely sampled path that the demo consumes.
                previous_progress = None
                for q in q_values:
                    pose = kin.pose(q, tool)
                    tcp = transform(pose.rotation.matrix(), pose.translation)
                    metrics["samples_checked"] += 1
                    arm_clearance = float(inspector.min_distance_to_obstacle(q))
                    metrics["minimum_arm_clearance_m"] = min(metrics["minimum_arm_clearance_m"], arm_clearance)
                    if arm_clearance < config["required_clearance_m"] or inspector.in_self_collision(q):
                        rejected = {"stage": name, "reasons": ["dense_arm_collision"]}
                        break
                    if name in {"grasp", "extract", "preplace", "place", "retreat"}:
                        source_axis = name in {"grasp", "extract"}
                        line_axis = axis if source_axis else np.array([0., 0., 1.])
                        line_origin = grasp_position if source_axis else dest_grasp_position
                        delta = tcp[:3, 3] - line_origin
                        lateral = float(np.linalg.norm(delta - np.dot(delta, line_axis) * line_axis))
                        angle = float(np.degrees(np.arccos(np.clip(np.dot(-tcp[:3, 2], line_axis), -1, 1))))
                        lateral_key = "max_axis_lateral_error_m" if source_axis else "max_destination_lateral_error_m"
                        angle_key = "max_axis_error_deg" if source_axis else "max_destination_axis_error_deg"
                        metrics[lateral_key] = max(metrics[lateral_key], lateral)
                        metrics[angle_key] = max(metrics[angle_key], angle)
                        progress = float(np.dot(delta, line_axis)) * (-1 if name in {"grasp", "preplace", "place"} else 1)
                        if lateral > config["axis_tolerance_m"] or angle > config["axis_angle_tolerance_deg"] or (previous_progress is not None and progress < previous_progress - .0001):
                            rejected = {"stage": name, "reasons": ["axis_contract"], "lateral": lateral, "angle": angle}
                            break
                        previous_progress = progress
                    if name in {"extract", "clearance", "upright", "transport", "preplace", "place"}:
                        carried = transformed_payload(case, config, tcp @ payload_in_tool)
                        for p in carried:
                            for obstacle in obstacle_world:
                                # Only intentional shaft penetration into foam is excluded.
                                if "foam" in obstacle.role and p.role == "shaft":
                                    continue
                                # Cheap broad phase before exact cuboid/sphere tests.
                                a = p.radius_m if p.radius_m is not None else np.linalg.norm(p.side_lengths) / 2
                                b = obstacle.radius_m if obstacle.radius_m is not None else np.linalg.norm(obstacle.side_lengths) / 2
                                if np.linalg.norm(p.position - obstacle.position) > a + b + config["required_clearance_m"]:
                                    continue
                                metrics["payload_pair_checks"] += 1
                                if collision_primitives_intersect(p, obstacle, required_clearance_m=config["required_clearance_m"]):
                                    rejected = {"stage": name, "reasons": ["carried_payload_collision"], "pair": [p.role, obstacle.role]}
                                    break
                            if rejected:
                                break
                    if rejected:
                        break
                if rejected:
                    break
                stages.append({"name": name, "samples": q_values.tolist()})
            if not rejected:
                dest_payload_pose = destination_grasp @ payload_in_tool
                new_placed = transformed_payload(case, config, dest_payload_pose)
                home_planner, home_inspector = make_planner_and_inspector(
                    primitives=(*obstacle_world, *new_placed), robot_description=robot,
                    planner_config_path=planner_path, tool_frame=tool)
                home, current = plan_return_ready(planner=home_planner, trajectory_generator=generator,
                    inspector=home_inspector, limits=limits, current=current, ready=ready, planning=settings)
                if not home["accepted"]:
                    rejected = {"stage": "home", "reasons": home["rejection_reasons"]}
                else:
                    stages.append({"name": "home", "samples": sampled_stage(home, generator).tolist()})
            if rejected:
                attempts.append({"roll_deg": roll, **rejected})
                print(f"  roll {roll}: {rejected}", flush=True)
                continue
            accepted = {"id": case["id"], "roll_deg": roll, "stages": stages, "audit": metrics,
                        "payload_in_tool": payload_in_tool.tolist(), "source_payload_pose": pin_frame.tolist(),
                        "destination_payload_pose": dest_payload_pose.tolist(), "failed_attempts": attempts}
            placed.extend(new_placed)
            break
        if accepted is None:
            plan["status"] = "failed"
            plan["failure"] = {"id": case["id"], "attempts": attempts}
            checkpoint()
            raise RuntimeError(f"No qualified transfer for pin {case['id']}")
        plan["sequence"].append(accepted)
        checkpoint()
        print(f"Pin {case['id']} passed: {accepted['audit']}", flush=True)
    plan["status"] = "passed"
    validate_plan(plan)
    checkpoint()
    print(f"PASSED {len(plan['sequence'])} complete transfers: {args.output}", flush=True)


if __name__ == "__main__":
    main()

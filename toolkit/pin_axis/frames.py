"""Read-only frame math for saved TM coordinates and the Isaac reference URDF."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def transform(xyz, rpy) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()  # Rz(yaw) Ry(pitch) Rx(roll)
    result[:3, 3] = xyz
    return result


def saved_flange(coordinates: dict) -> np.ndarray:
    pose = coordinates["flange_in_robot_base"]
    return transform(pose["xyz_m"], pose["rpy_rad"])


def handeye_transform(camera: dict) -> np.ndarray:
    values = camera["handeye_camera_to_flange"]
    return transform(np.array(values["translation_mm"]) / 1000,
                     np.deg2rad(values["rpy_deg"]))


def intrinsics_for(camera: dict, focus: float, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    matches = [row for row in camera["factory_intrinsics"] if row["focus"] == focus and
               (row["width_px"], row["height_px"]) == tuple(size)]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one factory calibration for focus {focus}, size {size}")
    return np.array(matches[0]["camera_matrix"], float), np.array(matches[0]["distortion"], float)


def urdf_link_poses(path: Path, joints: dict[str, float]) -> dict[str, np.ndarray]:
    robot = ET.parse(path).getroot()
    remaining = list(robot.findall("joint"))
    children = {joint.find("child").get("link") for joint in remaining}
    roots = {link.get("name") for link in robot.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("Expected a single URDF root")
    poses = {roots.pop(): np.eye(4)}
    while remaining:
        progress = False
        for joint in remaining[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            motion = np.eye(4)
            if joint.get("type") != "fixed":
                axis_element = joint.find("axis")
                axis = np.fromstring(axis_element.get("xyz", "1 0 0"), sep=" ")
                value = joints.get(joint.get("name"), 0.0)
                if joint.get("type") in {"revolute", "continuous"}:
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif joint.get("type") == "prismatic":
                    motion[:3, 3] = axis * value
                else:
                    raise ValueError("Unsupported URDF joint type")
            poses[child] = poses[parent] @ transform(xyz, rpy) @ motion
            remaining.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Disconnected or cyclic URDF")
    return poses

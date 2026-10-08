"""Provisional 3-D infinite shaft line from calibrated saved image lines.

Each observed shaft back-projects to a plane through its camera centre. Plane
intersection determines the line; visible segment endpoints are NOT assumed to
be matching physical head/tip landmarks across views. No control interfaces.
"""
from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares


def observation(image_id: str, endpoints: np.ndarray, camera_pose: np.ndarray,
                K: np.ndarray, distortion: np.ndarray) -> dict:
    endpoints = np.asarray(endpoints, dtype=float)
    if endpoints.shape != (2, 2) or np.linalg.norm(endpoints[1] - endpoints[0]) < 4:
        raise ValueError("Need two distinct image points on a visible shaft")
    # Intermediate samples permit distortion-aware line fitting as well as endpoints.
    pixels = endpoints[0] + np.linspace(0, 1, 7)[:, None] * (endpoints[1] - endpoints[0])
    undistorted = cv2.undistortPoints(pixels[:, None, :], K, distortion, P=K)[:, 0, :]
    homogeneous = np.column_stack((undistorted, np.ones(len(undistorted))))
    _, _, vt = np.linalg.svd(homogeneous)
    line = vt[-1]
    normal_camera = K.T @ line
    normal = camera_pose[:3, :3] @ normal_camera
    normal /= np.linalg.norm(normal)
    return {"id": image_id, "pixels": pixels, "undistorted": undistorted,
            "K": K, "distortion": distortion, "pose": camera_pose,
            "normal": normal, "offset": float(normal @ camera_pose[:3, 3])}


def linear_solve(observations: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    if len(observations) < 2:
        raise ValueError("At least two views are required")
    normals = np.array([o["normal"] for o in observations])
    offsets = np.array([o["offset"] for o in observations])
    _, singular, vt = np.linalg.svd(normals, full_matrices=True)
    conditioning = float(singular[1] / singular[0])
    if conditioning < 0.03:
        raise ValueError("Back-projection planes are nearly parallel")
    direction = vt[-1]
    direction /= np.linalg.norm(direction)
    point = np.linalg.lstsq(np.vstack((normals, direction)), np.r_[offsets, 0.0], rcond=None)[0]
    return point, direction, conditioning


def projected_line(point: np.ndarray, direction: np.ndarray, obs: dict) -> np.ndarray:
    pose = obs["pose"]
    normal_world = np.cross(direction, point - pose[:3, 3])
    line = np.linalg.solve(obs["K"].T, pose[:3, :3].T @ normal_world)
    scale = np.linalg.norm(line[:2])
    return line / max(scale, 1e-15)


def signed_residual(point: np.ndarray, direction: np.ndarray, obs: dict) -> np.ndarray:
    line = projected_line(point, direction, obs)
    return np.column_stack((obs["undistorted"], np.ones(len(obs["undistorted"])))) @ line


def errors(point: np.ndarray, direction: np.ndarray, observations: list[dict]) -> np.ndarray:
    return np.array([np.sqrt(np.mean(signed_residual(point, direction, obs) ** 2)) for obs in observations])


def refine(observations: list[dict], point: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Four local parameters avoid the line's translation/scale gauge freedoms.
    basis_seed = np.eye(3)[np.argmin(abs(direction))]
    u = np.cross(direction, basis_seed); u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    def unpack(parameters):
        axis = direction + parameters[2] * u + parameters[3] * v
        axis /= np.linalg.norm(axis)
        anchor = point + parameters[0] * u + parameters[1] * v
        anchor -= axis * (anchor @ axis)
        return anchor, axis
    def residual(parameters):
        anchor, axis = unpack(parameters)
        return np.concatenate([signed_residual(anchor, axis, obs) for obs in observations])
    fit = least_squares(residual, np.zeros(4), loss="soft_l1", f_scale=3,
                        x_scale=[0.02, 0.02, 0.1, 0.1], max_nfev=150)
    return unpack(fit.x)


def ray_line_coordinates(point: np.ndarray, direction: np.ndarray, obs: dict) -> tuple[np.ndarray, np.ndarray]:
    rays = np.column_stack((obs["undistorted"], np.ones(len(obs["undistorted"])))) @ np.linalg.inv(obs["K"]).T
    rays = rays @ obs["pose"][:3, :3].T
    values, depths = [], []
    for ray in rays:
        along, depth = np.linalg.lstsq(np.column_stack((direction, -ray)),
                                      obs["pose"][:3, 3] - point, rcond=None)[0]
        values.append(along); depths.append(depth)
    return np.array(values), np.array(depths)


def solve(observations: list[dict], threshold_px: float = 8.0) -> dict:
    result = {"status": "insufficient_consistent_views", "view_count": len(observations),
              "threshold_px": threshold_px, "axis_base": None, "point_on_axis_base_m": None,
              "inlier_ids": [], "views": [], "calibration_physically_verified": False}
    if len(observations) < 3:
        result["reason"] = "Need at least three detected views for a consistency check"
        return result
    best = None
    for i, j in combinations(range(len(observations)), 2):
        pair = [observations[i], observations[j]]
        if np.linalg.norm(pair[0]["pose"][:3, 3] - pair[1]["pose"][:3, 3]) < 0.015:
            continue
        try:
            point, direction, conditioning = linear_solve(pair)
        except ValueError:
            continue
        residual = errors(point, direction, observations)
        mask = residual <= threshold_px
        count = int(mask.sum())
        score = (count, -float(np.median(residual[mask])) if count else -1e9)
        if best is None or score > best[0]:
            best = (score, mask, point, direction)
    if best is None or best[0][0] < 3:
        result["reason"] = "No three-view consensus with sufficient camera baseline"
        return result
    _, mask, point, direction = best
    for _ in range(3):
        selected = [o for o, flag in zip(observations, mask) if flag]
        point, direction, conditioning = linear_solve(selected)
        point, direction = refine(selected, point, direction)
        residual = errors(point, direction, observations)
        updated = residual <= threshold_px
        if updated.sum() < 3:
            result["reason"] = "Consensus was lost during refinement"
            return result
        if np.array_equal(updated, mask):
            break
        mask = updated
    selected = [o for o, flag in zip(observations, mask) if flag]
    if direction[2] < 0:
        direction = -direction
    along, depths = zip(*(ray_line_coordinates(point, direction, o) for o in selected))
    positive_depth = float(np.mean(np.concatenate(depths) > 0))
    positions = np.concatenate(along)
    center = point + np.median(positions) * direction
    support_extent = point + np.quantile(positions, [0.02, 0.98])[:, None] * direction
    loo_residuals, loo_angles = [], []
    for i, obs in enumerate(selected):
        subset = selected[:i] + selected[i + 1:]
        try:
            p, d, _ = linear_solve(subset)
            if len(subset) >= 3:
                p, d = refine(subset, p, d)
            loo_residuals.append(float(errors(p, d, [obs])[0]))
            loo_angles.append(float(np.degrees(np.arccos(np.clip(abs(d @ direction), 0, 1)))))
        except ValueError:
            loo_residuals.append(None); loo_angles.append(None)
    result.update({"status": "provisional_consensus" if positive_depth == 1 else "failed_cheirality",
                   "reason": "Factory calibration and fixed-placement assumption; not independently measured accuracy",
                   "axis_base": direction.tolist(), "point_on_axis_base_m": point.tolist(),
                   "display_center_base_m": center.tolist(), "observed_support_extent_base_m": support_extent.tolist(),
                   "inlier_ids": [o["id"] for o in selected], "conditioning": conditioning,
                   "positive_depth_fraction": positive_depth,
                   "tilt_from_base_z_deg": float(np.degrees(np.arccos(np.clip(direction[2], -1, 1)))),
                   "azimuth_from_base_x_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
                   "median_inlier_residual_px": float(np.median(residual[mask])),
                   "max_inlier_residual_px": float(np.max(residual[mask])),
                   "median_all_detected_residual_px": float(np.median(residual)),
                   "leave_one_view_out_residual_px": loo_residuals,
                   "leave_one_view_out_axis_change_deg": loo_angles,
                   "views": [{"id": o["id"], "inlier": bool(flag), "residual_px": float(error)}
                             for o, flag, error in zip(observations, mask, residual)]})
    return result

"""Independent pin-head and clear-grasp estimation around detected shaft axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .detection import PinAxisDetection
from .geometry import line_distances, serializable_vec


@dataclass(frozen=True)
class PinFeatureConfig:
    pin_head_radius_m: float = 0.0022
    head_center_to_grasp_m: float = 0.0072
    clear_shaft_length_m: float = 0.010
    head_search_radius_m: float = 0.0045
    head_top_quantile: float = 0.997
    minimum_head_radial_extent_m: float = 0.0011
    head_shell_tolerance_m: float = 0.0006
    minimum_head_shell_inliers: int = 10
    minimum_observed_shaft_length_m: float = 0.005
    grasp_corridor_radius_m: float = 0.004
    grasp_corridor_half_length_m: float = 0.002
    shaft_exclusion_radius_m: float = 0.001
    maximum_grasp_corridor_obstacle_points: int = 10


@dataclass(frozen=True)
class PinFeatureEstimate:
    detection_id: int
    estimated_head_center: np.ndarray
    estimated_head_surface_top: np.ndarray
    estimated_grasp_center: np.ndarray
    clear_shaft_near_head: np.ndarray
    clear_shaft_near_specimen: np.ndarray
    head_radial_extent_m: float
    head_shell_inliers: int
    head_shell_residual_m: float | None
    head_resolved: bool
    grasp_corridor_obstacle_points: int
    grasp_corridor_clear: bool
    geometry_candidate: bool
    physical_pick_allowed: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "detection_id": int(self.detection_id),
            "estimated_head_center": serializable_vec(self.estimated_head_center),
            "estimated_head_surface_top": serializable_vec(
                self.estimated_head_surface_top
            ),
            "estimated_grasp_center": serializable_vec(self.estimated_grasp_center),
            "clear_shaft_near_head": serializable_vec(self.clear_shaft_near_head),
            "clear_shaft_near_specimen": serializable_vec(
                self.clear_shaft_near_specimen
            ),
            "head_radial_extent_m": round(float(self.head_radial_extent_m), 6),
            "head_shell_inliers": int(self.head_shell_inliers),
            "head_shell_residual_m": (
                None
                if self.head_shell_residual_m is None
                else round(float(self.head_shell_residual_m), 6)
            ),
            "head_resolved": bool(self.head_resolved),
            "grasp_corridor_obstacle_points": int(
                self.grasp_corridor_obstacle_points
            ),
            "grasp_corridor_clear": bool(self.grasp_corridor_clear),
            "geometry_candidate": bool(self.geometry_candidate),
            "physical_pick_allowed": False,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _validate_config(config: PinFeatureConfig) -> None:
    values = asdict(config)
    for key, value in values.items():
        if key in {
            "minimum_head_shell_inliers",
            "maximum_grasp_corridor_obstacle_points",
        }:
            if int(value) < 0:
                raise ValueError(f"{key} must be non-negative")
        elif key == "head_top_quantile":
            if not 0.5 < float(value) <= 1.0:
                raise ValueError("head_top_quantile must be in (0.5, 1]")
        elif float(value) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if config.head_center_to_grasp_m <= config.pin_head_radius_m:
        raise ValueError("grasp center must lie below the configured pin head")
    if config.shaft_exclusion_radius_m >= config.grasp_corridor_radius_m:
        raise ValueError("shaft exclusion radius must be inside the grasp corridor")


def estimate_pin_features(
    points: np.ndarray,
    detection: PinAxisDetection,
    *,
    config: PinFeatureConfig | None = None,
) -> PinFeatureEstimate:
    """Estimate the head and clear grasp from XYZ geometry around one axis."""
    cfg = config or PinFeatureConfig()
    _validate_config(cfg)
    cloud = np.asarray(points, dtype=float)
    radial = line_distances(cloud, detection.point_on_axis, detection.axis_up)
    axial = (cloud - detection.point_on_axis) @ detection.axis_up
    near_axis = radial <= cfg.head_search_radius_m
    if not np.any(near_axis):
        raise ValueError("No point-cloud support exists near the detected axis")

    top_axial = float(np.quantile(axial[near_axis], cfg.head_top_quantile))
    surface_top = detection.point_on_axis + top_axial * detection.axis_up
    head_center = surface_top - cfg.pin_head_radius_m * detection.axis_up
    center_axial = top_axial - cfg.pin_head_radius_m

    in_head_span = near_axis & (
        np.abs(axial - center_axial) <= 1.25 * cfg.pin_head_radius_m
    )
    head_radial = radial[in_head_span]
    radial_extent = (
        0.0 if head_radial.size == 0 else float(np.quantile(head_radial, 0.90))
    )
    shell_residual = np.abs(
        np.linalg.norm(cloud[in_head_span] - head_center, axis=1)
        - cfg.pin_head_radius_m
    )
    shell_mask = shell_residual <= cfg.head_shell_tolerance_m
    shell_inliers = int(np.sum(shell_mask))
    shell_fit = (
        None
        if shell_inliers == 0
        else float(np.sqrt(np.mean(shell_residual[shell_mask] ** 2)))
    )
    head_resolved = (
        radial_extent >= cfg.minimum_head_radial_extent_m
        and shell_inliers >= cfg.minimum_head_shell_inliers
    )

    grasp = head_center - cfg.head_center_to_grasp_m * detection.axis_up
    near_head = head_center - cfg.pin_head_radius_m * detection.axis_up
    near_specimen = near_head - cfg.clear_shaft_length_m * detection.axis_up
    grasp_axial = (grasp - detection.point_on_axis) @ detection.axis_up
    in_grasp_slice = (
        np.abs(axial - grasp_axial) <= cfg.grasp_corridor_half_length_m
    )
    corridor_obstacles = in_grasp_slice & (
        (radial > cfg.shaft_exclusion_radius_m)
        & (radial <= cfg.grasp_corridor_radius_m)
    )
    obstacle_count = int(np.sum(corridor_obstacles))
    corridor_clear = obstacle_count <= cfg.maximum_grasp_corridor_obstacle_points

    geometry_reasons: list[str] = []
    if detection.length < cfg.minimum_observed_shaft_length_m:
        geometry_reasons.append("observed_shaft_segment_too_short")
    if not head_resolved:
        geometry_reasons.append("expanded_pin_head_not_resolved")
    if not corridor_clear:
        geometry_reasons.append("grasp_corridor_obstructed")
    geometry_candidate = not geometry_reasons
    reasons = [*geometry_reasons, "source_is_virtual_not_physical"]

    return PinFeatureEstimate(
        detection_id=detection.detection_id,
        estimated_head_center=head_center,
        estimated_head_surface_top=surface_top,
        estimated_grasp_center=grasp,
        clear_shaft_near_head=near_head,
        clear_shaft_near_specimen=near_specimen,
        head_radial_extent_m=radial_extent,
        head_shell_inliers=shell_inliers,
        head_shell_residual_m=shell_fit,
        head_resolved=head_resolved,
        grasp_corridor_obstacle_points=obstacle_count,
        grasp_corridor_clear=corridor_clear,
        geometry_candidate=geometry_candidate,
        physical_pick_allowed=False,
        rejection_reasons=tuple(reasons),
    )


def estimate_all_pin_features(
    points: np.ndarray,
    detections: list[PinAxisDetection],
    *,
    config: PinFeatureConfig | None = None,
) -> list[PinFeatureEstimate]:
    return [
        estimate_pin_features(points, detection, config=config)
        for detection in detections
    ]

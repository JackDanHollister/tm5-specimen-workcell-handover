"""Convert multiview pin consensus into unscored Isaac planning candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .geometry import (
    normalize,
    quaternion_xyzw_from_matrix,
    rotation_matrix_from_z_axis,
    serializable_vec,
)
from .multiview_consensus import ConsensusPin


@dataclass(frozen=True)
class PlanningBridgeConfig:
    orientation_samples: int = 12
    planning_grasp_below_head_m: float = 0.005
    pregrasp_clearance_m: float = 0.045
    lift_distance_m: float = 0.030
    minimum_supporting_views: int = 2
    maximum_axis_disagreement_deg: float = 4.0
    maximum_head_spread_m: float = 0.0015
    maximum_grasp_spread_m: float = 0.0015


@dataclass(frozen=True)
class GraspOrientationCandidate:
    candidate_id: int
    orientation_id: int
    roll_about_pin_axis_deg: float
    grasp_below_head_m: float
    pregrasp_position: np.ndarray
    grasp_position: np.ndarray
    lift_position: np.ndarray
    pin_axis_up: np.ndarray
    tool_z_axis_robot: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    simulation_planning_status: str
    physical_pick_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "orientation_id": int(self.orientation_id),
            "roll_about_pin_axis_deg": round(
                float(self.roll_about_pin_axis_deg), 3
            ),
            "grasp_below_head_m": round(float(self.grasp_below_head_m), 6),
            "pregrasp_position_xyz_m": serializable_vec(self.pregrasp_position),
            "grasp_position_xyz_m": serializable_vec(self.grasp_position),
            "lift_position_xyz_m": serializable_vec(self.lift_position),
            "pin_axis_up": serializable_vec(self.pin_axis_up),
            "tool_z_axis_robot": serializable_vec(self.tool_z_axis_robot),
            "tool_convention": (
                "pin_grasp_tcp local +Z points down the shaft opposite pin_axis_up"
            ),
            "rotation_matrix_row_major": serializable_vec(
                self.rotation_matrix.reshape(-1)
            ),
            "quaternion_xyzw": serializable_vec(self.quaternion_xyzw),
            "simulation_planning_status": self.simulation_planning_status,
            "physical_pick_allowed": False,
        }


@dataclass(frozen=True)
class PinPlanningTarget:
    candidate_id: int
    source_frame_id: str
    target_frame_id: str
    pin_axis_up: np.ndarray
    head_center: np.ndarray
    grasp_center: np.ndarray
    foam_entry: np.ndarray
    supporting_views: tuple[str, ...]
    maximum_axis_disagreement_deg: float
    maximum_head_spread_m: float
    maximum_grasp_spread_m: float
    simulation_planning_allowed: bool
    physical_pick_allowed: bool
    simulation_rejection_reasons: tuple[str, ...]
    physical_rejection_reasons: tuple[str, ...]
    orientations: tuple[GraspOrientationCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "pin_axis_up": serializable_vec(self.pin_axis_up),
            "head_center_xyz_m": serializable_vec(self.head_center),
            "grasp_center_xyz_m": serializable_vec(self.grasp_center),
            "foam_entry_xyz_m": serializable_vec(self.foam_entry),
            "supporting_views": list(self.supporting_views),
            "supporting_view_count": len(self.supporting_views),
            "maximum_axis_disagreement_deg": round(
                float(self.maximum_axis_disagreement_deg), 3
            ),
            "maximum_head_spread_m": round(float(self.maximum_head_spread_m), 6),
            "maximum_grasp_spread_m": round(
                float(self.maximum_grasp_spread_m), 6
            ),
            "simulation_planning_allowed": bool(self.simulation_planning_allowed),
            "simulation_planning_status": (
                "unscored_requires_isaac_cumotion"
                if self.simulation_planning_allowed
                else "rejected_before_planning"
            ),
            "physical_pick_allowed": False,
            "simulation_rejection_reasons": list(
                self.simulation_rejection_reasons
            ),
            "physical_rejection_reasons": list(
                self.physical_rejection_reasons
            ),
            "orientations": [item.to_dict() for item in self.orientations],
        }


@dataclass(frozen=True)
class OrientationPlanAssessment:
    """Planner evidence for one sampled grasp orientation."""

    candidate_id: int
    orientation_id: int
    path_found: bool
    sampled_self_collision: bool
    minimum_clearance_m: float | None
    joint_limit_margin_rad: float | None
    path_length_rad: float | None
    goal_tolerance_met: bool = True

    def rejection_reasons(self, *, minimum_clearance_m: float) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.path_found:
            reasons.append("planner_found_no_path")
        if self.sampled_self_collision:
            reasons.append("sampled_self_collision")
        if not self.goal_tolerance_met:
            reasons.append("goal_tolerance_not_met")
        if self.minimum_clearance_m is None:
            reasons.append("minimum_clearance_not_reported")
        elif not np.isfinite(self.minimum_clearance_m):
            reasons.append("minimum_clearance_invalid")
        elif self.minimum_clearance_m < minimum_clearance_m:
            reasons.append("minimum_clearance_below_required")
        if self.joint_limit_margin_rad is None:
            reasons.append("joint_limit_margin_not_reported")
        elif (
            not np.isfinite(self.joint_limit_margin_rad)
            or self.joint_limit_margin_rad < 0.0
        ):
            reasons.append("joint_limit_margin_invalid")
        if self.path_length_rad is None:
            reasons.append("path_length_not_reported")
        elif not np.isfinite(self.path_length_rad) or self.path_length_rad < 0.0:
            reasons.append("path_length_invalid")
        return tuple(reasons)

    def to_dict(self, *, minimum_clearance_m: float) -> dict[str, Any]:
        reasons = self.rejection_reasons(
            minimum_clearance_m=minimum_clearance_m
        )
        return {
            **asdict(self),
            "accepted": not reasons,
            "rejection_reasons": list(reasons),
            "physical_pick_allowed": False,
        }


def _validated_config(config: PlanningBridgeConfig) -> PlanningBridgeConfig:
    if config.orientation_samples < 1:
        raise ValueError("orientation_samples must be positive")
    for name in (
        "planning_grasp_below_head_m",
        "pregrasp_clearance_m",
        "lift_distance_m",
        "maximum_axis_disagreement_deg",
        "maximum_head_spread_m",
        "maximum_grasp_spread_m",
    ):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.minimum_supporting_views < 1:
        raise ValueError("minimum_supporting_views must be positive")
    return config


def validate_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("source_to_target must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError("source_to_target must have homogeneous final row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError("source_to_target rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-8):
        raise ValueError("source_to_target rotation must be right-handed")
    return transform


def transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    transform = validate_rigid_transform(matrix)
    value = np.asarray(point, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("point must contain three finite values")
    return transform[:3, :3] @ value + transform[:3, 3]


def transform_direction(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    transform = validate_rigid_transform(matrix)
    return normalize(transform[:3, :3] @ normalize(direction))


def _foam_entry(
    point_on_axis: np.ndarray,
    axis_up: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    normal = normalize(plane_normal)
    denominator = float(np.dot(axis_up, normal))
    if abs(denominator) < 1.0e-9:
        raise ValueError("pin axis is parallel to the drawer plane")
    distance = float(np.dot(plane_origin - point_on_axis, normal)) / denominator
    return point_on_axis + distance * axis_up


def _simulation_rejections(
    consensus: ConsensusPin,
    config: PlanningBridgeConfig,
) -> list[str]:
    reasons: list[str] = []
    if not consensus.accepted:
        reasons.append("multiview_consensus_not_accepted")
    if len(consensus.supporting_views) < config.minimum_supporting_views:
        reasons.append("insufficient_supporting_views_for_planning")
    if consensus.maximum_axis_disagreement_deg > config.maximum_axis_disagreement_deg:
        reasons.append("axis_disagreement_above_planning_gate")
    if consensus.maximum_head_spread_m > config.maximum_head_spread_m:
        reasons.append("head_spread_above_planning_gate")
    if consensus.maximum_grasp_spread_m > config.maximum_grasp_spread_m:
        reasons.append("grasp_spread_above_planning_gate")
    return list(dict.fromkeys(reasons))


def _physical_rejections(consensus: ConsensusPin) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *consensus.rejection_reasons,
                "isaac_cumotion_assessment_not_run",
                "simulation_candidate_not_physical_authority",
            ]
        )
    )


def _sample_orientations(
    *,
    candidate_id: int,
    pin_axis_up: np.ndarray,
    head_position: np.ndarray,
    config: PlanningBridgeConfig,
    x_hint_target: np.ndarray,
) -> tuple[GraspOrientationCandidate, ...]:
    tool_z = -normalize(pin_axis_up)
    reference = rotation_matrix_from_z_axis(tool_z, x_hint=x_hint_target)
    grasp_position = (
        head_position - config.planning_grasp_below_head_m * pin_axis_up
    )
    pregrasp = grasp_position + config.pregrasp_clearance_m * pin_axis_up
    lift = grasp_position + config.lift_distance_m * pin_axis_up
    candidates: list[GraspOrientationCandidate] = []
    for orientation_id in range(config.orientation_samples):
        roll = 2.0 * math.pi * orientation_id / config.orientation_samples
        cosine = math.cos(roll)
        sine = math.sin(roll)
        local_roll = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = reference @ local_roll
        candidates.append(
            GraspOrientationCandidate(
                candidate_id=candidate_id,
                orientation_id=orientation_id,
                roll_about_pin_axis_deg=math.degrees(roll),
                grasp_below_head_m=config.planning_grasp_below_head_m,
                pregrasp_position=pregrasp.copy(),
                grasp_position=grasp_position.copy(),
                lift_position=lift.copy(),
                pin_axis_up=pin_axis_up.copy(),
                tool_z_axis_robot=tool_z.copy(),
                rotation_matrix=rotation,
                quaternion_xyzw=quaternion_xyzw_from_matrix(rotation),
                simulation_planning_status="unscored_requires_isaac_cumotion",
                physical_pick_allowed=False,
            )
        )
    return tuple(candidates)


def build_pin_planning_target(
    consensus: ConsensusPin,
    *,
    source_frame_id: str,
    target_frame_id: str,
    source_to_target: np.ndarray,
    source_plane_origin: np.ndarray,
    source_plane_normal: np.ndarray,
    x_hint_target: np.ndarray | None = None,
    config: PlanningBridgeConfig | None = None,
) -> PinPlanningTarget:
    """Transform one consensus pin and sample unconstrained jaw rolls.

    Generated poses are simulation candidates only. Isaac/cuMotion still has to
    test every orientation for reachability and collision clearance.
    """
    if not source_frame_id.strip() or not target_frame_id.strip():
        raise ValueError("source and target frame identifiers must not be empty")
    cfg = _validated_config(config or PlanningBridgeConfig())
    transform = validate_rigid_transform(source_to_target)
    axis = transform_direction(transform, consensus.axis_up)
    head = transform_point(transform, consensus.head_center)
    grasp = transform_point(transform, consensus.grasp_center)
    plane_origin = transform_point(transform, np.asarray(source_plane_origin, dtype=float))
    plane_normal = transform_direction(
        transform,
        np.asarray(source_plane_normal, dtype=float),
    )
    entry = _foam_entry(grasp, axis, plane_origin, plane_normal)
    simulation_reasons = _simulation_rejections(consensus, cfg)
    physical_reasons = _physical_rejections(consensus)
    planning_allowed = not simulation_reasons
    hint = (
        np.array([1.0, 0.0, 0.0])
        if x_hint_target is None
        else normalize(np.asarray(x_hint_target, dtype=float))
    )
    orientations = (
        _sample_orientations(
            candidate_id=consensus.candidate_id,
            pin_axis_up=axis,
            head_position=head,
            config=cfg,
            x_hint_target=hint,
        )
        if planning_allowed
        else tuple()
    )
    return PinPlanningTarget(
        candidate_id=consensus.candidate_id,
        source_frame_id=source_frame_id,
        target_frame_id=target_frame_id,
        pin_axis_up=axis,
        head_center=head,
        grasp_center=grasp,
        foam_entry=entry,
        supporting_views=consensus.supporting_views,
        maximum_axis_disagreement_deg=consensus.maximum_axis_disagreement_deg,
        maximum_head_spread_m=consensus.maximum_head_spread_m,
        maximum_grasp_spread_m=consensus.maximum_grasp_spread_m,
        simulation_planning_allowed=planning_allowed,
        physical_pick_allowed=False,
        simulation_rejection_reasons=tuple(simulation_reasons),
        physical_rejection_reasons=tuple(physical_reasons),
        orientations=orientations,
    )


def select_best_orientation(
    assessments: list[OrientationPlanAssessment],
    *,
    minimum_clearance_m: float,
) -> OrientationPlanAssessment | None:
    """Select only among planner-accepted orientations.

    Clearance is primary, followed by joint-limit margin and shorter joint-space
    path length. This function never grants physical execution authority.
    """
    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m < 0.0:
        raise ValueError("minimum_clearance_m must be finite and non-negative")
    accepted = [
        item
        for item in assessments
        if not item.rejection_reasons(minimum_clearance_m=minimum_clearance_m)
    ]
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda item: (
            float(item.minimum_clearance_m),
            float(item.joint_limit_margin_rad),
            -float(item.path_length_rad),
            -int(item.orientation_id),
        ),
    )


def bridge_metadata(config: PlanningBridgeConfig | None = None) -> dict[str, Any]:
    cfg = _validated_config(config or PlanningBridgeConfig())
    return {
        "config": asdict(cfg),
        "orientation_sampling": "uniform_roll_about_detected_pin_axis",
        "ranking_order": [
            "maximum_minimum_sampled_clearance",
            "maximum_joint_limit_margin",
            "minimum_joint_space_path_length",
        ],
        "planner_required": "Isaac/cuMotion",
        "physical_pick_allowed": False,
    }

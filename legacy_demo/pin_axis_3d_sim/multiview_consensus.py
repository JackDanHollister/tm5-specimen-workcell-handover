"""Per-view pin refinement and fail-closed multiview pose consensus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from .detection import DetectionResult, PinAxisDetection
from .geometry import (
    angle_between_deg,
    fit_line_pca,
    line_distances,
    line_endpoints_from_points,
    normalize,
    serializable_vec,
)
from .pin_features import (
    PinFeatureConfig,
    PinFeatureEstimate,
    estimate_pin_features,
)


@dataclass(frozen=True)
class ViewRefinementConfig:
    shaft_start_below_head_m: float = 0.003
    shaft_end_below_head_m: float = 0.0115
    shaft_search_radius_m: float = 0.0013
    minimum_refinement_points: int = 8


@dataclass(frozen=True)
class RefinedViewPin:
    view_name: str
    detection: PinAxisDetection
    features: PinFeatureEstimate
    refinement_point_count: int

    def to_dict(self) -> dict:
        return {
            "view_name": self.view_name,
            "detection": self.detection.to_dict(),
            "features": self.features.to_dict(),
            "refinement_point_count": int(self.refinement_point_count),
        }


@dataclass(frozen=True)
class ConsensusConfig:
    candidate_head_match_distance_m: float = 0.006
    candidate_axis_match_angle_deg: float = 12.0
    minimum_supporting_views: int = 2
    maximum_axis_disagreement_deg: float = 4.0
    maximum_head_spread_m: float = 0.0015
    maximum_grasp_spread_m: float = 0.0015


@dataclass(frozen=True)
class ConsensusPin:
    candidate_id: int
    supporting_views: tuple[str, ...]
    axis_up: np.ndarray
    head_center: np.ndarray
    grasp_center: np.ndarray
    maximum_axis_disagreement_deg: float
    maximum_head_spread_m: float
    maximum_grasp_spread_m: float
    accepted: bool
    physical_pick_allowed: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "candidate_id": int(self.candidate_id),
            "supporting_views": list(self.supporting_views),
            "supporting_view_count": len(self.supporting_views),
            "axis_up": serializable_vec(self.axis_up),
            "head_center": serializable_vec(self.head_center),
            "grasp_center": serializable_vec(self.grasp_center),
            "maximum_axis_disagreement_deg": round(
                float(self.maximum_axis_disagreement_deg), 3
            ),
            "maximum_head_spread_m": round(float(self.maximum_head_spread_m), 6),
            "maximum_grasp_spread_m": round(
                float(self.maximum_grasp_spread_m), 6
            ),
            "accepted": bool(self.accepted),
            "physical_pick_allowed": False,
            "rejection_reasons": list(self.rejection_reasons),
        }

    def as_detection(self) -> PinAxisDetection:
        base = self.head_center - 0.012 * self.axis_up
        return PinAxisDetection(
            detection_id=self.candidate_id,
            point_on_axis=self.head_center,
            axis_up=self.axis_up,
            base=base,
            head=self.head_center,
            length=0.012,
            inlier_count=len(self.supporting_views),
            cluster_point_count=len(self.supporting_views),
            radial_rms=max(self.maximum_head_spread_m, 1.0e-6),
            angle_from_plane_normal_deg=angle_between_deg(
                self.axis_up,
                np.array([0.0, 0.0, 1.0]),
            ),
            score=float(len(self.supporting_views)),
        )


def _validate_refinement_config(config: ViewRefinementConfig) -> None:
    if not 0.0 < config.shaft_start_below_head_m < config.shaft_end_below_head_m:
        raise ValueError("shaft refinement interval must be positive and ordered")
    if config.shaft_search_radius_m <= 0.0:
        raise ValueError("shaft_search_radius_m must be positive")
    if config.minimum_refinement_points < 2:
        raise ValueError("minimum_refinement_points must be at least two")


def refine_view_detection(
    points: np.ndarray,
    detection: PinAxisDetection,
    plane_normal: np.ndarray,
    *,
    config: ViewRefinementConfig | None = None,
    feature_config: PinFeatureConfig | None = None,
) -> tuple[PinAxisDetection, PinFeatureEstimate, int]:
    """Refit only the nominal clear shaft segment inferred below its head."""
    cfg = config or ViewRefinementConfig()
    _validate_refinement_config(cfg)
    cloud = np.asarray(points, dtype=float)
    initial_features = estimate_pin_features(
        cloud,
        detection,
        config=feature_config,
    )
    relative = cloud - initial_features.estimated_head_center
    axial = relative @ detection.axis_up
    radial = line_distances(cloud, detection.point_on_axis, detection.axis_up)
    mask = (
        (axial <= -cfg.shaft_start_below_head_m)
        & (axial >= -cfg.shaft_end_below_head_m)
        & (radial <= cfg.shaft_search_radius_m)
    )
    refinement_points = cloud[mask]
    if refinement_points.shape[0] < cfg.minimum_refinement_points:
        return detection, initial_features, int(refinement_points.shape[0])

    point, axis = fit_line_pca(refinement_points, orient_hint=plane_normal)
    base, head, length = line_endpoints_from_points(refinement_points, point, axis)
    residual = line_distances(refinement_points, point, axis)
    radial_rms = float(np.sqrt(np.mean(residual * residual)))
    refined = PinAxisDetection(
        detection_id=detection.detection_id,
        point_on_axis=point,
        axis_up=axis,
        base=base,
        head=head,
        length=float(length),
        inlier_count=int(refinement_points.shape[0]),
        cluster_point_count=detection.cluster_point_count,
        radial_rms=radial_rms,
        angle_from_plane_normal_deg=angle_between_deg(axis, plane_normal),
        score=float(refinement_points.shape[0] * length / max(radial_rms, 1.0e-5)),
    )
    features = estimate_pin_features(cloud, refined, config=feature_config)
    return refined, features, int(refinement_points.shape[0])


def refine_view_result(
    view_name: str,
    points: np.ndarray,
    detection: DetectionResult,
    *,
    config: ViewRefinementConfig | None = None,
    feature_config: PinFeatureConfig | None = None,
) -> list[RefinedViewPin]:
    return [
        RefinedViewPin(
            view_name=view_name,
            detection=refined,
            features=features,
            refinement_point_count=count,
        )
        for item in detection.detections
        for refined, features, count in [
            refine_view_detection(
                points,
                item,
                detection.plane.normal,
                config=config,
                feature_config=feature_config,
            )
        ]
    ]


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        a = self.find(first)
        b = self.find(second)
        if a != b:
            self.parent[b] = a


def _maximum_pairwise_axis_angle(items: list[RefinedViewPin]) -> float:
    if len(items) < 2:
        return 0.0
    return max(
        angle_between_deg(
            first.detection.axis_up,
            second.detection.axis_up,
            unsigned_axis=True,
        )
        for first, second in combinations(items, 2)
    )


def _consensus_geometry(
    members: list[RefinedViewPin],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Fuse one observation per view and return its agreement metrics."""
    weights = np.asarray(
        [
            max(member.detection.inlier_count, 1)
            / max(member.detection.radial_rms, 1.0e-4)
            for member in members
        ],
        dtype=float,
    )
    weights /= np.sum(weights)
    reference_axis = members[0].detection.axis_up
    oriented_axes = np.vstack(
        [
            member.detection.axis_up
            if np.dot(member.detection.axis_up, reference_axis) >= 0.0
            else -member.detection.axis_up
            for member in members
        ]
    )
    axis = normalize(np.sum(weights[:, None] * oriented_axes, axis=0))
    heads = np.vstack(
        [member.features.estimated_head_center for member in members]
    )
    grasps = np.vstack(
        [member.features.estimated_grasp_center for member in members]
    )
    head = np.sum(weights[:, None] * heads, axis=0)
    grasp = np.sum(weights[:, None] * grasps, axis=0)
    return (
        axis,
        head,
        grasp,
        _maximum_pairwise_axis_angle(members),
        float(np.max(np.linalg.norm(heads - head, axis=1))),
        float(np.max(np.linalg.norm(grasps - grasp, axis=1))),
    )


def _robust_agreeing_members(
    members: list[RefinedViewPin],
    config: ConsensusConfig,
) -> list[RefinedViewPin]:
    """Keep the largest independently observed subset that passes every gate.

    A third view is useful evidence, but one poor shaft fit must not veto two
    other calibrated views that agree within the original fail-closed limits.
    No gate is relaxed: an outlier is discarded only when a complete subset of
    at least ``minimum_supporting_views`` passes all three agreement checks.
    """
    if len(members) <= config.minimum_supporting_views:
        return members
    viable: list[tuple[tuple[float, ...], list[RefinedViewPin]]] = []
    for size in range(len(members), config.minimum_supporting_views - 1, -1):
        for subset_tuple in combinations(members, size):
            subset = list(subset_tuple)
            _, _, _, axis_disagreement, head_spread, grasp_spread = (
                _consensus_geometry(subset)
            )
            if (
                axis_disagreement <= config.maximum_axis_disagreement_deg
                and head_spread <= config.maximum_head_spread_m
                and grasp_spread <= config.maximum_grasp_spread_m
            ):
                normalized_disagreement = (
                    axis_disagreement / config.maximum_axis_disagreement_deg
                    + head_spread / config.maximum_head_spread_m
                    + grasp_spread / config.maximum_grasp_spread_m
                )
                total_score = sum(item.detection.score for item in subset)
                viable.append(
                    ((-float(size), normalized_disagreement, -total_score), subset)
                )
        if viable:
            break
    return min(viable, key=lambda item: item[0])[1] if viable else members


def build_consensus(
    view_results: list[RefinedViewPin],
    *,
    config: ConsensusConfig | None = None,
) -> list[ConsensusPin]:
    cfg = config or ConsensusConfig()
    eligible = [item for item in view_results if item.features.geometry_candidate]
    if not eligible:
        return []
    union = _UnionFind(len(eligible))
    for first_index, second_index in combinations(range(len(eligible)), 2):
        first = eligible[first_index]
        second = eligible[second_index]
        if first.view_name == second.view_name:
            continue
        head_distance = float(
            np.linalg.norm(
                first.features.estimated_head_center
                - second.features.estimated_head_center
            )
        )
        axis_angle = angle_between_deg(
            first.detection.axis_up,
            second.detection.axis_up,
            unsigned_axis=True,
        )
        if (
            head_distance <= cfg.candidate_head_match_distance_m
            and axis_angle <= cfg.candidate_axis_match_angle_deg
        ):
            union.union(first_index, second_index)

    groups: dict[int, list[RefinedViewPin]] = {}
    for index, item in enumerate(eligible):
        groups.setdefault(union.find(index), []).append(item)

    provisional: list[ConsensusPin] = []
    for items in groups.values():
        # One observation per view is allowed to influence a consensus pose.
        unique: dict[str, RefinedViewPin] = {}
        for item in items:
            existing = unique.get(item.view_name)
            if existing is None or item.detection.score > existing.detection.score:
                unique[item.view_name] = item
        members = _robust_agreeing_members(list(unique.values()), cfg)
        (
            axis,
            head,
            grasp,
            axis_disagreement,
            head_spread,
            grasp_spread,
        ) = _consensus_geometry(members)
        reasons: list[str] = []
        if len(members) < cfg.minimum_supporting_views:
            reasons.append("insufficient_supporting_views")
        if axis_disagreement > cfg.maximum_axis_disagreement_deg:
            reasons.append("axis_disagreement_above_gate")
        if head_spread > cfg.maximum_head_spread_m:
            reasons.append("head_position_disagreement_above_gate")
        if grasp_spread > cfg.maximum_grasp_spread_m:
            reasons.append("grasp_position_disagreement_above_gate")
        provisional.append(
            ConsensusPin(
                candidate_id=-1,
                supporting_views=tuple(sorted(unique)),
                axis_up=axis,
                head_center=head,
                grasp_center=grasp,
                maximum_axis_disagreement_deg=axis_disagreement,
                maximum_head_spread_m=head_spread,
                maximum_grasp_spread_m=grasp_spread,
                accepted=not reasons,
                physical_pick_allowed=False,
                rejection_reasons=tuple([*reasons, "source_is_virtual_not_physical"]),
            )
        )

    ordered = sorted(
        provisional,
        key=lambda item: (float(item.head_center[0]), float(item.head_center[1])),
    )
    return [
        ConsensusPin(
            candidate_id=index,
            supporting_views=item.supporting_views,
            axis_up=item.axis_up,
            head_center=item.head_center,
            grasp_center=item.grasp_center,
            maximum_axis_disagreement_deg=item.maximum_axis_disagreement_deg,
            maximum_head_spread_m=item.maximum_head_spread_m,
            maximum_grasp_spread_m=item.maximum_grasp_spread_m,
            accepted=item.accepted,
            physical_pick_allowed=False,
            rejection_reasons=item.rejection_reasons,
        )
        for index, item in enumerate(ordered)
    ]


def config_metadata(
    refinement: ViewRefinementConfig,
    consensus: ConsensusConfig,
) -> dict:
    return {
        "view_refinement": asdict(refinement),
        "consensus": asdict(consensus),
    }

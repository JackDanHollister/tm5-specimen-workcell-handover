"""Conservative primitive collision scene for the virtual museum drawer."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .geometry import rotation_matrix_from_z_axis, serializable_vec
from .pin_planning_bridge import validate_rigid_transform


@dataclass(frozen=True)
class DrawerCollisionConfig:
    foam_thickness_m: float = 0.040
    surface_clutter_height_m: float = 0.0025
    wall_thickness_m: float = 0.012
    wall_height_m: float = 0.035
    specimen_padding_xy_m: float = 0.001
    specimen_padding_z_m: float = 0.001
    selected_open_body_width_m: float = 0.005
    selected_open_wing_thickness_m: float = 0.0015
    other_pin_shaft_width_m: float = 0.0016
    other_pin_head_radius_m: float = 0.003


@dataclass(frozen=True)
class CandidateSpecimenAssociation:
    candidate_id: int
    specimen_id: int
    foam_entry: np.ndarray
    specimen_center: np.ndarray
    lateral_distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "specimen_id": int(self.specimen_id),
            "foam_entry_xyz_m": serializable_vec(self.foam_entry, digits=9),
            "specimen_center_xyz_m": serializable_vec(
                self.specimen_center,
                digits=9,
            ),
            "lateral_distance_m": round(float(self.lateral_distance_m), 9),
            "association_basis": (
                "nearest unique specimen centre to detected-axis foam intersection"
            ),
            "ground_truth_match_used": False,
        }


@dataclass(frozen=True)
class CollisionPrimitive:
    primitive_type: str
    role: str
    position: np.ndarray
    rotation_matrix: np.ndarray
    side_lengths: np.ndarray | None = None
    radius_m: float | None = None
    source_specimen_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "primitive_type": self.primitive_type,
            "role": self.role,
            "source_specimen_id": self.source_specimen_id,
            "position_xyz_m": serializable_vec(self.position, digits=9),
            "rotation_matrix_row_major": serializable_vec(
                self.rotation_matrix.reshape(-1), digits=9
            ),
            "physical_pick_allowed": False,
        }
        if self.side_lengths is not None:
            result["side_lengths_m"] = serializable_vec(
                self.side_lengths, digits=9
            )
        if self.radius_m is not None:
            result["radius_m"] = round(float(self.radius_m), 9)
        return result


def _validated_config(config: DrawerCollisionConfig) -> DrawerCollisionConfig:
    for name, value in asdict(config).items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return config


def _point(transform: np.ndarray, value: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ np.asarray(value, dtype=float) + transform[:3, 3]


def _cuboid(
    *,
    role: str,
    position: np.ndarray,
    side_lengths: np.ndarray,
    rotation_matrix: np.ndarray,
    source_specimen_id: int | None = None,
) -> CollisionPrimitive:
    sides = np.asarray(side_lengths, dtype=float)
    if sides.shape != (3,) or not np.all(np.isfinite(sides)) or np.any(sides <= 0.0):
        raise ValueError(f"{role} cuboid sides must be three finite positive values")
    return CollisionPrimitive(
        primitive_type="cuboid",
        role=role,
        position=np.asarray(position, dtype=float),
        rotation_matrix=np.asarray(rotation_matrix, dtype=float),
        side_lengths=sides,
        source_specimen_id=source_specimen_id,
    )


def _sphere(
    *,
    role: str,
    position: np.ndarray,
    radius_m: float,
    source_specimen_id: int,
) -> CollisionPrimitive:
    return CollisionPrimitive(
        primitive_type="sphere",
        role=role,
        position=np.asarray(position, dtype=float),
        rotation_matrix=np.eye(3),
        radius_m=float(radius_m),
        source_specimen_id=source_specimen_id,
    )


def transform_collision_primitive(
    primitive: CollisionPrimitive, transform: np.ndarray
) -> CollisionPrimitive:
    """Apply a rigid transform to a collision primitive."""

    rigid = validate_rigid_transform(transform)
    rotation = rigid[:3, :3] @ primitive.rotation_matrix
    return CollisionPrimitive(
        primitive_type=primitive.primitive_type,
        role=primitive.role,
        position=_point(rigid, primitive.position),
        rotation_matrix=rotation,
        side_lengths=(
            None if primitive.side_lengths is None else primitive.side_lengths.copy()
        ),
        radius_m=primitive.radius_m,
        source_specimen_id=primitive.source_specimen_id,
    )


def collision_primitives_intersect(
    first: CollisionPrimitive,
    second: CollisionPrimitive,
    *,
    required_clearance_m: float = 0.0,
) -> bool:
    """Test cuboid/sphere primitives with a symmetric clearance inflation."""

    clearance = float(required_clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("required clearance must be finite and nonnegative")
    supported = {"cuboid", "sphere"}
    if first.primitive_type not in supported or second.primitive_type not in supported:
        raise ValueError("only cuboid and sphere collision primitives are supported")
    inflation = clearance / 2.0

    if first.primitive_type == "sphere" and second.primitive_type == "sphere":
        if first.radius_m is None or second.radius_m is None:
            raise ValueError("sphere radius is missing")
        distance = float(np.linalg.norm(first.position - second.position))
        return distance <= first.radius_m + second.radius_m + clearance

    if first.primitive_type == "sphere":
        sphere, box = first, second
    elif second.primitive_type == "sphere":
        sphere, box = second, first
    else:
        sphere = None
        box = None
    if sphere is not None and box is not None:
        if sphere.radius_m is None or box.side_lengths is None:
            raise ValueError("sphere/cuboid dimensions are missing")
        local = box.rotation_matrix.T @ (sphere.position - box.position)
        half = box.side_lengths / 2.0 + inflation
        closest = np.clip(local, -half, half)
        return float(np.linalg.norm(local - closest)) <= sphere.radius_m + inflation

    if first.side_lengths is None or second.side_lengths is None:
        raise ValueError("cuboid side lengths are missing")
    # Separating-axis theorem for two 3-D oriented boxes.  The small epsilon
    # handles nearly parallel axes without weakening the requested clearance.
    axes_a = first.rotation_matrix
    axes_b = second.rotation_matrix
    half_a = first.side_lengths / 2.0 + inflation
    half_b = second.side_lengths / 2.0 + inflation
    relative = axes_a.T @ axes_b
    absolute = np.abs(relative) + 1.0e-12
    translation = axes_a.T @ (second.position - first.position)
    for index in range(3):
        if abs(translation[index]) > half_a[index] + float(
            np.dot(half_b, absolute[index, :])
        ):
            return False
    for index in range(3):
        projected = abs(float(np.dot(translation, relative[:, index])))
        if projected > half_b[index] + float(np.dot(half_a, absolute[:, index])):
            return False
    for a_index in range(3):
        for b_index in range(3):
            projected = abs(
                translation[(a_index + 2) % 3]
                * relative[(a_index + 1) % 3, b_index]
                - translation[(a_index + 1) % 3]
                * relative[(a_index + 2) % 3, b_index]
            )
            radius_a = (
                half_a[(a_index + 1) % 3]
                * absolute[(a_index + 2) % 3, b_index]
                + half_a[(a_index + 2) % 3]
                * absolute[(a_index + 1) % 3, b_index]
            )
            radius_b = (
                half_b[(b_index + 1) % 3]
                * absolute[a_index, (b_index + 2) % 3]
                + half_b[(b_index + 2) % 3]
                * absolute[a_index, (b_index + 1) % 3]
            )
            if projected > radius_a + radius_b:
                return False
    return True


def _specimens(result: dict[str, Any]) -> list[dict[str, Any]]:
    drawer = result.get("drawer")
    if not isinstance(drawer, dict):
        raise ValueError("realistic drawer metadata is missing")
    specimens = drawer.get("specimens")
    if not isinstance(specimens, list) or not specimens:
        raise ValueError("realistic drawer specimens are missing")
    identifiers = [int(item["specimen_id"]) for item in specimens]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("specimen identifiers must be unique")
    return specimens


def _clean_consensus_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    clean = [
        item
        for item in result.get("profiles", [])
        if item.get("profile_name") == "clean"
    ]
    if len(clean) != 1:
        raise ValueError("realistic drawer result must contain one clean profile")
    candidates = clean[0].get("consensus_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("clean consensus geometry candidates are missing")
    accepted = [item for item in candidates if item.get("accepted") is True]
    if not accepted:
        raise ValueError("clean consensus has no accepted geometry candidates")
    return accepted


def candidate_specimen_associations(
    result: dict[str, Any],
    *,
    maximum_lateral_distance_m: float = 0.012,
) -> tuple[CandidateSpecimenAssociation, ...]:
    """Associate detections to virtual payloads without evaluation truth IDs.

    The detector supplies a head point and an upward shaft axis. Intersecting
    that line with the known foam plane produces the pin entry point. Each entry
    is then paired once with the closest specimen centre. The virtual specimen
    identifiers are retained only so the existing synthetic payload and
    collision geometry can be selected after this geometry-only association.
    """

    if (
        not math.isfinite(maximum_lateral_distance_m)
        or maximum_lateral_distance_m <= 0.0
    ):
        raise ValueError("maximum association distance must be finite and positive")
    candidates = _clean_consensus_candidates(result)
    specimens = _specimens(result)
    specimen_centres: dict[int, np.ndarray] = {}
    for specimen in specimens:
        specimen_id = int(specimen["specimen_id"])
        centre = np.asarray(specimen["center_xyz_m"], dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError(f"specimen {specimen_id} centre is invalid")
        specimen_centres[specimen_id] = centre

    candidate_entries: dict[int, np.ndarray] = {}
    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        head = np.asarray(candidate["head_center"], dtype=float)
        axis = np.asarray(candidate["axis_up"], dtype=float)
        if (
            head.shape != (3,)
            or axis.shape != (3,)
            or not np.all(np.isfinite(head))
            or not np.all(np.isfinite(axis))
        ):
            raise ValueError(f"candidate {candidate_id} geometry is invalid")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-12:
            raise ValueError(f"candidate {candidate_id} axis is degenerate")
        axis /= axis_norm
        if axis[2] <= 1.0e-6:
            raise ValueError(f"candidate {candidate_id} axis does not rise from foam")
        entry = head - (head[2] / axis[2]) * axis
        candidate_entries[candidate_id] = entry

    pair_distances = sorted(
        (
            float(np.linalg.norm(entry[:2] - centre[:2])),
            candidate_id,
            specimen_id,
        )
        for candidate_id, entry in candidate_entries.items()
        for specimen_id, centre in specimen_centres.items()
    )
    assigned_candidates: set[int] = set()
    assigned_specimens: set[int] = set()
    associations: list[CandidateSpecimenAssociation] = []
    for distance, candidate_id, specimen_id in pair_distances:
        if candidate_id in assigned_candidates or specimen_id in assigned_specimens:
            continue
        if distance > maximum_lateral_distance_m:
            continue
        assigned_candidates.add(candidate_id)
        assigned_specimens.add(specimen_id)
        associations.append(
            CandidateSpecimenAssociation(
                candidate_id=candidate_id,
                specimen_id=specimen_id,
                foam_entry=candidate_entries[candidate_id],
                specimen_center=specimen_centres[specimen_id],
                lateral_distance_m=distance,
            )
        )

    missing = sorted(set(candidate_entries) - assigned_candidates)
    if missing:
        raise ValueError(
            "consensus candidates lack unique geometry associations within "
            f"{maximum_lateral_distance_m:.6f} m: {missing}"
        )
    return tuple(sorted(associations, key=lambda item: item.candidate_id))


def candidate_to_specimen_map(result: dict[str, Any]) -> dict[int, int]:
    """Return the geometry-derived candidate-to-payload association."""

    return {
        item.candidate_id: item.specimen_id
        for item in candidate_specimen_associations(result)
    }


def build_empty_drawer_collision_primitives(
    *,
    drawer_width_m: float,
    drawer_height_m: float,
    source_to_target: np.ndarray,
    config: DrawerCollisionConfig | None = None,
) -> tuple[CollisionPrimitive, ...]:
    """Build foam and walls for an empty virtual destination drawer."""

    cfg = _validated_config(config or DrawerCollisionConfig())
    transform = validate_rigid_transform(source_to_target)
    width = float(drawer_width_m)
    height = float(drawer_height_m)
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0.0:
        raise ValueError("drawer dimensions must be finite and positive")
    rotation = transform[:3, :3]
    wall_z = cfg.wall_height_m / 2.0
    primitives = [
        _cuboid(
            role="destination_foam",
            position=_point(
                transform,
                np.array([0.0, 0.0, -cfg.foam_thickness_m / 2.0]),
            ),
            side_lengths=np.array([width, height, cfg.foam_thickness_m]),
            rotation_matrix=rotation,
        )
    ]
    wall_specs = (
        (
            np.array([-width / 2.0 - cfg.wall_thickness_m / 2.0, 0.0, wall_z]),
            np.array(
                [cfg.wall_thickness_m, height + 2.0 * cfg.wall_thickness_m, cfg.wall_height_m]
            ),
        ),
        (
            np.array([width / 2.0 + cfg.wall_thickness_m / 2.0, 0.0, wall_z]),
            np.array(
                [cfg.wall_thickness_m, height + 2.0 * cfg.wall_thickness_m, cfg.wall_height_m]
            ),
        ),
        (
            np.array([0.0, -height / 2.0 - cfg.wall_thickness_m / 2.0, wall_z]),
            np.array([width, cfg.wall_thickness_m, cfg.wall_height_m]),
        ),
        (
            np.array([0.0, height / 2.0 + cfg.wall_thickness_m / 2.0, wall_z]),
            np.array([width, cfg.wall_thickness_m, cfg.wall_height_m]),
        ),
    )
    primitives.extend(
        _cuboid(
            role="destination_drawer_wall",
            position=_point(transform, position),
            side_lengths=sides,
            rotation_matrix=rotation,
        )
        for position, sides in wall_specs
    )
    return tuple(primitives)


def build_placed_specimen_collision_primitives(
    result: dict[str, Any],
    *,
    placements: list[dict[str, Any]],
    source_to_target: np.ndarray,
    config: DrawerCollisionConfig | None = None,
) -> tuple[CollisionPrimitive, ...]:
    """Build conservative bodies and vertical pins already placed downstream."""

    cfg = _validated_config(config or DrawerCollisionConfig())
    transform = validate_rigid_transform(source_to_target)
    frame_rotation = transform[:3, :3]
    specimens = {int(item["specimen_id"]): item for item in _specimens(result)}
    seen: set[int] = set()
    primitives: list[CollisionPrimitive] = []
    for placement in placements:
        specimen_id = int(placement["specimen_id"])
        if specimen_id in seen:
            raise ValueError(f"duplicate placed specimen: {specimen_id}")
        if specimen_id not in specimens:
            raise ValueError(f"placed specimen is not present: {specimen_id}")
        seen.add(specimen_id)
        specimen = specimens[specimen_id]
        slot = np.asarray(placement["slot_local_xyz_m"], dtype=float)
        if slot.shape != (3,) or not np.all(np.isfinite(slot)):
            raise ValueError(f"placed specimen {specimen_id} slot is invalid")
        yaw = math.radians(float(placement.get("destination_body_yaw_deg", 0.0)))
        if not math.isfinite(yaw):
            raise ValueError(f"placed specimen {specimen_id} yaw is invalid")
        yaw_rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        specimen_rotation = frame_rotation @ yaw_rotation
        maximum_height = float(specimen["maximum_body_height_m"])
        width_m = float(specimen["width_m"])
        body_length_m = float(specimen["body_length_m"])
        pin_length_m = float(specimen["pin_length_m"])
        dimensions = (maximum_height, width_m, body_length_m, pin_length_m)
        if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
            raise ValueError(f"placed specimen {specimen_id} dimensions are invalid")
        primitives.append(
            _cuboid(
                role="placed_specimen_body",
                source_specimen_id=specimen_id,
                position=_point(
                    transform, slot + np.array([0.0, 0.0, maximum_height / 2.0])
                ),
                side_lengths=np.array(
                    [
                        width_m + 2.0 * cfg.specimen_padding_xy_m,
                        body_length_m + 2.0 * cfg.specimen_padding_xy_m,
                        maximum_height + 2.0 * cfg.specimen_padding_z_m,
                    ]
                ),
                rotation_matrix=specimen_rotation,
            )
        )
        primitives.append(
            _cuboid(
                role="placed_pin_shaft",
                source_specimen_id=specimen_id,
                position=_point(
                    transform, slot + np.array([0.0, 0.0, pin_length_m / 2.0])
                ),
                side_lengths=np.array(
                    [
                        cfg.other_pin_shaft_width_m,
                        cfg.other_pin_shaft_width_m,
                        pin_length_m,
                    ]
                ),
                rotation_matrix=frame_rotation,
            )
        )
        primitives.append(
            _sphere(
                role="placed_pin_head",
                source_specimen_id=specimen_id,
                position=_point(
                    transform, slot + np.array([0.0, 0.0, pin_length_m])
                ),
                radius_m=cfg.other_pin_head_radius_m,
            )
        )
    return tuple(primitives)


def build_drawer_collision_primitives(
    result: dict[str, Any],
    *,
    source_to_target: np.ndarray,
    selected_specimen_id: int,
    config: DrawerCollisionConfig | None = None,
    removed_specimen_ids: set[int] | None = None,
) -> tuple[CollisionPrimitive, ...]:
    """Build a target-specific world for the current sequential drawer state."""
    cfg = _validated_config(config or DrawerCollisionConfig())
    transform = validate_rigid_transform(source_to_target)
    frame_rotation = transform[:3, :3]
    drawer = result["drawer"]
    width = float(drawer["drawer_width_m"])
    height = float(drawer["drawer_height_m"])
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0.0:
        raise ValueError("drawer dimensions must be finite and positive")
    specimens = _specimens(result)
    specimen_ids = {int(item["specimen_id"]) for item in specimens}
    if selected_specimen_id not in specimen_ids:
        raise ValueError("selected specimen is not present in the drawer")
    removed = set() if removed_specimen_ids is None else set(removed_specimen_ids)
    unknown_removed = sorted(removed - specimen_ids)
    if unknown_removed:
        raise ValueError(f"removed specimens are not present: {unknown_removed}")
    if selected_specimen_id in removed:
        raise ValueError("selected specimen is already removed")

    primitives: list[CollisionPrimitive] = [
        _cuboid(
            role="foam",
            position=_point(
                transform,
                np.array([0.0, 0.0, -cfg.foam_thickness_m / 2.0]),
            ),
            side_lengths=np.array([width, height, cfg.foam_thickness_m]),
            rotation_matrix=frame_rotation,
        ),
        _cuboid(
            role="surface_clutter_envelope",
            position=_point(
                transform,
                np.array([0.0, 0.0, cfg.surface_clutter_height_m / 2.0]),
            ),
            side_lengths=np.array([width, height, cfg.surface_clutter_height_m]),
            rotation_matrix=frame_rotation,
        ),
    ]

    wall_z = cfg.wall_height_m / 2.0
    wall_specs = (
        (
            np.array([-width / 2.0 - cfg.wall_thickness_m / 2.0, 0.0, wall_z]),
            np.array(
                [cfg.wall_thickness_m, height + 2.0 * cfg.wall_thickness_m, cfg.wall_height_m]
            ),
        ),
        (
            np.array([width / 2.0 + cfg.wall_thickness_m / 2.0, 0.0, wall_z]),
            np.array(
                [cfg.wall_thickness_m, height + 2.0 * cfg.wall_thickness_m, cfg.wall_height_m]
            ),
        ),
        (
            np.array([0.0, -height / 2.0 - cfg.wall_thickness_m / 2.0, wall_z]),
            np.array([width, cfg.wall_thickness_m, cfg.wall_height_m]),
        ),
        (
            np.array([0.0, height / 2.0 + cfg.wall_thickness_m / 2.0, wall_z]),
            np.array([width, cfg.wall_thickness_m, cfg.wall_height_m]),
        ),
    )
    primitives.extend(
        _cuboid(
            role="drawer_wall",
            position=_point(transform, position),
            side_lengths=sides,
            rotation_matrix=frame_rotation,
        )
        for position, sides in wall_specs
    )

    for specimen in specimens:
        specimen_id = int(specimen["specimen_id"])
        if specimen_id in removed:
            continue
        base = np.asarray(specimen["center_xyz_m"], dtype=float)
        if base.shape != (3,) or not np.all(np.isfinite(base)):
            raise ValueError(f"specimen {specimen_id} base is invalid")
        maximum_height = float(specimen["maximum_body_height_m"])
        width_m = float(specimen["width_m"])
        body_length_m = float(specimen["body_length_m"])
        yaw = math.radians(float(specimen["body_yaw_deg"]))
        yaw_rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        specimen_rotation = frame_rotation @ yaw_rotation
        if (
            specimen_id == selected_specimen_id
            and specimen.get("morphology") == "open_butterfly"
        ):
            # The open specimen cannot be represented as one solid bounding box:
            # doing so fills the empty space above and between its thin wings and
            # falsely collides with every grasp roll. Retain its central body and
            # both wing sides as conservative, morphology-aware primitives.
            wing_height = max(0.0, maximum_height - 0.0043)
            core_height = 0.0056
            core_center = base + np.array([0.0, 0.0, wing_height + 0.0015])
            primitives.append(
                _cuboid(
                    role="selected_specimen_body_core",
                    source_specimen_id=specimen_id,
                    position=_point(transform, core_center),
                    side_lengths=np.array(
                        [
                            cfg.selected_open_body_width_m
                            + 2.0 * cfg.specimen_padding_xy_m,
                            0.62 * body_length_m
                            + 2.0 * cfg.specimen_padding_xy_m,
                            core_height + 2.0 * cfg.specimen_padding_z_m,
                        ]
                    ),
                    rotation_matrix=specimen_rotation,
                )
            )
            for side in (-1.0, 1.0):
                local_offset = yaw_rotation @ np.array(
                    [side * width_m * 0.25, 0.0, wing_height]
                )
                primitives.append(
                    _cuboid(
                        role="selected_specimen_wing",
                        source_specimen_id=specimen_id,
                        position=_point(transform, base + local_offset),
                        side_lengths=np.array(
                            [
                                0.50 * width_m,
                                0.76 * body_length_m,
                                cfg.selected_open_wing_thickness_m
                                + 2.0 * cfg.specimen_padding_z_m,
                            ]
                        ),
                        rotation_matrix=specimen_rotation,
                    )
                )
        else:
            primitives.append(
                _cuboid(
                    role=(
                        "selected_specimen_body"
                        if specimen_id == selected_specimen_id
                        else "specimen_body"
                    ),
                    source_specimen_id=specimen_id,
                    position=_point(
                        transform,
                        base + np.array([0.0, 0.0, maximum_height / 2.0]),
                    ),
                    side_lengths=np.array(
                        [
                            width_m + 2.0 * cfg.specimen_padding_xy_m,
                            body_length_m + 2.0 * cfg.specimen_padding_xy_m,
                            maximum_height + 2.0 * cfg.specimen_padding_z_m,
                        ]
                    ),
                    rotation_matrix=specimen_rotation,
                )
            )
        if specimen_id == selected_specimen_id:
            continue

        axis = np.asarray(specimen["pin_axis_up"], dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or axis_norm <= 1.0e-12:
            raise ValueError(f"specimen {specimen_id} pin axis is invalid")
        axis /= axis_norm
        length = float(specimen["pin_length_m"])
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"specimen {specimen_id} pin length is invalid")
        primitives.append(
            _cuboid(
                role="other_pin_shaft",
                source_specimen_id=specimen_id,
                position=_point(transform, base + axis * length / 2.0),
                side_lengths=np.array(
                    [cfg.other_pin_shaft_width_m, cfg.other_pin_shaft_width_m, length]
                ),
                rotation_matrix=frame_rotation @ rotation_matrix_from_z_axis(axis),
            )
        )
        primitives.append(
            _sphere(
                role="other_pin_head",
                source_specimen_id=specimen_id,
                position=_point(transform, base + axis * length),
                radius_m=cfg.other_pin_head_radius_m,
            )
        )
    return tuple(primitives)


def collision_scene_metadata(
    primitives: tuple[CollisionPrimitive, ...],
    config: DrawerCollisionConfig | None = None,
) -> dict[str, Any]:
    cfg = _validated_config(config or DrawerCollisionConfig())
    roles = Counter(item.role for item in primitives)
    return {
        "config": asdict(cfg),
        "primitive_count": len(primitives),
        "role_counts": dict(sorted(roles.items())),
        "target_contact_policy": (
            "retain selected specimen morphology; omit only selected pin shaft and head"
        ),
        "surface_clutter_policy": (
            "foam-wide conservative height envelope for labels, debris, and loose pins"
        ),
        "geometry_status": "conservative_virtual_primitives_not_physical_registration",
        "physical_pick_allowed": False,
    }

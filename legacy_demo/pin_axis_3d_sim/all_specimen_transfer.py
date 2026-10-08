"""Geometry contracts for indexed all-specimen two-drawer transfers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .geometry import normalize, serializable_vec
from .pin_planning_bridge import validate_rigid_transform


def canonicalize_periodic_joint_path(
    path: list[np.ndarray],
    *,
    joint_index: int,
    minimum_rad: float,
    maximum_rad: float,
    period_rad: float = 2.0 * math.pi,
) -> tuple[np.ndarray, ...]:
    """Choose a continuous periodic-joint representation inside an interval."""

    if not path:
        raise ValueError("periodic joint path must not be empty")
    if joint_index < 0 or any(joint_index >= np.asarray(row).size for row in path):
        raise ValueError("periodic joint index is outside the path")
    if not all(
        math.isfinite(value)
        for value in (minimum_rad, maximum_rad, period_rad)
    ):
        raise ValueError("periodic joint interval must be finite")
    if minimum_rad >= maximum_rad or period_rad <= 0.0:
        raise ValueError("periodic joint interval or period is invalid")
    if maximum_rad - minimum_rad >= period_rad:
        raise ValueError("periodic joint interval must be narrower than one period")

    result = [np.asarray(row, dtype=np.float64).copy() for row in path]
    first = float(result[0][joint_index])
    if first < minimum_rad - 1.0e-12 or first > maximum_rad + 1.0e-12:
        raise ValueError("periodic joint path starts outside the allowed interval")
    previous = first
    for row in result[1:]:
        value = float(row[joint_index])
        lower_k = math.ceil((minimum_rad - value) / period_rad - 1.0e-12)
        upper_k = math.floor((maximum_rad - value) / period_rad + 1.0e-12)
        candidates = [value + period_rad * k for k in range(lower_k, upper_k + 1)]
        if not candidates:
            raise ValueError("periodic joint pose has no allowed equivalent")
        selected = min(candidates, key=lambda candidate: abs(candidate - previous))
        row[joint_index] = selected
        previous = selected
    return tuple(result)


def axis_aligned_source_poses(
    grasp_pose: np.ndarray,
    *,
    pregrasp_clearance_m: float,
    extraction_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pregrasp, grasp and first lift on the detected pin axis."""

    raw_grasp = np.asarray(grasp_pose, dtype=float)
    if raw_grasp.shape != (4, 4) or not np.all(np.isfinite(raw_grasp)):
        raise ValueError("grasp pose must be a finite 4x4 transform")
    grasp = raw_grasp.copy()
    u, _, vt = np.linalg.svd(grasp[:3, :3])
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    grasp[:3, :3] = rotation
    grasp[3] = [0.0, 0.0, 0.0, 1.0]
    grasp = validate_rigid_transform(grasp)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (pregrasp_clearance_m, extraction_distance_m)
    ):
        raise ValueError("source-axis clearances must be finite and positive")
    axis_up = normalize(-grasp[:3, 2])
    pregrasp = grasp.copy()
    pregrasp[:3, 3] += pregrasp_clearance_m * axis_up
    extracted = grasp.copy()
    extracted[:3, 3] += extraction_distance_m * axis_up
    return pregrasp, grasp.copy(), extracted


@dataclass(frozen=True)
class DestinationGridConfig:
    columns: int = 8
    rows: int = 5
    minimum_x_m: float = -0.34
    maximum_x_m: float = 0.34
    minimum_y_m: float = -0.10
    maximum_y_m: float = 0.10


@dataclass(frozen=True)
class DestinationPackingConfig:
    """Clearances for a top-left, column-major museum-drawer layout."""

    drawer_width_m: float = 0.80
    drawer_height_m: float = 0.50
    outer_margin_m: float = 0.030
    minimum_y_m: float = -0.160
    maximum_y_m: float = 0.160
    column_gap_m: float = 0.015
    row_gap_m: float = 0.008
    footprint_padding_m: float = 0.004


@dataclass(frozen=True)
class PackedDestination:
    specimen_id: int
    column_index: int
    row_index: int
    slot_local_xyz_m: np.ndarray
    upright_footprint_width_m: float
    upright_footprint_length_m: float
    source_pin_lateral_offset_m: float
    source_swept_envelope_width_m: float
    source_swept_envelope_length_m: float

    def to_dict(self) -> dict[str, Any]:
        upright_area = (
            self.upright_footprint_width_m * self.upright_footprint_length_m
        )
        source_area = (
            self.source_swept_envelope_width_m
            * self.source_swept_envelope_length_m
        )
        return {
            "specimen_id": self.specimen_id,
            "column_index": self.column_index,
            "row_index": self.row_index,
            "slot_local_xyz_m": serializable_vec(self.slot_local_xyz_m, digits=9),
            "destination_body_yaw_deg": 0.0,
            "upright_footprint_width_m": round(
                self.upright_footprint_width_m, 9
            ),
            "upright_footprint_length_m": round(
                self.upright_footprint_length_m, 9
            ),
            "source_pin_lateral_offset_m": round(
                self.source_pin_lateral_offset_m, 9
            ),
            "source_swept_envelope_width_m": round(
                self.source_swept_envelope_width_m, 9
            ),
            "source_swept_envelope_length_m": round(
                self.source_swept_envelope_length_m, 9
            ),
            "source_swept_envelope_area_m2": round(source_area, 12),
            "upright_footprint_area_m2": round(upright_area, 12),
            "estimated_xy_envelope_area_reduction_fraction": round(
                max(0.0, 1.0 - upright_area / source_area), 9
            ),
        }


@dataclass(frozen=True)
class FlowPackedDestination:
    """One ordered specimen in a tightly flowed destination column."""

    specimen_id: int
    source_column_index: int
    source_row_index: int
    destination_column_index: int
    destination_row_index: int
    slot_local_xyz_m: np.ndarray
    upright_footprint_width_m: float
    upright_footprint_length_m: float
    source_pin_lateral_offset_m: float
    source_swept_envelope_width_m: float
    source_swept_envelope_length_m: float

    def to_dict(self) -> dict[str, Any]:
        upright_area = (
            self.upright_footprint_width_m * self.upright_footprint_length_m
        )
        source_area = (
            self.source_swept_envelope_width_m
            * self.source_swept_envelope_length_m
        )
        return {
            "specimen_id": self.specimen_id,
            "source_column_index": self.source_column_index,
            "source_row_index": self.source_row_index,
            "destination_column_index": self.destination_column_index,
            "destination_row_index": self.destination_row_index,
            "slot_local_xyz_m": serializable_vec(self.slot_local_xyz_m, digits=9),
            "destination_body_yaw_deg": 0.0,
            "upright_footprint_width_m": round(
                self.upright_footprint_width_m, 9
            ),
            "upright_footprint_length_m": round(
                self.upright_footprint_length_m, 9
            ),
            "source_pin_lateral_offset_m": round(
                self.source_pin_lateral_offset_m, 9
            ),
            "source_swept_envelope_width_m": round(
                self.source_swept_envelope_width_m, 9
            ),
            "source_swept_envelope_length_m": round(
                self.source_swept_envelope_length_m, 9
            ),
            "source_swept_envelope_area_m2": round(source_area, 12),
            "upright_footprint_area_m2": round(upright_area, 12),
            "estimated_xy_envelope_area_reduction_fraction": round(
                max(0.0, 1.0 - upright_area / source_area), 9
            ),
        }


def _validated_packing_config(
    config: DestinationPackingConfig,
) -> DestinationPackingConfig:
    for name, value in asdict(config).items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if min(config.drawer_width_m, config.drawer_height_m) <= 0.0:
        raise ValueError("drawer dimensions must be positive")
    if config.outer_margin_m < 0.0 or config.footprint_padding_m < 0.0:
        raise ValueError("packing margins and padding must be nonnegative")
    if config.column_gap_m < 0.0 or config.row_gap_m < 0.0:
        raise ValueError("packing gaps must be nonnegative")
    if 2.0 * config.outer_margin_m >= min(
        config.drawer_width_m, config.drawer_height_m
    ):
        raise ValueError("packing outer margin leaves no usable drawer area")
    if config.minimum_y_m >= config.maximum_y_m:
        raise ValueError("packing y bounds must be ordered")
    if (
        config.minimum_y_m < -config.drawer_height_m / 2.0
        or config.maximum_y_m > config.drawer_height_m / 2.0
    ):
        raise ValueError("packing y bounds must remain inside the drawer")
    return config


def _ordered_specimen_columns(
    specimens: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if not specimens:
        raise ValueError("at least one specimen is required for packing")
    grouped: dict[int, list[dict[str, Any]]] = {}
    identifiers: set[int] = set()
    locations: set[tuple[int, int]] = set()
    for specimen in specimens:
        specimen_id = int(specimen["specimen_id"])
        column = int(specimen["column_index"])
        row = int(specimen["row_index"])
        if specimen_id in identifiers:
            raise ValueError(f"duplicate specimen identifier: {specimen_id}")
        if min(column, row) < 0 or (column, row) in locations:
            raise ValueError(
                "specimen column and row indices must be unique and nonnegative"
            )
        identifiers.add(specimen_id)
        locations.add((column, row))
        grouped.setdefault(column, []).append(specimen)
    ordered_columns = sorted(grouped)
    if ordered_columns != list(range(len(ordered_columns))):
        raise ValueError("specimen columns must be contiguous from zero")

    result: list[tuple[dict[str, Any], ...]] = []
    for column in ordered_columns:
        members = sorted(grouped[column], key=lambda item: int(item["row_index"]))
        rows = [int(item["row_index"]) for item in members]
        if rows != list(range(len(rows))):
            raise ValueError(f"rows in column {column} must be contiguous from zero")
        dimensions = [
            float(item[name])
            for item in members
            for name in ("width_m", "body_length_m")
        ]
        if not all(
            math.isfinite(value) and value > 0.0 for value in dimensions
        ):
            raise ValueError("specimen footprint dimensions must be finite and positive")
        result.append(tuple(members))
    return tuple(result)


def pack_verticalized_specimens(
    specimens: list[dict[str, Any]],
    config: DestinationPackingConfig | None = None,
) -> tuple[PackedDestination, ...]:
    """Pack specimens in columns, top-to-bottom and then left-to-right.

    The destination footprint is based on the insect body after the pin is made
    vertical.  The source pin tilt is retained separately as a conservative
    lateral swept-envelope estimate; it does not artificially enlarge the
    final upright packing footprint.
    """

    cfg = _validated_packing_config(config or DestinationPackingConfig())
    if not specimens:
        raise ValueError("at least one specimen is required for packing")
    grouped: dict[int, list[dict[str, Any]]] = {}
    identifiers: set[int] = set()
    locations: set[tuple[int, int]] = set()
    for specimen in specimens:
        specimen_id = int(specimen["specimen_id"])
        column = int(specimen["column_index"])
        row = int(specimen["row_index"])
        if specimen_id in identifiers:
            raise ValueError(f"duplicate specimen identifier: {specimen_id}")
        if min(column, row) < 0 or (column, row) in locations:
            raise ValueError("specimen column and row indices must be unique and nonnegative")
        identifiers.add(specimen_id)
        locations.add((column, row))
        grouped.setdefault(column, []).append(specimen)
    ordered_columns = sorted(grouped)
    if ordered_columns != list(range(len(ordered_columns))):
        raise ValueError("specimen columns must be contiguous from zero")

    effective_widths: dict[int, float] = {}
    for column, members in grouped.items():
        rows = sorted(int(item["row_index"]) for item in members)
        if rows != list(range(len(rows))):
            raise ValueError(f"rows in column {column} must be contiguous from zero")
        widths = [float(item["width_m"]) for item in members]
        lengths = [float(item["body_length_m"]) for item in members]
        if not all(math.isfinite(value) and value > 0.0 for value in widths + lengths):
            raise ValueError("specimen footprint dimensions must be finite and positive")
        effective_widths[column] = max(widths) + 2.0 * cfg.footprint_padding_m

    packed_width = sum(effective_widths.values()) + cfg.column_gap_m * max(
        0, len(ordered_columns) - 1
    )
    usable_width = cfg.drawer_width_m - 2.0 * cfg.outer_margin_m
    if packed_width > usable_width + 1.0e-12:
        raise ValueError(
            f"packed columns require {packed_width:.6f} m but only "
            f"{usable_width:.6f} m is available"
        )

    packed: list[PackedDestination] = []
    x_cursor = -cfg.drawer_width_m / 2.0 + cfg.outer_margin_m
    for column in ordered_columns:
        column_width = effective_widths[column]
        x_center = x_cursor + column_width / 2.0
        # The source drawer's row zero is at local +Y.  Start at the same
        # visual top edge in the destination and fill toward local -Y so that
        # source top-left maps to destination top-left even though the two
        # drawers have different world rotations.
        y_cursor = cfg.maximum_y_m
        members = sorted(grouped[column], key=lambda item: int(item["row_index"]))
        for specimen in members:
            length = float(specimen["body_length_m"]) + 2.0 * cfg.footprint_padding_m
            width = float(specimen["width_m"]) + 2.0 * cfg.footprint_padding_m
            y_center = y_cursor - length / 2.0
            if y_center - length / 2.0 < cfg.minimum_y_m - 1.0e-12:
                raise ValueError(
                    f"column {column} does not fit within the drawer height"
                )
            axis = normalize(np.asarray(specimen["pin_axis_up"], dtype=float))
            pin_length = float(specimen["pin_length_m"])
            if not math.isfinite(pin_length) or pin_length <= 0.0:
                raise ValueError("pin lengths must be finite and positive")
            lateral = pin_length * axis[:2]
            packed.append(
                PackedDestination(
                    specimen_id=int(specimen["specimen_id"]),
                    column_index=column,
                    row_index=int(specimen["row_index"]),
                    slot_local_xyz_m=np.array([x_center, y_center, 0.0]),
                    upright_footprint_width_m=width,
                    upright_footprint_length_m=length,
                    source_pin_lateral_offset_m=float(np.linalg.norm(lateral)),
                    source_swept_envelope_width_m=width + 2.0 * abs(float(lateral[0])),
                    source_swept_envelope_length_m=length + 2.0 * abs(float(lateral[1])),
                )
            )
            y_cursor -= length + cfg.row_gap_m
        x_cursor += column_width + cfg.column_gap_m
    return tuple(packed)


def pack_verticalized_specimens_ordered_flow(
    specimens: list[dict[str, Any]],
    config: DestinationPackingConfig | None = None,
) -> tuple[FlowPackedDestination, ...]:
    """Flow one source-ordered stream through tightly filled columns.

    Source columns are flattened in column/row order.  Each destination column
    is filled from its top edge until the next ordered footprint cannot fit;
    that same footprint starts the next column.  No specimen can be skipped or
    reordered to fill a smaller remainder.
    """

    cfg = _validated_packing_config(config or DestinationPackingConfig())
    source_columns = _ordered_specimen_columns(specimens)
    ordered = [item for column in source_columns for item in column]
    available_height = cfg.maximum_y_m - cfg.minimum_y_m

    measured: list[tuple[dict[str, Any], float, float]] = []
    for specimen in ordered:
        width = float(specimen["width_m"]) + 2.0 * cfg.footprint_padding_m
        length = float(specimen["body_length_m"]) + 2.0 * cfg.footprint_padding_m
        if length > available_height + 1.0e-12:
            raise ValueError(
                f"specimen {int(specimen['specimen_id'])} does not fit within "
                "the destination packing height"
            )
        measured.append((specimen, width, length))

    destination_columns: list[list[tuple[dict[str, Any], float, float]]] = []
    current: list[tuple[dict[str, Any], float, float]] = []
    used_height = 0.0
    for record in measured:
        additional = record[2] + (cfg.row_gap_m if current else 0.0)
        if current and used_height + additional > available_height + 1.0e-12:
            destination_columns.append(current)
            current = []
            used_height = 0.0
            additional = record[2]
        current.append(record)
        used_height += additional
    if current:
        destination_columns.append(current)

    column_widths = [
        max(width for _, width, _ in column) for column in destination_columns
    ]
    packed_width = sum(column_widths) + cfg.column_gap_m * max(
        0, len(column_widths) - 1
    )
    usable_width = cfg.drawer_width_m - 2.0 * cfg.outer_margin_m
    if packed_width > usable_width + 1.0e-12:
        raise ValueError(
            f"flow-packed columns require {packed_width:.6f} m but only "
            f"{usable_width:.6f} m is available"
        )

    packed: list[FlowPackedDestination] = []
    x_cursor = -cfg.drawer_width_m / 2.0 + cfg.outer_margin_m
    for destination_column, (members, column_width) in enumerate(
        zip(destination_columns, column_widths)
    ):
        x_center = x_cursor + column_width / 2.0
        y_cursor = cfg.maximum_y_m
        for destination_row, (specimen, width, length) in enumerate(members):
            y_center = y_cursor - length / 2.0
            axis = normalize(np.asarray(specimen["pin_axis_up"], dtype=float))
            pin_length = float(specimen["pin_length_m"])
            if not math.isfinite(pin_length) or pin_length <= 0.0:
                raise ValueError("pin lengths must be finite and positive")
            lateral = pin_length * axis[:2]
            packed.append(
                FlowPackedDestination(
                    specimen_id=int(specimen["specimen_id"]),
                    source_column_index=int(specimen["column_index"]),
                    source_row_index=int(specimen["row_index"]),
                    destination_column_index=destination_column,
                    destination_row_index=destination_row,
                    slot_local_xyz_m=np.array([x_center, y_center, 0.0]),
                    upright_footprint_width_m=width,
                    upright_footprint_length_m=length,
                    source_pin_lateral_offset_m=float(np.linalg.norm(lateral)),
                    source_swept_envelope_width_m=(
                        width + 2.0 * abs(float(lateral[0]))
                    ),
                    source_swept_envelope_length_m=(
                        length + 2.0 * abs(float(lateral[1]))
                    ),
                )
            )
            y_cursor -= length + cfg.row_gap_m
        x_cursor += column_width + cfg.column_gap_m
    return tuple(packed)


def verticalizing_payload_rotation(
    *,
    source_pin_axis_world: np.ndarray,
    source_body_axis_world: np.ndarray,
    destination_body_axis_world: np.ndarray,
) -> np.ndarray:
    """Return the rigid rotation that makes the pin vertical and body aligned."""

    source_z = normalize(np.asarray(source_pin_axis_world, dtype=float))
    body = normalize(np.asarray(source_body_axis_world, dtype=float))
    source_x_raw = body - float(np.dot(body, source_z)) * source_z
    if float(np.linalg.norm(source_x_raw)) <= 1.0e-9:
        raise ValueError("source body axis is parallel to the source pin")
    source_x = normalize(source_x_raw)
    source_y = normalize(np.cross(source_z, source_x))
    source_x = normalize(np.cross(source_y, source_z))

    destination_z = np.array([0.0, 0.0, 1.0])
    destination_body = normalize(np.asarray(destination_body_axis_world, dtype=float))
    destination_x_raw = destination_body.copy()
    destination_x_raw[2] = 0.0
    if float(np.linalg.norm(destination_x_raw)) <= 1.0e-9:
        raise ValueError("destination body axis must have a horizontal component")
    destination_x = normalize(destination_x_raw)
    destination_y = normalize(np.cross(destination_z, destination_x))
    destination_x = normalize(np.cross(destination_y, destination_z))

    source_frame = np.column_stack((source_x, source_y, source_z))
    destination_frame = np.column_stack(
        (destination_x, destination_y, destination_z)
    )
    rotation = destination_frame @ source_frame.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9):
        raise ValueError("verticalizing payload rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-9):
        raise ValueError("verticalizing payload rotation is not right-handed")
    return rotation


def destination_slots(config: DestinationGridConfig | None = None) -> tuple[np.ndarray, ...]:
    cfg = config or DestinationGridConfig()
    if cfg.columns < 1 or cfg.rows < 1:
        raise ValueError("destination grid dimensions must be positive")
    values = (
        cfg.minimum_x_m,
        cfg.maximum_x_m,
        cfg.minimum_y_m,
        cfg.maximum_y_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("destination grid bounds must be finite")
    if cfg.minimum_x_m > cfg.maximum_x_m or cfg.minimum_y_m > cfg.maximum_y_m:
        raise ValueError("destination grid bounds must be ordered")
    xs = np.linspace(cfg.minimum_x_m, cfg.maximum_x_m, cfg.columns)
    ys = np.linspace(cfg.minimum_y_m, cfg.maximum_y_m, cfg.rows)
    return tuple(np.array([x, y, 0.0]) for y in ys for x in xs)


def indexed_drawer_transform(
    *,
    local_slot_xyz_m: np.ndarray,
    world_drop_xyz_m: np.ndarray,
    yaw_deg: float,
) -> np.ndarray:
    slot = np.asarray(local_slot_xyz_m, dtype=float)
    drop = np.asarray(world_drop_xyz_m, dtype=float)
    if slot.shape != (3,) or drop.shape != (3,) or not np.all(np.isfinite([slot, drop])):
        raise ValueError("slot and drop positions must contain three finite values")
    if not math.isfinite(yaw_deg):
        raise ValueError("drawer yaw must be finite")
    yaw = math.radians(yaw_deg)
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = drop - rotation @ slot
    return validate_rigid_transform(transform)


def rigid_payload_transform(
    *,
    source_tool_rotation: np.ndarray,
    destination_tool_rotation: np.ndarray,
    source_foam_entry_world_m: np.ndarray,
    destination_pin_base_world_m: np.ndarray,
) -> np.ndarray:
    source_rotation = np.asarray(source_tool_rotation, dtype=float)
    destination_rotation = np.asarray(destination_tool_rotation, dtype=float)
    entry = np.asarray(source_foam_entry_world_m, dtype=float)
    base = np.asarray(destination_pin_base_world_m, dtype=float)
    if source_rotation.shape != (3, 3) or destination_rotation.shape != (3, 3):
        raise ValueError("tool rotations must be 3x3 matrices")
    source_u, _, source_vt = np.linalg.svd(source_rotation)
    destination_u, _, destination_vt = np.linalg.svd(destination_rotation)
    source_rotation = source_u @ source_vt
    destination_rotation = destination_u @ destination_vt
    rotation = destination_rotation @ source_rotation.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = base - rotation @ entry
    return validate_rigid_transform(transform)


def transfer_geometry_record(
    *,
    slot: np.ndarray,
    destination_drawer_transform: np.ndarray,
    payload_transform: np.ndarray,
    source_axis_world: np.ndarray,
) -> dict:
    mapped_axis = normalize(payload_transform[:3, :3] @ normalize(source_axis_world))
    return {
        "destination_slot_local_xyz_m": serializable_vec(slot, digits=9),
        "destination_drawer_transform_row_major": serializable_vec(
            destination_drawer_transform.reshape(-1), digits=9
        ),
        "payload_source_world_to_destination_world_row_major": serializable_vec(
            payload_transform.reshape(-1), digits=9
        ),
        "destination_detected_pin_axis_world": serializable_vec(mapped_axis, digits=9),
        "destination_detected_pin_tilt_deg": round(
            math.degrees(math.acos(float(np.clip(mapped_axis[2], -1.0, 1.0)))),
            9,
        ),
    }

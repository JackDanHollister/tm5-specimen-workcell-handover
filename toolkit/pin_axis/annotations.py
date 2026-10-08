"""Checked sparse labels and specimen-separated training splits."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from inference import sha256


def load_checked(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    if document.get("schema") != "tm5_pin_annotations_v1":
        raise ValueError("Expected an exported tm5_pin_annotations_v1 file")
    rows, ids, specimens, hashes = [], set(), {}, {}
    for row in document["images"]:
        if row.get("reviewed") is not True:
            continue
        if row["id"] in ids:
            raise ValueError("Duplicate image ID")
        ids.add(row["id"])
        specimen = row.get("specimen_id", "").strip()
        split = row.get("split")
        if not specimen or not row.get("placement_id", "").strip():
            raise ValueError("Checked examples need specimen and placement IDs")
        if split not in {"train", "validation", "test"}:
            raise ValueError("Assign each checked specimen to train, validation or test")
        if specimens.setdefault(specimen, split) != split:
            raise ValueError(f"Specimen leakage across splits: {specimen}")
        source = Path(row["source_path"])
        digest = sha256(source)
        if digest != row["source_sha256"]:
            raise ValueError(f"Image hash mismatch: {source}")
        if hashes.setdefault(digest, split) != split:
            raise ValueError("The same image appears in different splits")
        with Image.open(source) as image:
            size = np.array(image.size)
        for segment in row["segments"]:
            points = np.array(segment["points"], dtype=float)
            width = float(segment["width_px"])
            if (segment["kind"] not in {"pin", "not_pin"} or points.shape != (2, 2)
                    or not np.isfinite(points).all() or (points < 0).any() or (points >= size).any()
                    or not np.isfinite(width) or not 1 <= width <= 80
                    or np.linalg.norm(points[1] - points[0]) < 4):
                raise ValueError(f"Invalid annotation segment: {row['id']}")
        if row.get("visibility") == "unusable":
            continue
        if not row["segments"] or row.get("visibility") not in {"visible", "partial"}:
            raise ValueError("Checked usable examples need visible-shaft or negative annotations")
        rows.append(row)
    if not rows:
        raise ValueError("No checked usable annotations; mark the saved images and export them first")
    if not {"train", "validation"}.issubset({r["split"] for r in rows}):
        raise ValueError("Need training and validation examples from different specimens")
    return rows


def annotation_masks(row: dict, size: tuple[int, int]) -> np.ndarray:
    masks = []
    for kind in ("pin", "not_pin"):
        mask = Image.new("L", size)
        draw = ImageDraw.Draw(mask)
        for segment in row["segments"]:
            if segment["kind"] == kind:
                draw.line([tuple(p) for p in segment["points"]], fill=255,
                          width=max(1, round(segment["width_px"])))
        masks.append(np.array(mask, dtype=np.float32) / 255)
    return np.stack(masks)


def sparse_patch_labels(masks: np.ndarray, feature_shape: tuple[int, int]) -> np.ndarray:
    height, width = feature_shape
    fractions = np.stack([cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
                          for mask in masks])
    pin, negative = fractions > 0.02
    labels = np.full((height, width), -1, dtype=np.int64)
    labels[pin & ~negative] = 1
    labels[negative & ~pin] = 0
    # Unmarked and contradictory patches stay unknown; neither is background.
    return labels


def crop_boxes(masks: np.ndarray, side: int = 1024) -> list[tuple[int, int, int, int]]:
    height, width = masks.shape[1:]
    occupied = np.any(masks > 0, axis=0)
    boxes = []
    for y in range(0, height, side):
        for x in range(0, width, side):
            if occupied[y:y + side, x:x + side].any():
                boxes.append((x, y, min(width, x + side), min(height, y + side)))
    return boxes

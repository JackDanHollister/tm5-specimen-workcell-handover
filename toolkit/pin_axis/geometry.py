"""DINO-guided paired shaft edges and an undirected 2-D image angle.

All thresholds are development heuristics, not calibrated confidence or accuracy.
Coordinates use original image pixels: x right, y down, angle clockwise modulo 180.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

PARAMETERS = {
    "min_edge_px": 25.0, "max_edge_angle_deg": 5.0,
    "min_width_px": 1.5, "max_width_px": 18.0,
    "min_pair_overlap_px": 25.0, "min_evidence_length_px": 80.0,
    "min_pin_probability": 0.20, "min_contrast_255": 7.0,
    "alternative_score_ratio": 0.75,
}


def angle_difference(a: float, b: float) -> float:
    return abs((a - b + 90) % 180 - 90)


def sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.remap(image.astype(np.float32), points[:, 0].astype(np.float32)[None],
                     points[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).ravel()


def line_points(a: np.ndarray, b: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    count = max(2, int(np.linalg.norm(b - a) / spacing) + 1)
    return a + np.linspace(0, 1, count)[:, None] * (b - a)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([float(start), float(end)])
    return merged


def extract_axis(rgb: np.ndarray, probabilities: np.ndarray) -> dict:
    height, width = rgb.shape[:2]
    if probabilities.shape[0] != 5 or not np.isfinite(probabilities).all():
        raise ValueError("Expected five finite class-probability maps")
    pin = cv2.resize(probabilities[1], (width, height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result = {"status": "no_reliable_axis", "angle_deg": None,
              "endpoints_xy": None, "axis_3d": None,
              "coordinate_system": "original image pixels; x right, y down",
              "angle_convention": "clockwise from image +x, modulo 180; no head/tip sign",
              "pin_probability_peak": float(pin.max()), "reasons": [],
              "candidates": [], "thresholds": PARAMETERS.copy()}
    if pin.max() < 0.30:
        result["reasons"] = ["Weak pin response from the saved model"]
        return result

    raw = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if raw is None:
        result["reasons"] = ["No straight image edges"]
        return result
    edges = []
    for line in raw.reshape(-1, 4):
        a, b = line[:2].astype(float), line[2:].astype(float)
        length = float(np.linalg.norm(b - a))
        if length < PARAMETERS["min_edge_px"]:
            continue
        support = sample(pin, line_points(a, b, 4))
        if support.mean() < 0.10 or support.max() < 0.30:
            continue
        direction = (b - a) / length
        if direction[0] < 0:
            direction = -direction
        edges.append((a, b, direction, length))

    pairs = []
    for i, (a, b, direction, length) in enumerate(edges):
        normal = np.array([-direction[1], direction[0]])
        for c, d, other, other_length in edges[i + 1:]:
            cosine = abs(float(direction @ other))
            if cosine < math.cos(math.radians(PARAMETERS["max_edge_angle_deg"])):
                continue
            ca, da = float((c - a) @ normal), float((d - a) @ normal)
            separation = abs((ca + da) / 2)
            if (not PARAMETERS["min_width_px"] <= separation <= PARAMETERS["max_width_px"]
                    or ca * da <= 0 or abs(ca - da) > max(4, separation)):
                continue
            first = sorted((float(a @ direction), float(b @ direction)))
            second = sorted((float(c @ direction), float(d @ direction)))
            start, end = max(first[0], second[0]), min(first[1], second[1])
            overlap = end - start
            if overlap < PARAMETERS["min_pair_overlap_px"] or overlap / min(length, other_length) < 0.4:
                continue
            # Interpolate both actual edges at the same longitudinal coordinates.
            def on_edge(p, q, t):
                return p + (q - p) * ((t - p @ direction) / ((q - p) @ direction))
            p = (on_edge(a, b, start) + on_edge(c, d, start)) / 2
            q = (on_edge(a, b, end) + on_edge(c, d, end)) / 2
            pts = line_points(p, q)
            pp = sample(pin, pts)
            side = normal * (separation / 2 + 4)
            center = sample(gray, pts)
            flanks = (sample(gray, pts + side) + sample(gray, pts - side)) / 2
            contrast = float(np.median(np.abs(center - flanks)))
            probability = float(pp.mean())
            if probability < PARAMETERS["min_pin_probability"] or contrast < PARAMETERS["min_contrast_255"]:
                continue
            pair_direction = (q - p) / np.linalg.norm(q - p)
            angle = float(math.degrees(math.atan2(pair_direction[1], pair_direction[0])) % 180)
            score = overlap * probability * min(1.0, contrast / 25)
            pairs.append({"p": p, "q": q, "direction": pair_direction, "angle": angle,
                          "length": overlap, "width": separation, "probability": probability,
                          "contrast": contrast, "score": score})

    # One side of a reflective shaft can disappear into the background. Check
    # narrow bright/dark cross-sections along each surviving edge as a second
    # source of evidence. A step edge alone has no contrast on BOTH flanks.
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.1).astype(np.float32)
    for a, b, direction, length in edges:
        normal = np.array([-direction[1], direction[0]])
        pts = line_points(a, b, 3)
        offsets = np.arange(-12, 13, dtype=float)
        centers = pts[:, None, :] + offsets[None, :, None] * normal
        centers_flat = centers.reshape(-1, 2)
        intensity = sample(smooth, centers_flat).reshape(len(pts), -1)
        support = sample(pin, centers_flat).reshape(len(pts), -1)
        responses, widths = [], []
        for shaft_width in (3, 5, 7, 9, 12, 16):
            side = normal * (shaft_width / 2 + 3)
            left = sample(smooth, centers_flat + side).reshape(intensity.shape)
            right = sample(smooth, centers_flat - side).reshape(intensity.shape)
            dl, dr = intensity - left, intensity - right
            response = np.where(dl * dr > 0, np.minimum(abs(dl), abs(dr)), 0)
            responses.append(response * np.minimum(1, support / 0.4))
            widths.append(shaft_width)
        stack = np.stack(responses)
        # Find a coherent stripe, not unrelated maxima at each cross-section.
        median_response = np.median(stack, axis=1)
        wi, oi = np.unravel_index(np.argmax(median_response), median_response.shape)
        if median_response[wi, oi] < 10:
            continue
        local = stack[wi].copy()
        local[:, abs(offsets - offsets[oi]) > 3] = 0
        indices = local.argmax(axis=1)
        quality = local[np.arange(len(pts)), indices]
        valid = quality >= 8
        if valid.mean() < 0.7 or valid.sum() < 8:
            continue
        ridge = centers[np.arange(len(pts)), indices][valid]
        probability = float(sample(pin, ridge).mean())
        if probability < 0.30:
            continue
        vx, vy, cx, cy = cv2.fitLine(ridge.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        axis = np.array([vx, vy], dtype=float)
        origin = np.array([cx, cy], dtype=float)
        if np.sqrt(np.mean(((ridge - origin) @ np.array([-vy, vx])) ** 2)) > 2.5:
            continue
        projection = (ridge - origin) @ axis
        p, q = origin + np.array([projection.min(), projection.max()])[:, None] * axis
        span = float(np.ptp(projection))
        angle = float(math.degrees(math.atan2(vy, vx)) % 180)
        contrast = float(np.median(quality[valid]))
        pairs.append({"p": p, "q": q, "direction": axis, "angle": angle,
                      "length": span, "width": widths[wi], "probability": probability,
                      "contrast": contrast, "score": span * probability * min(1, contrast / 25)})

    # Combine collinear *visible* intervals; hidden gaps never count as evidence.
    groups = []
    for seed in sorted(pairs, key=lambda p: p["score"], reverse=True):
        direction = seed["direction"]
        normal = np.array([-direction[1], direction[0]])
        if any(angle_difference(seed["angle"], g["angle_deg"]) < 5 and
               abs((seed["p"] - np.array(g["endpoints_xy"][0])) @ normal) < 12 for g in groups):
            continue
        members = [p for p in pairs if angle_difference(seed["angle"], p["angle"]) < 4 and
                   max(abs((p["p"] - seed["p"]) @ normal),
                       abs((p["q"] - seed["p"]) @ normal)) < 7]
        points = np.concatenate([line_points(p["p"], p["q"], 3) for p in members])
        vx, vy, cx, cy = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        axis = np.array([vx, vy], dtype=float)
        origin = np.array([cx, cy], dtype=float)
        projected = (points - origin) @ axis
        ends = origin + np.array([projected.min(), projected.max()])[:, None] * axis
        intervals = merge_intervals([tuple(sorted(((p["p"] - origin) @ axis,
                                                 (p["q"] - origin) @ axis))) for p in members])
        evidence = sum(end - start for start, end in intervals)
        weights = np.array([p["length"] for p in members])
        probability = float(np.average([p["probability"] for p in members], weights=weights))
        contrast = float(np.average([p["contrast"] for p in members], weights=weights))
        residual = float(np.sqrt(np.mean(((points - origin) @ np.array([-vy, vx])) ** 2)))
        groups.append({"angle_deg": float(math.degrees(math.atan2(vy, vx)) % 180),
                       "endpoints_xy": ends.tolist(), "visible_length_px": evidence,
                       "span_px": float(np.ptp(projected)), "fit_rms_px": residual,
                       "mean_pin_probability": probability, "contrast_255": contrast,
                       "width_px": float(np.median([p["width"] for p in members])),
                       "score": evidence * probability * min(1, contrast / 25),
                       "visible_segments_xy": [[(origin + lo * axis).tolist(),
                                                (origin + hi * axis).tolist()] for lo, hi in intervals]})
    groups.sort(key=lambda g: g["score"], reverse=True)
    result["candidates"] = groups[:8]
    if not groups:
        result["reasons"] = ["No narrow, contrasting shaft with pin support"]
        return result
    best = groups[0]
    if best["visible_length_px"] < PARAMETERS["min_evidence_length_px"]:
        result["reasons"].append("Too little visible shaft for this prototype's length gate")
    alternatives = [g for g in groups[1:] if g["visible_length_px"] >= PARAMETERS["min_evidence_length_px"]
                    and g["score"] >= best["score"] * PARAMETERS["alternative_score_ratio"]]
    if alternatives:
        result["reasons"].append("Competing shaft candidates: pin identity needs review")
    if not result["reasons"]:
        result.update(status="candidate_review_required", angle_deg=best["angle_deg"],
                      endpoints_xy=best["endpoints_xy"])
        result["reasons"] = ["Geometric checks passed; correct-pin identity and angle still need visual review"]
    return result

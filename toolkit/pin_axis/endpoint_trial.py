"""Saved-image endpoint experiment; no inference, network or hardware interfaces.

Profiles are measured on the existing, frozen 3-D line. Positive distance is
robot-base +Z (head-up placement assumption), not the top of an image. Semantic
boundaries and image contrast are proposals, never physical head/grasp truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, binary_closing, label


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(points, row):
    pose, K, distortion = (np.asarray(row[k], float) for k in ("camera_pose", "K", "distortion"))
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels = cv2.projectPoints(points, cv2.Rodrigues(pose[:3, :3].T)[0],
                              -pose[:3, :3].T @ pose[:3, 3], K, distortion)[0][:, 0]
    return pixels, camera[:, 2]


def sample(array, pixels):
    return cv2.remap(np.asarray(array, np.float32), pixels[:, 0].astype(np.float32)[None],
                     pixels[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))[0]


def intervals(mask, coordinates, minimum_mm=0.):
    labels, count = label(mask)
    return [(int(np.where(labels == i)[0][0]), int(np.where(labels == i)[0][-1]))
            for i in range(1, count + 1)
            if np.ptp(coordinates[labels == i]) >= minimum_mm]


def profile_view(row, fit, probabilities, rgb, grid):
    anchor = np.asarray(fit["display_center_base_m"])
    axis = np.asarray(fit["axis_base"])
    if axis[2] < 0:
        axis = -axis
    pixels, depth = project(anchor + grid[:, None] / 1000 * axis, row)
    tangent = np.gradient(pixels, axis=0)
    scale = np.linalg.norm(tangent, axis=1) / np.median(np.diff(grid))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    height, width = rgb.shape[:2]
    valid = ((depth > 0) & (pixels[:, 0] >= 24) & (pixels[:, 0] < width - 24) &
             (pixels[:, 1] >= 24) & (pixels[:, 1] < height - 24))
    # Match the existing cv2.resize pixel-centre convention without expanding
    # five full image-sized maps. Camera intrinsics still refer to source pixels.
    probability_profiles = []
    for field in probabilities:
        values = []
        for offset in (-4, 0, 4):
            xy = (pixels + offset * normal + .5) * [field.shape[1]/width, field.shape[0]/height] - .5
            values.append(sample(field, xy))
        probability_profiles.append(np.mean(np.nan_to_num(values), axis=0))
    probability_profiles = np.asarray(probability_profiles)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    strip = np.stack([sample(gray, pixels + offset * normal) for offset in range(-20, 21)], axis=1)
    strip = np.nan_to_num(strip)
    background = np.median(strip[:, [0, 1, 2, 38, 39, 40]], axis=1)
    # Shiny metal can be brighter OR darker than the foam. A dark-only ridge
    # misses the well-lit shaft ends in several otherwise useful views.
    contrast = np.maximum(background - np.min(strip[:, 14:27], axis=1),
                          np.max(strip[:, 14:27], axis=1) - background)
    contrast = gaussian_filter1d(np.nan_to_num(contrast), 1)
    widths = np.sum((background[:, None] - strip) > 12, axis=1)
    support_upper = max((np.asarray(fit["observed_support_extent_base_m"]) - anchor) @ axis * 1000)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    reference_region = (grid > support_upper + 2) & (grid < support_upper + 6) & valid
    broad_occupancy = np.zeros(len(grid))
    if reference_region.sum() > 10:
        reference = np.concatenate([sample(lab, pixels + offset*scale[:,None]*normal)[reference_region]
                                    for offset in (-3, -2, 2, 3)])
        reference = reference[np.isfinite(reference).all(axis=1)]
        if len(reference) > 10:
            foam_color = np.median(reference, axis=0)
            residual = np.linalg.norm(reference - foam_color, axis=1)
            threshold = max(20., float(np.median(residual) + 4*np.median(abs(residual-np.median(residual)))))
            swath = np.stack([sample(lab, pixels + offset*scale[:,None]*normal)
                              for offset in np.linspace(-1.5, 1.5, 25)], axis=1)
            broad_occupancy = gaussian_filter1d(np.mean(np.linalg.norm(swath-foam_color,axis=2)>threshold,axis=1),2)
    median_scale = float(np.median(scale[valid])) if valid.any() else 0.
    camera = np.asarray(row["camera_pose"])
    ray = anchor - camera[:3, 3]; ray /= np.linalg.norm(ray)
    viewing_angle = float(np.degrees(np.arccos(np.clip(abs(ray @ axis), 0, 1))))
    view_fit = next((v for v in fit["views"] if v["id"] == row["id"]), None)
    vote = bool(row["capture_accepted"] and view_fit and view_fit["inlier"] and viewing_angle >= 15 and median_scale >= 4)
    return {"id": row["id"], "specimen_id": row["specimen_id"], "path": row["path"],
            "source_sha256": row["source_sha256"], "vote_eligible": vote,
            "shaft_fit": view_fit, "viewing_angle_deg": viewing_angle,
            "median_pixels_per_mm": median_scale, "pixels": pixels,
            "valid": valid, "probabilities": probability_profiles,
            "contrast": contrast, "dark_width_px": widths, "broad_occupancy": broad_occupancy}


def propose_endpoints(profile, grid, support_upper_mm):
    probabilities = profile["probabilities"]
    pin = gaussian_filter1d(np.nan_to_num(probabilities[1]), 2)
    obstacle = gaussian_filter1d(np.nan_to_num(probabilities[2:].max(axis=0)), 2)
    evidence = ((pin > .30) & profile["valid"] &
                (grid < support_upper_mm + 10) & (grid > support_upper_mm - 22))
    # Bridge only sub-millimetre signal gaps, never missing specimen sections.
    evidence = binary_closing(evidence, structure=np.ones(5))
    segments = intervals(evidence, grid, 1.0)
    segments = [(a, b) for a, b in segments if pin[a:b+1].max() >= .30]
    if not segments:
        return {"status": "no_upper_pin_evidence", "head_candidate_mm": None,
                "obstruction_candidate_mm": None, "exposed_candidate_mm": None,
                "native_obstruction_mm": None, "native_span_mm": None}
    lo, hi = max(segments, key=lambda pair: grid[pair[1]])
    # Refine the free end with native-pixel contrast. DINO is a coarse patch
    # map and may continue into background beyond the actual metal pin.
    local = (grid >= grid[hi] - 1.5) & (grid <= grid[hi] + .8) & profile["valid"]
    reference = ((grid > support_upper_mm + 3) & (grid < support_upper_mm + 8) & profile["valid"])
    threshold = max(18., float(np.quantile(profile["contrast"][reference], .95))) if reference.any() else 18.
    edge_segments = intervals((profile["contrast"] > threshold) & local, grid, .15)
    eligible = [(a, b) for a, b in edge_segments if a <= hi and b >= hi - 8]
    head_index = min(eligible, key=lambda pair: abs(pair[1]-hi))[1] if eligible else hi
    head = float(grid[head_index])
    blocker = ((obstacle > .30) & (obstacle > pin) & profile["valid"] &
               (grid < head - 1.0) & (grid > head - 35))
    blocks = intervals(blocker, grid, 1.0)
    lower = float(grid[max(blocks, key=lambda pair: pair[1])[1]]) if blocks else None
    kind = None
    if lower is not None:
        index = int(np.argmin(abs(grid - lower)))
        kind = ["specimen", "mount", "label"][int(probabilities[2:, index].argmax())]
    broad = ((profile["broad_occupancy"] > .60) & profile["valid"] &
             (grid < head - 1.5) & (grid > head - 35))
    broad_blocks = intervals(broad, grid, .4)
    native_lower = float(grid[max(broad_blocks,key=lambda pair:pair[1])[1]]) if broad_blocks else None
    native_reason = "candidate_projected_obstruction" if native_lower is not None else "no_localized_broad_object"
    if native_lower is not None and head - native_lower <= 1.6:
        # A segment cut off by the search window is censored, not a measured
        # boundary. In particular, do not manufacture a 1.5 mm exposed length.
        native_lower = None
        native_reason = "object_reaches_head_exclusion_region_boundary_unknown"
    return {"status": "candidate_review_required", "head_candidate_mm": head,
            "obstruction_candidate_mm": lower, "obstruction_class": kind,
            "exposed_candidate_mm": None if lower is None else head - lower,
            "head_pixel_xy": profile["pixels"][head_index].tolist(),
            "obstruction_pixel_xy": None if lower is None else profile["pixels"][np.argmin(abs(grid-lower))].tolist(),
            "native_obstruction_mm": native_lower,
            "native_obstruction_reason": native_reason,
            "native_span_mm": None if native_lower is None else head-native_lower}


def consensus(values, tolerance_mm=1.5, minimum_views=3):
    """Densest fixed-radius group; report disagreements, never silently erase."""
    values = [(key, float(value)) for key, value in values if value is not None and np.isfinite(value)]
    if not values:
        return {"status": "insufficient_views", "median_mm": None, "support_ids": [], "other_ids": []}
    array = np.array([v for _, v in values])
    masks = [abs(array - value) <= tolerance_mm for value in array]
    mask = max(masks, key=lambda m: (int(m.sum()), -float(np.ptp(array[m]))))
    for _ in range(len(values)):
        updated = abs(array - np.median(array[mask])) <= tolerance_mm
        if np.array_equal(mask, updated):
            break
        mask = updated
    center = float(np.median(array[mask]))
    majority = mask.sum() >= minimum_views and mask.sum() > len(values)/2
    return {"status": "provisional_consensus" if majority else "insufficient_or_competing_views",
            "median_mm": center, "range_mm": [float(array[mask].min()), float(array[mask].max())],
            "support_ids": [key for (key, _), flag in zip(values, mask) if flag],
            "other_ids": [key for (key, _), flag in zip(values, mask) if not flag],
            "all_view_count": len(values), "tolerance_mm": tolerance_mm,
            "interval_kind": "observed_agreement_not_calibrated_accuracy"}


def run(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Trial output must be separate from the frozen source run")
    if output.exists():
        raise FileExistsError("Use a fresh trial directory; existing evidence is immutable")
    inputs = {name: sha256(source / name) for name in ("reconstruction.json", "detections.json", "cache/model.json")}
    records = json.loads((source / "detections.json").read_text())
    reconstruction = json.loads((source / "reconstruction.json").read_text())
    output.mkdir(parents=True)
    (output / "profiles").mkdir()
    grid = np.arange(-45., 45.0001, .1)
    fits = {g["specimen_id"]: g["primary"] for g in reconstruction["specimens"]}
    summaries = []
    for number, row in enumerate(records):
        path = Path(row["path"])
        if sha256(path) != row["source_sha256"]:
            raise ValueError(f"Source image changed: {row['id']}")
        fit = fits[row["specimen_id"]]
        probabilities_path = source / "cache" / row["id"] / "probabilities.npz"
        probabilities = np.load(probabilities_path, allow_pickle=False)["probabilities"]
        if probabilities.shape[0] != 5 or not np.isfinite(probabilities).all():
            raise ValueError("Invalid cached five-class prediction")
        rgb = np.array(Image.open(path).convert("RGB"))
        profile = profile_view(row, fit, probabilities, rgb, grid)
        support = (np.asarray(fit["observed_support_extent_base_m"]) - fit["display_center_base_m"]) @ np.asarray(fit["axis_base"]) * 1000
        proposal = propose_endpoints(profile, grid, max(support))
        arrays = {key: profile.pop(key) for key in ("pixels", "valid", "probabilities", "contrast", "dark_width_px", "broad_occupancy")}
        np.savez_compressed(output / "profiles" / f"{row['id']}.npz", grid_mm=grid, **arrays)
        summaries.append({**profile, **proposal, "probabilities_sha256": sha256(probabilities_path)})
        if number % 10 == 0:
            print(f"Profiled {number+1}/{len(records)}", flush=True)
    groups = []
    for sid, fit in fits.items():
        views = [r for r in summaries if r["specimen_id"] == sid and r["vote_eligible"]]
        groups.append({"specimen_id": sid, "eligible_views": len(views),
                       "head": consensus([(r["id"], r["head_candidate_mm"]) for r in views]),
                       "obstruction": consensus([(r["id"], r["obstruction_candidate_mm"]) for r in views]),
                       "paired_exposed_length": consensus([(r["id"], r["exposed_candidate_mm"]) for r in views]),
                       "native_obstruction": consensus([(r["id"], r["native_obstruction_mm"]) for r in views]),
                       "native_span": consensus([(r["id"], r["native_span_mm"]) for r in views]),
                       "axis_up": fit["axis_base"], "anchor_base_m": fit["display_center_base_m"]})
    result = {"format_version": 1, "status": "exploratory_review_required", "source": str(source),
              "source_hashes": inputs, "source_code_sha256": sha256(Path(__file__)),
              "physical_pick_allowed": False, "robot_connected": False, "training_performed": False,
              "method": "DINO-assisted native contrast endpoint; head-up sign; frozen 3D shaft profiles",
              "limits": ["head-up placement assumption", "factory calibration physically unverified",
                         "semantic boundaries are not physical contact landmarks", "same development views fit the shaft",
                         "agreement ranges exclude shared calibration and detector bias", "no jaw clearance or force approval",
                         "outer pin end includes pinhead; reported spans are not bare-shaft grip lengths",
                         "native broad-object boundary assumes locally similar foam color; can be projected occlusion"],
              "views": summaries, "specimens": groups}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if inputs != {name: sha256(source / name) for name in inputs}:
        raise RuntimeError("Frozen source changed during trial")
    for group in groups:
        print("Pin",group["specimen_id"],"head supports",len(group["head"]["support_ids"]),
              "of",group["eligible_views"],"native span",group["native_span"]["status"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

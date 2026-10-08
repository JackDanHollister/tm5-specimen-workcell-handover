"""Offline interactive robot-base view and original-image reprojection evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from report import review_crop
from multiview import projected_line

COLORS = ["#32d5b8", "#ffa65d", "#6cb9ff", "#e78aff", "#ff6f82", "#c4df63", "#a99fff"]


def project(points, pose, K, distortion):
    rotation = pose[:3, :3].T
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    rvec = cv2.Rodrigues(rotation)[0]
    tvec = -rotation @ pose[:3, 3]
    pixels = cv2.projectPoints(points, rvec, tvec, K, distortion)[0][:, 0]
    return pixels, camera[:, 2]


def axis_curve(center, axis, pose, K, distortion, size):
    """Clip an undistorted image line before applying lens distortion.

    Projecting arbitrarily distant 3-D endpoints can sample outside the lens'
    field of view, where a distortion polynomial can fold back into the image.
    """
    width, height = size
    line = projected_line(center, axis, {"pose": pose, "K": K})
    direction = np.array([-line[1], line[0]])
    foot = np.array([width / 2, height / 2])
    foot -= line[:2] * (line[:2] @ foot + line[2])
    lower, upper = -float("inf"), float("inf")
    for coordinate, delta, bound in zip(foot, direction, (width - 1, height - 1)):
        if abs(delta) < 1e-12:
            if coordinate < 0 or coordinate > bound:
                return np.empty((0, 2))
            continue
        lo, hi = sorted((-coordinate / delta, (bound - coordinate) / delta))
        lower, upper = max(lower, lo), min(upper, hi)
    if lower >= upper:
        return np.empty((0, 2))
    undistorted = foot + np.linspace(lower, upper, 160)[:, None] * direction
    rays = np.column_stack((undistorted, np.ones(len(undistorted)))) @ np.linalg.inv(K).T
    return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, distortion)[0][:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    records = json.loads((root / "detections.json").read_text())
    results = json.loads((root / "reconstruction.json").read_text())
    assets = root / "reprojection"
    assets.mkdir(exist_ok=True)
    cards = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    groups = {s["specimen_id"]: s for s in results["specimens"]}
    for i, group in enumerate(groups.values()):
        group["color"] = COLORS[i % len(COLORS)]
    for row in records:
        group = groups[row["specimen_id"]]
        solution = group["primary"]
        image = Image.open(row["path"]).convert("RGB")
        probs = np.load(root / "cache" / row["id"] / "probabilities.npz", allow_pickle=False)["probabilities"]
        crop = review_crop(row["geometry"], probs, image.size)
        pose, K, dist = np.array(row["camera_pose"]), np.array(row["K"]), np.array(row["distortion"])
        row["projected_3d_pixels"] = []
        if solution.get("axis_base") is not None:
            center = np.array(solution["display_center_base_m"])
            axis = np.array(solution["axis_base"])
            pixels = axis_curve(center, axis, pose, K, dist, image.size)
            row["projected_3d_pixels"] = pixels.tolist()
            if row["geometry"]["angle_deg"] is None:
                # Locate the crop from the other views' 3-D fit, not a weak blob.
                mid, dep = project(center[None], pose, K, dist)
                if dep[0] > 0 and 0 <= mid[0, 0] < image.width and 0 <= mid[0, 1] < image.height:
                    side = min(900, image.width, image.height)
                    xy = np.clip(mid[0] - side / 2, [0, 0], [image.width - side, image.height - side]).astype(int)
                    crop = [int(xy[0]), int(xy[1]), side, side]
        row["crop_xywh"] = crop
        x, y, w, h = crop
        crop_image = image.crop((x, y, x + w, y + h))
        crop_image.save(assets / f"{row['id']}.jpg", quality=94)
        row["preview"] = f"reprojection/{row['id']}.jpg"
        view_fit = next((v for v in solution["views"] if v["id"] == row["id"]), None)
        row["fit"] = view_fit
        row["undistorted_plane_rays"] = None
        if row["geometry"]["endpoints_xy"] is not None:
            endpoints = np.array(row["geometry"]["endpoints_xy"], float)
            rays = cv2.undistortPoints(endpoints[:, None], K, dist)[:, 0]
            rays = np.column_stack((rays, np.ones(2))) @ pose[:3, :3].T
            row["undistorted_plane_rays"] = rays.tolist()
        draw = ImageDraw.Draw(crop_image)
        if row["geometry"]["angle_deg"] is not None:
            for segment in row["geometry"]["candidates"][0]["visible_segments_xy"]:
                draw.line([tuple(np.array(p) - [x, y]) for p in segment], fill="#00edc5", width=4)
        projected = row["projected_3d_pixels"]
        for a, b in zip(projected[::2], projected[1::2]):
            # Dashed reprojection through hidden regions represents a model line.
            if all(abs(v) < 1e6 for point in (a, b) for v in point):
                draw.line([tuple(np.array(a) - [x, y]), tuple(np.array(b) - [x, y])], fill="#fc65ef", width=3)
        crop_image.save(assets / f"{row['id']}-overlay.jpg", quality=94)
        row["overlay"] = f"reprojection/{row['id']}-overlay.jpg"
    for specimen, group in groups.items():
        eligible = [r for r in records if r["specimen_id"] == specimen and r["fit"] and r["fit"]["inlier"]]
        chosen = sorted(eligible, key=lambda r: r["fit"]["residual_px"])
        # Show two different supporting views and any rejected detection.
        selected = chosen[:1] + chosen[len(chosen)//2:len(chosen)//2+1]
        rejected = [r for r in records if r["specimen_id"] == specimen and r["fit"] and not r["fit"]["inlier"]]
        if rejected:
            selected += rejected[:1]
        if not selected:
            selected = [r for r in records if r["specimen_id"] == specimen][:2]
        for row in selected:
            card = Image.new("RGB", (480, 545), "#101822")
            card.paste(Image.open(root / row["overlay"]).resize((480, 480)), (0, 65))
            d = ImageDraw.Draw(card)
            d.text((10, 8), f"Specimen {specimen} / {row['id']}", font=font, fill="white")
            caption = (f"{'Supports fit' if row['fit']['inlier'] else 'Rejected'} · {row['fit']['residual_px']:.1f} px"
                       if row['fit'] else "No accepted shaft: not used in a 3D fit")
            d.text((10, 34), caption, font=font, fill="#ffc978")
            cards.append(card)
    sheet = Image.new("RGB", (1920, 545 * max(1, (len(cards)+3)//4)), "#101822")
    for i, card in enumerate(cards):sheet.paste(card, ((i%4)*480, (i//4)*545))
    sheet.save(root / "reprojection-contact.jpg", quality=94)

    # Exportable standard scientific plot, alongside the interactive exploration.
    fig = plt.figure(figsize=(13, 6), layout="constrained")
    for panel in (1, 2):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for group in groups.values():
            p = group["primary"]
            if p["axis_base"] is None:continue
            center, axis = np.array(p["display_center_base_m"]), np.array(p["axis_base"])
            segment = (center + np.array([-.025,.025])[:,None] * axis) * 1000
            ax.plot(*segment.T, color=group["color"], linewidth=3, label=f"Specimen {group['specimen_id']}")
            ax.text(*center*1000, group["specimen_id"])
            if panel == 1:
                cameras = np.array([r["camera_pose"] for r in records if r["specimen_id"] == group["specimen_id"]])[:, :3, 3] * 1000
                ax.scatter(*cameras.T, s=7, color=group["color"], alpha=.35)
        if panel == 1:
            links = np.array([records[0]["robot_link_positions"][k] for k in ["base","link_1","link_2","link_3","link_4","link_5","flange"]])*1000
            ax.plot(*links.T,color="gray",linewidth=4,label="Isaac URDF joint chain")
            ax.set_title("Robot base, camera positions and fitted shafts")
        else:
            ax.set_title("Shaft detail: 50 mm display segments")
            ax.legend(fontsize=8)
        ax.set_xlabel("Robot X (mm)");ax.set_ylabel("Robot Y (mm)");ax.set_zlabel("Robot Z (mm)")
        bounds=np.array([ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]);centers=bounds.mean(axis=1);radius=np.ptp(bounds,axis=1).max()/2
        ax.set_xlim(centers[0]-radius,centers[0]+radius);ax.set_ylim(centers[1]-radius,centers[1]+radius);ax.set_zlim(centers[2]-radius,centers[2]+radius);ax.set_box_aspect((1,1,1))
    fig.suptitle("Provisional factory-calibrated reconstruction · shaft lengths shown for display only")
    fig.savefig(root / "robot-space.png",dpi=180);plt.close(fig)
    with (root / "specimen-axes.csv").open("w",newline="") as stream:
        fields=["specimen","captures","detected_views","supporting_views","tilt_deg","azimuth_deg","axis_x","axis_y","axis_z","median_reprojection_px","status"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group in groups.values():
            p=group["primary"];axis=p["axis_base"] or [None]*3
            values = [group["specimen_id"], group["image_count"], group["detected_view_count"],
                      len(p["inlier_ids"]), p.get("tilt_from_base_z_deg"), p.get("azimuth_from_base_x_deg"),
                      *axis, p.get("median_inlier_residual_px"), p["status"]]
            writer.writerow(dict(zip(fields, values)))
    # Keep only browser-useful data; the full audit remains in reconstruction.json.
    payload={"specimens":list(groups.values()),"views":records,"audit":results["frame_audit"]}
    template=Path(__file__).with_name("multiview.html").read_text()
    (root / "index.html").write_text(template.replace("__DATA__",json.dumps(payload,allow_nan=False).replace("<","\\u003c")))
    print(root / "index.html")


if __name__ == "__main__":main()

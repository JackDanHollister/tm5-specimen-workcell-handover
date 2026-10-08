"""Read-only image overlays and cross-view profile plots for endpoint trials."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from endpoint_trial import project

COLORS = {"single": "#ff9e24", "combined": "#00cee8", "obstruction": "#ff3ba0", "native": "#6eff65"}


def crop_for(row, group):
    center, _ = project(np.array(group["anchor_base_m"])[None], row)
    image = Image.open(row["path"]).convert("RGB")
    side = min(900, image.width, image.height)
    start = np.clip(center[0] - side / 2, [0, 0], [image.width-side, image.height-side]).astype(int)
    return image.crop((*start, *(start + side))), start


def draw_view(ax, row, view, group, small=False):
    cropped, origin = crop_for(row, group)
    ax.imshow(cropped)
    anchor, axis = np.array(group["anchor_base_m"]), np.array(group["axis_up"])
    head = group["head"]["median_mm"]
    for key, color, marker, label_text in (("head_candidate_mm", COLORS["single"], "o", "Single-view end"),
            ("obstruction_candidate_mm", COLORS["obstruction"], "s", "DINO boundary"),
            ("native_obstruction_mm", COLORS["native"], "^", "Image-width boundary")):
        value = view.get(key)
        if value is not None:
            pixel, _ = project((anchor + axis * value/1000)[None], row)
            xy = pixel[0] - origin
            ax.scatter(*xy, s=110 if small else 220, facecolors="none", edgecolors=color, marker=marker, linewidths=2, label=label_text)
    if head is not None:
        xy = project((anchor + axis * head/1000)[None], row)[0][0] - origin
        ax.scatter(*xy, s=140 if small else 260, c=COLORS["combined"], marker="+", linewidths=2, label="Combined end")
    start, end = -20, 25
    points, _ = project(anchor + np.array([start, end])[:,None]/1000*axis, row)
    ax.plot(*(points-origin).T, color="white", alpha=.45, lw=.8, linestyle="--")
    ax.set_xlim(0, cropped.width); ax.set_ylim(cropped.height, 0)
    ax.set_xticks([]); ax.set_yticks([])
    label_text = "shaft-fit support" if view["vote_eligible"] else "diagnostic only"
    amount = view.get("native_span_mm")
    estimate = f"image-boundary span {amount:.1f} mm (unverified)" if amount is not None else "no image-boundary span"
    ax.set_title(f"{view['id']} · {label_text}\n{estimate}", fontsize=9 if small else 12)


def select_views(views, group, count=6):
    eligible = [v for v in views if v["vote_eligible"]]
    chosen = []
    def add(row):
        if row is not None and row not in chosen and len(chosen)<count:
            chosen.append(row)
    if eligible:
        for index in (0, len(eligible)//2, len(eligible)-1):
            add(eligible[index])
    add(next((v for v in views if v["id"] in group["head"]["other_ids"]), None))
    add(next((v for v in views if v["vote_eligible"] and v["head_candidate_mm"] is None), None))
    add(next((v for v in views if not v["vote_eligible"]), None))
    for view in views:
        add(view)
    return chosen


def make_report(trial):
    results = json.loads((trial / "results.json").read_text())
    records = {row["id"]: row for row in json.loads((Path(results["source"])/"detections.json").read_text())}
    groups = {g["specimen_id"]: g for g in results["specimens"]}
    images = trial / "views"
    images.mkdir(exist_ok=False)
    plt.rcParams.update({"figure.facecolor": "#101b29", "axes.facecolor": "#101b29", "text.color": "#eaf3ff",
                         "axes.labelcolor": "#eaf3ff", "xtick.color": "#c2d2e4", "ytick.color": "#c2d2e4",
                         "axes.titlecolor": "#eaf3ff", "axes.edgecolor": "#648099"})
    for view in results["views"]:
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_view(ax, records[view["id"]], view, groups[view["specimen_id"]])
        ax.legend(loc="lower left", fontsize=8, facecolor="#101b29", labelcolor="white")
        fig.text(.5, .025, "End-to-boundary candidates include the pinhead · NOT usable grasp lengths", ha="center", fontsize=9)
        fig.savefig(images/f"{view['id']}.jpg", dpi=125, bbox_inches="tight", pil_kwargs={"quality":90})
        plt.close(fig)
    for group in groups.values():
        sid = group["specimen_id"]
        views = [v for v in results["views"] if v["specimen_id"] == sid]
        fig = plt.figure(figsize=(15, 13), layout="constrained")
        layout = fig.add_gridspec(3, 3, height_ratios=[1, 1, .75])
        for index, view in enumerate(select_views(views, group)):
            draw_view(fig.add_subplot(layout[index//3,index%3]), records[view["id"]], view, group, small=True)
        ax = fig.add_subplot(layout[2,:])
        grid = None
        for view in views:
            p = np.load(trial/"profiles"/f"{view['id']}.npz")
            grid = p["grid_mm"]
            if view["vote_eligible"]:
                ax.plot(grid, p["probabilities"][1], alpha=.3, color="#ff9e24", lw=.8)
                ax.plot(grid, p["probabilities"][2:].max(axis=0), alpha=.2, color="#ff3ba0", lw=.8)
                if "broad_occupancy" in p:
                    ax.plot(grid, p["broad_occupancy"], alpha=.15, color=COLORS["native"], lw=.8)
        for key, color in (("head", COLORS["combined"]),("obstruction",COLORS["obstruction"])):
            candidate = group[key]
            if candidate["median_mm"] is not None:
                ax.axvline(candidate["median_mm"], color=color, lw=2)
                ax.axvspan(*candidate["range_mm"], color=color, alpha=.15)
        ax.set_xlim(-25, 30); ax.set_ylim(0,1)
        ax.set_xlabel("Distance along frozen 3D shaft (mm); positive = head-up direction")
        ax.set_ylabel("DINO class score")
        ax.set_title("Orange: pin scores; pink: other DINO classes; green: image-width cue. Bands are agreement, NOT accuracy.", fontsize=10)
        fig.suptitle(f"Pin {sid}: individual-image marks and combined endpoint hypothesis\nOrange circle: single end | Cyan +: combined end | Pink square: DINO boundary | Green triangle: image-width boundary", fontsize=12)
        fig.savefig(trial/f"pin-{sid}-review.png", dpi=125)
        plt.close(fig)
    sections = []
    for group in groups.values():
        sid = group["specimen_id"]
        views = [v for v in results["views"] if v["specimen_id"] == sid]
        thumbnails = "".join(f'<a href="views/{v["id"]}.jpg"><img loading="lazy" src="views/{v["id"]}.jpg" alt="{v["id"]}"></a>' for v in views)
        sections.append(f'<section id="pin-{sid}"><h2>Pin {sid}</h2><img class="summary" src="pin-{sid}-review.png"><details><summary>All {len(views)} views</summary><div class="grid">{thumbnails}</div></details></section>')
    page = '<!doctype html><meta charset="utf-8"><title>Pin endpoint review</title><style>body{background:#101b29;color:#eaf3ff;font:18px system-ui;max-width:1400px;margin:30px auto;padding:16px}a{color:#00cee8}.summary{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px}.grid img{width:100%}section{margin:40px 0}summary{cursor:pointer}</style><h1>Saved-image pin endpoint trial</h1><p>117 images, seven separate placements. No robot connection, new captures or training. Marks are exploratory candidates, not physical measurements. Orange: individual end; cyan: combined end; pink: DINO obstruction; green: image-width obstruction. Detector dropouts and projected occlusions can create false boundaries. Spans include the pinhead and are NOT bare-shaft grasp lengths.</p>' + "".join(sections)
    (trial/"index.html").write_text(page)
    print(trial/"index.html")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",type=Path,required=True)
    make_report(parser.parse_args().trial)

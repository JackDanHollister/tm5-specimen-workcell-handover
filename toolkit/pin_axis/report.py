"""Generate a standalone local review/annotation page from cached predictions."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import extract_axis
from inference import sha256


def data_url(image: Image.Image, kind: str = "PNG") -> str:
    stream = io.BytesIO()
    image.save(stream, format=kind, **({"quality": 94} if kind == "JPEG" else {}))
    mime = "jpeg" if kind == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(stream.getvalue()).decode()


def review_crop(result: dict, probs: np.ndarray, size: tuple[int, int]) -> list[int]:
    width, height = size
    if result["endpoints_xy"] is not None:
        center = np.mean(result["endpoints_xy"], axis=0)
    else:
        # Strong local components beat a large sum of low background responses.
        field = probs[1]
        threshold = max(0.15, float(field.max()) * 0.65)
        count, labels, _, centroids = cv2.connectedComponentsWithStats((field > threshold).astype("uint8"))
        if count > 1:
            winner = max(range(1, count), key=lambda i: float(field[labels == i].sum()))
            center = (centroids[winner] + 0.5) * [width / field.shape[1], height / field.shape[0]]
        else:
            foreground = probs[2:].max(axis=0)
            y, x = np.unravel_index(foreground.argmax(), foreground.shape)
            center = np.array([(x + 0.5) * width / foreground.shape[1],
                               (y + 0.5) * height / foreground.shape[0]])
    side = min(900, width, height)
    x, y = np.clip(center - side / 2, [0, 0], [width - side, height - side]).astype(int)
    return [int(x), int(y), side, side]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    inputs = json.loads((root / "manifest.json").read_text())
    model = json.loads((root / "cache/model.json").read_text())
    records, cards, embedded = [], [], []
    assets = root / "overlays"
    assets.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for row in inputs:
        folder = root / "cache" / row["id"]
        meta = json.loads((folder / "input.json").read_text())
        path = Path(meta["path"])
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"Source changed after inference: {path}")
        image = Image.open(path).convert("RGB")
        probs = np.load(folder / "probabilities.npz", allow_pickle=False)["probabilities"]
        result = extract_axis(np.array(image), probs)
        (folder / "geometry.json").write_text(json.dumps(result, indent=2))
        crop = review_crop(result, probs, image.size)
        coordinate_path = path.with_name("coordinates.json")
        capture = json.loads(coordinate_path.read_text()) if coordinate_path.is_file() else None
        record = {**meta, **result, "review_crop_xywh": crop,
                  "coordinates_path": str(coordinate_path) if capture else None,
                  "coordinates_sha256": sha256(coordinate_path) if capture else None,
                  "capture_metadata": capture, "validation": "development sample; no measured angle truth"}
        records.append(record)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        if result["angle_deg"] is not None:
            for segment in result["candidates"][0]["visible_segments_xy"]:
                draw.line([tuple(p) for p in segment], fill=(0, 255, 220), width=4)
            for p in result["endpoints_xy"]:
                x, y = p
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 200, 50), width=3)
        overlay.save(assets / f"{row['id']}.png")
        x, y, w, h = crop
        card = Image.new("RGB", (500, 565), "#101b29")
        card.paste(overlay.crop((x, y, x + w, y + h)).resize((500, 500)), (0, 65))
        label = "No reliable angle" if result["angle_deg"] is None else f"{result['angle_deg']:.1f} degrees | needs review"
        d = ImageDraw.Draw(card)
        d.text((12, 9), row["id"], font=font, fill="white")
        d.text((12, 34), label, font=font, fill="#ffcb6b")
        cards.append(card)
        pin_image = Image.fromarray(np.round(probs[1] * 255).astype("uint8"))
        # All visual assets are embedded: file:// works without a web server.
        embedded.append({**{k: v for k, v in record.items() if k != "capture_metadata"},
                         "image_data": data_url(image, "JPEG"), "pin_data": data_url(pin_image)})
    examples = [r for r in records if r.get("role") != "training_reference_not_evaluation"]
    summary = {"images": len(examples),
               "candidates": sum(r["angle_deg"] is not None for r in examples),
               "no_reliable_axis": sum(r["angle_deg"] is None for r in examples),
               "accuracy_measured": False, "axis_3d_implemented": False,
               "model": model,
               "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    (root / "results.json").write_text(json.dumps({"summary": summary, "images": records}, indent=2))
    with (root / "results.csv").open("w", newline="") as stream:
        fields = ["id", "path", "status", "angle_deg", "sha256", "coordinates_path"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    contact = Image.new("RGB", (2000, 565 * ((len(cards) + 3) // 4)), "#101b29")
    for i, card in enumerate(cards):
        contact.paste(card, ((i % 4) * 500, (i // 4) * 565))
    contact.save(root / "contact-sheet.jpg", quality=94)
    template = Path(__file__).with_name("review.html").read_text()
    payload = json.dumps({"summary": summary, "images": embedded}).replace("<", "\\u003c")
    (root / "index.html").write_text(template.replace("__REVIEW_DATA__", payload))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"model", "source_sha256"}}, indent=2))
    print(root / "index.html")


if __name__ == "__main__":
    main()

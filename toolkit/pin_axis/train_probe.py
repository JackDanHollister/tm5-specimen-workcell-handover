"""Fit a new binary DINOv3 patch probe from checked sparse native-pixel marks.

The backbone is frozen. Unmarked pixels are unknown. Whole specimens must stay
in one split. Test examples are excluded from feature extraction and selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from annotations import annotation_masks, crop_boxes, load_checked, sparse_patch_labels
from inference import DEFAULT_MODEL, SavedDino, sha256


def fit_probe(datasets: dict, iterations: int = 300, seed: int = 17) -> tuple[dict, list]:
    torch.manual_seed(seed)
    head = torch.nn.Linear(datasets["train"][0].shape[1], 2).to(datasets["train"][0].device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    history, best, best_state = [], float("inf"), None
    for iteration in range(iterations):
        head.train()
        x, y, weights = datasets["train"]
        optimizer.zero_grad()
        loss = (F.cross_entropy(head(x), y, reduction="none") * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        if iteration % 10 == 0 or iteration == iterations - 1:
            head.eval()
            with torch.no_grad():
                vx, vy, vw = datasets["validation"]
                logits = head(vx)
                val_loss = float((F.cross_entropy(logits, vy, reduction="none") * vw).sum() / vw.sum())
                recalls = [float((logits.argmax(1)[vy == c] == c).float().mean()) for c in (0, 1)]
            history.append({"iteration": iteration + 1, "train_loss": float(loss.detach()),
                            "validation_loss": val_loss, "sparse_patch_balanced_accuracy": sum(recalls) / 2})
            if val_loss < best:
                best = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    return best_state, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    rows = load_checked(args.annotations)  # Validate labels BEFORE loading a GPU model.
    args.output.mkdir(parents=True, exist_ok=False)
    model = SavedDino(args.model_dir, args.device)
    collected, provenance = {"train": [], "validation": []}, []
    for row in rows:
        if row["split"] == "test":
            continue
        image = Image.open(row["source_path"]).convert("RGB")
        masks = annotation_masks(row, image.size)
        for box in crop_boxes(masks):
            x0, y0, x1, y1 = box
            features, info = model.encode(image.crop(box))
            labels = sparse_patch_labels(masks[:, y0:y1, x0:x1], tuple(features.shape[-2:]))
            known = labels.ravel() >= 0
            if not known.any():
                continue
            x = features[0].permute(1, 2, 0).reshape(-1, features.shape[1]).float().cpu().numpy()[known]
            y = labels.ravel()[known]
            collected[row["split"]].append((row["id"], x, y))
            provenance.append({"id": row["id"], "source_sha256": row["source_sha256"],
                               "specimen_id": row["specimen_id"], "split": row["split"],
                               "crop_xyxy": box, "labelled_patches": int(known.sum()), **info})
    # Feature extraction is finished; release the large frozen backbone.
    model_info = model.provenance
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    datasets = {}
    for split, chunks in collected.items():
        if not chunks:
            raise ValueError(f"No labelled patches in {split}")
        xs, ys, ws = [], [], []
        # Equal image/class contributions stop frames with many marked pixels dominating.
        for image_id in sorted({c[0] for c in chunks}):
            parts = [c for c in chunks if c[0] == image_id]
            x, y = np.concatenate([c[1] for c in parts]), np.concatenate([c[2] for c in parts])
            weights = np.zeros(len(y), dtype=np.float32)
            for label in (0, 1):
                chosen = y == label
                if chosen.any():
                    weights[chosen] = 1 / chosen.sum()
            xs.append(x); ys.append(y); ws.append(weights)
        x, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        if set(y.tolist()) != {0, 1}:
            raise ValueError(f"{split} needs both marked pin and explicitly marked non-pin examples")
        datasets[split] = tuple(torch.from_numpy(a).to(args.device) for a in (x, y, weights))
    state, history = fit_probe(datasets, args.iterations)
    checkpoint = {"schema": "tm5_dinov3_binary_probe_v1", "state_dict": state,
                  "classes": ["marked_non_pin", "patch_contains_marked_pin_shaft"],
                  "feature_dim": 1280, "backbone_sha256": model_info["backbone_sha256"]}
    torch.save(checkpoint, args.output / "pin_probe.pt")
    (args.output / "training.json").write_text(json.dumps({
        "annotations_sha256": sha256(args.annotations), "model": model_info,
        "features": provenance, "history": history, "seed": 17,
        "held_out_test_images_used": 0, "ground_truth_3d": False,
        "metrics_scope": "Checked sparse image patches only; not shaft-angle or whole-image accuracy",
        "label_rule": "More than 2 percent marked pixel area; overlapping positive/negative marks ignored",
    }, indent=2))
    print(args.output / "pin_probe.pt")


if __name__ == "__main__":
    main()

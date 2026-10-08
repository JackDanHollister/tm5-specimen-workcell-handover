"""Offline contracts: axis convention, rejection, annotation leakage and probe fit."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from annotations import load_checked, sparse_patch_labels
from geometry import angle_difference, extract_axis
from inference import sha256
from train_probe import fit_probe


def scene(angle, second=False, occlusion=False):
    rgb = np.full((512, 512, 3), 170, dtype=np.uint8)
    pin = np.zeros((512, 512), dtype=np.float32)
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    normal = np.array([-direction[1], direction[0]])
    for offset in ([-65, 65] if second else [0]):
        a, b = (np.array([256, 256]) + normal * offset + np.array([-155, 155])[:, None] * direction).astype(int)
        cv2.line(rgb, tuple(a), tuple(b), (30, 30, 30), 7, cv2.LINE_AA)
        cv2.line(pin, tuple(a), tuple(b), 0.95, 18)
    if occlusion:
        cv2.rectangle(rgb, (0, 220), (511, 280), (230, 230, 230), -1)
        pin[220:281] = 0
    probs = np.zeros((5, 32, 32), dtype=np.float32)
    probs[1] = cv2.resize(pin, (32, 32), interpolation=cv2.INTER_AREA)
    probs[0] = 1 - probs[1]
    return rgb, probs


class GeometryTests(unittest.TestCase):
    def test_angles_and_coordinate_convention(self):
        for angle in (0, 28, 90, 131, 178):
            with self.subTest(angle=angle):
                result = extract_axis(*scene(angle))
                self.assertIsNotNone(result["angle_deg"], result["reasons"])
                self.assertLess(angle_difference(result["angle_deg"], angle), 1.5)
                self.assertIsNone(result["axis_3d"])

    def test_hidden_gap_does_not_count_as_visible_length(self):
        result = extract_axis(*scene(90, occlusion=True))
        self.assertIsNotNone(result["angle_deg"])
        best = result["candidates"][0]
        self.assertGreater(best["span_px"] - best["visible_length_px"], 40)

    def test_two_equal_pins_require_identity_review(self):
        result = extract_axis(*scene(90, second=True))
        self.assertIsNone(result["angle_deg"])
        self.assertTrue(any("Competing" in reason for reason in result["reasons"]))

    def test_blank_and_broad_dark_object_are_not_shafts(self):
        rgb, probs = scene(90)
        rgb[:] = 170
        self.assertIsNone(extract_axis(rgb, probs)["angle_deg"])
        cv2.rectangle(rgb, (170, 150), (330, 370), (30, 30, 30), -1)
        probs[1] = 0.9; probs[0] = 0.1
        self.assertIsNone(extract_axis(rgb, probs)["angle_deg"])


class AnnotationTests(unittest.TestCase):
    def test_unmarked_and_conflicting_patches_stay_unknown(self):
        masks = np.zeros((2, 64, 64), dtype=np.float32)
        masks[0, 0:16, 0:16] = 1
        masks[1, 0:16, 16:32] = 1
        masks[:, 16:32, 0:16] = 1
        labels = sparse_patch_labels(masks, (4, 4))
        self.assertEqual(labels[0, 0], 1)
        self.assertEqual(labels[0, 1], 0)
        self.assertEqual(labels[1, 0], -1)
        self.assertEqual(labels[3, 3], -1)

    def test_specimen_split_leakage_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, split in enumerate(("train", "validation")):
                image = root / f"{i}.png"
                Image.new("RGB", (64, 64), (i, i, i)).save(image)
                rows.append({"id": str(i), "source_path": str(image), "source_sha256": sha256(image),
                             "reviewed": True, "specimen_id": "same-specimen", "placement_id": str(i),
                             "split": split, "visibility": "visible", "segments": [
                                 {"kind": "pin", "points": [[20, 20], [20, 40]], "width_px": 5}]})
            path = root / "annotations.json"
            path.write_text(json.dumps({"schema": "tm5_pin_annotations_v1", "images": rows}))
            with self.assertRaisesRegex(ValueError, "Specimen leakage"):
                load_checked(path)

    def test_empty_labels_cannot_launch_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps({"schema": "tm5_pin_annotations_v1", "images": []}))
            with self.assertRaisesRegex(ValueError, "No checked"):
                load_checked(path)

    def test_small_probe_learns_separable_features(self):
        generator = torch.Generator().manual_seed(123)
        datasets = {}
        for split in ("train", "validation"):
            y = torch.arange(60) % 2
            x = torch.randn(60, 8, generator=generator) * 0.1
            x[:, 0] += y.float() * 2 - 1
            datasets[split] = x, y, torch.ones(60)
        state, history = fit_probe(datasets, iterations=150)
        self.assertEqual(tuple(state["weight"].shape), (2, 8))
        self.assertGreater(history[-1]["sparse_patch_balanced_accuracy"], 0.95)
        self.assertLess(history[-1]["validation_loss"], history[0]["validation_loss"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

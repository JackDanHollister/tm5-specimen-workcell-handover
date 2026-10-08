"""Local saved-image DINOv3 inference. No robot, ROS, camera or network client."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

CLASS_NAMES = ("Background", "Pin", "Specimen", "Mount", "Label")
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'models/dino'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SavedDino:
    def __init__(self, model_dir: Path = DEFAULT_MODEL, device: str = "cuda"):
        self.device = torch.device(device)
        self.model_dir = model_dir.resolve()
        repo = self.model_dir / "dinov3"
        weights = self.model_dir / "checkpoints/dinov3_vith16plus_pretrain_lvd1689m.pth"
        classifier = self.model_dir / "classifier.pt"
        for path in (repo / "hubconf.py", weights, classifier):
            if not path.is_file():
                raise FileNotFoundError(path)
        state = torch.load(classifier, map_location="cpu", weights_only=True)
        if tuple(state["weight"].shape) != (5, 1280, 1, 1):
            raise ValueError("Expected the saved five-class ViT-H+/16 probe")
        self.head = torch.nn.Conv2d(1280, 5, 1)
        self.head.load_state_dict(state)
        self.head.eval().to(self.device)
        # Both source and weights are local; no implicit model download.
        self.backbone = torch.hub.load(str(repo), "dinov3_vith16plus", source="local",
                                       weights=str(weights)).eval().to(self.device)
        self.provenance = {
            "architecture": "dinov3_vith16plus", "classes": list(CLASS_NAMES),
            "classifier_path": str(classifier), "classifier_sha256": sha256(classifier),
            "backbone_path": str(weights), "backbone_sha256": sha256(weights),
            "device": str(self.device), "torch": torch.__version__,
            "preprocessing": "RGB; ImageNet normalization; nearest multiple of 16 resize",
            "probabilities_calibrated": False, "training_performed": False,
        }

    @torch.inference_mode()
    def encode(self, image: Image.Image, max_side: int = 0) -> tuple[torch.Tensor, dict]:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
        nw, nh = [16 * max(1, round(v * scale / 16)) for v in (width, height)]
        resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
        array = np.array(resized, dtype=np.float32) / 255
        x = torch.from_numpy(array).permute(2, 0, 1)[None].to(self.device)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        start = time.monotonic()
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            features = self.backbone.get_intermediate_layers(
                (x - mean) / std, n=1, reshape=True, norm=True)[0]
        return features, {"source_size": [width, height], "model_size": [nw, nh],
                          "inference_seconds": time.monotonic() - start}

    @torch.inference_mode()
    def predict(self, image: Image.Image, max_side: int = 0) -> tuple[np.ndarray, dict]:
        features, info = self.encode(image, max_side)
        probabilities = F.softmax(self.head(features.float()), dim=1)[0].cpu().numpy()
        return probabilities, info

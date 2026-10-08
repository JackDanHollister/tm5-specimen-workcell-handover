"""Cache inference once; geometry and report can be revised without rerunning DINO."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from inference import DEFAULT_MODEL, SavedDino, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON list of {id, path} saved images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-side", type=int, default=0, help="0 retains native resolution")
    args = parser.parse_args()
    inputs = json.loads(args.manifest.read_text())
    ids = [row["id"] for row in inputs]
    if len(set(ids)) != len(ids) or any(not s or set(s) - set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") for s in ids):
        raise ValueError("IDs must be unique and use letters, digits, underscores or hyphens")
    # A fresh run is intentional: never silently combine stale model/image caches.
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "pid": os.getpid(), "completed": [],
             "started_utc": datetime.now(timezone.utc).isoformat()}
    state_path = args.output / "state.json"
    state_path.write_text(json.dumps(state, indent=2))
    try:
        print("Loading local ViT-H+/16 and saved classifier", flush=True)
        model = SavedDino(args.model_dir, args.device)
        (args.output / "model.json").write_text(json.dumps(model.provenance, indent=2))
        for row in inputs:
            path = Path(row["path"]).resolve(strict=True)
            target = args.output / row["id"]
            target.mkdir()
            with Image.open(path) as image:
                probs, info = model.predict(image, args.max_side)
            np.savez_compressed(target / "probabilities.npz", probabilities=probs)
            record = {**row, "path": str(path), "sha256": sha256(path), **info}
            (target / "input.json").write_text(json.dumps(record, indent=2))
            state["completed"].append(row["id"])
            state_path.write_text(json.dumps(state, indent=2))
            print(row["id"], info, "peak pin", float(probs[1].max()), flush=True)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

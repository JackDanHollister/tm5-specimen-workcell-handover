"""Export only annotated endpoint-review artifacts; never original captures."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(trial, output):
    trial, output = trial.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Review export must use a fresh directory")
    result = json.loads((trial / "results.json").read_text())
    source = Path(result["source"])
    if output == source or source in output.parents or trial in output.parents:
        raise ValueError("Do not export into frozen evidence")
    # Reverify inputs before publication; retain hashes but not local paths.
    for name, expected in result["source_hashes"].items():
        assert sha256(source / name) == expected, name
    for view in result["views"]:
        assert sha256(Path(view["path"])) == view["source_sha256"], view["id"]
        assert sha256(source / "cache" / view["id"] / "probabilities.npz") == view["probabilities_sha256"], view["id"]
    assert sha256(Path(__file__).with_name("endpoint_trial.py")) == result["source_code_sha256"]
    output.mkdir(parents=True)
    (output / "views").mkdir()
    for view in result["views"]:
        relative = Path("views") / f"{view['id']}.jpg"
        with Image.open(trial / relative) as image:
            image.verify()
        shutil.copyfile(trial / relative, output / relative)
        view.pop("path")
    page = (trial / "index.html").read_text()
    for group in result["specimens"]:
        name = f"pin-{group['specimen_id']}-review"
        with Image.open(trial / f"{name}.png") as image:
            image.convert("RGB").save(output / f"{name}.jpg", quality=92, optimize=True)
        page = page.replace(f"{name}.png", f"{name}.jpg")
    (output / "index.html").write_text(page)
    result.pop("source")
    result["review_version"] = "2026-09-24-v4"
    result["trial_results_sha256"] = sha256(trial / "results.json")
    result["report_code_sha256"] = sha256(Path(__file__).with_name("endpoint_report.py"))
    result["export_code_sha256"] = sha256(Path(__file__))
    result["input_reverification"] = "all source images, probability caches and root manifests matched"
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files))
    print(f"Exported {len(result['views'])} annotated views, {len(result['specimens'])} comparison sheets")
    for group in result["specimens"]:
        print(group["specimen_id"], "head", len(group["head"]["support_ids"]), "/", group["eligible_views"],
              "native span", len(group["native_span"]["support_ids"]), "/", group["eligible_views"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.trial, args.output)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def resolve(relative):
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError('Package path escapes repository')
    if not path.exists():
        raise FileNotFoundError(f'{relative} is missing; run scripts/fetch_assets.py')
    return path


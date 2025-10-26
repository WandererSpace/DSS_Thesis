from __future__ import annotations
from pathlib import Path
import json
import yaml

def find_project_root(start: Path | None = None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "config.yaml").exists() or (p / "config.json").exists():
            return p
    return start  # fallback

def load_config(root: Path | None = None):
    root = Path(root) if root else find_project_root()
    yml = root / "config.yaml"
    jsn = root / "config.json"
    if yml.exists():
        cfg = yaml.safe_load(yml.read_text(encoding="utf-8"))
    elif jsn.exists():
        cfg = json.loads(jsn.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("No config.yaml or config.json at project root.")
    cfg["_root"] = str(root)
    return cfg, root

def abspath(root: Path, rel: str) -> Path:
    return (root / rel).resolve()

def ensure_dirs(*paths: Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

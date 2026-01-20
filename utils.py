import yaml
import json
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r", encoding = "utf-8") as f:
        obj = yaml.safe_load(f)
    return obj

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding = "utf-8") as f:
        obj = json.load(f)
    return obj["data"]


def project_root() -> Path:
    return Path(__file__).resolve().parent

def resolve_dataset_path(user_path: str) -> str:
    root_path = project_root()
    p = Path(user_path)
    if not p.is_absolute():
        p = root_path / p
    if p.is_dir():
        raise ValueError(f"json_path points to a directory, expected a JSON file: {p}")
    if not p.exists():
        raise FileNotFoundError(
            "Dataset JSON not found.\n"
            f"- resolved_path: {p}\n"
            "- tip: pass an explicit path via '--json-path ...' or set env JSON_PATH\n"
            f"- tip: your project root is: {root_path}\n"
        )
    return str(p)


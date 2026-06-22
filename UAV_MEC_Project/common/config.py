from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

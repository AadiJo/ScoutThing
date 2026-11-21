"""Pinned model weights and field data, listed in `assets.lock.json` at the repo root.

Stages call `asset("bumpers")` to get the local path of a file. `scout fetch-assets`
fills `assets/` (download or `--from-local` copy) and checks every sha256. Files keep
their upstream path under `assets/`.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
LOCK_PATH = ROOT / "assets.lock.json"

AssetName = Literal[
    "digits",
    "embedder",
    "bumpers",
    "bumpers_offpalette",
    "hub_zone",
    "field_tags",
    "field_wireframe",
    "field2d_image",
    "field2d_meta",
]


class LockedFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    size: int
    lfs: bool


class Lock(BaseModel):
    """`assets.lock.json`: every file pinned to one upstream commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    commit: str
    files: dict[AssetName, LockedFile]

    def url(self, f: LockedFile) -> str:
        host = "media.githubusercontent.com/media" if f.lfs else "raw.githubusercontent.com"
        return f"https://{host}/{self.repo}/{self.commit}/{f.path}"


def load_lock() -> Lock:
    lock = Lock.model_validate_json(LOCK_PATH.read_bytes())
    missing = set(get_args(AssetName)) - set(lock.files)
    if missing:
        raise ValueError(f"{LOCK_PATH} lacks assets {sorted(missing)}")
    return lock


def asset(name: AssetName) -> Path:
    """Local path of a pinned asset. Raises FileNotFoundError if it was never fetched."""
    path = ASSETS_DIR / load_lock().files[name].path
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `scout fetch-assets`")
    return path


def sha256_file(path: Path) -> str:
    """Streamed sha256 of a file, hex."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def asset_hashes() -> dict[str, str | None]:
    """sha256 of every asset on disk (None when missing), for the run manifest."""
    out: dict[str, str | None] = {}
    for name, f in load_lock().files.items():
        path = ASSETS_DIR / f.path
        out[name] = sha256_file(path) if path.is_file() else None
    return out


def fetch_assets(from_local: Path | None = None) -> None:
    """Make `assets/` match the lock: skip files whose hash already matches, otherwise
    copy from `from_local/<path>` or download, then verify. Raises on a hash mismatch."""
    lock = load_lock()
    for name, f in lock.files.items():
        dest = ASSETS_DIR / f.path
        if dest.is_file() and sha256_file(dest) == f.sha256:
            print(f"ok        {name}  {f.path}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        if from_local is not None:
            shutil.copyfile(from_local / f.path, tmp)
            how = "copied"
        else:
            with urllib.request.urlopen(lock.url(f)) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            how = "fetched"
        got = sha256_file(tmp)
        if got != f.sha256:
            tmp.unlink()
            raise ValueError(f"{name}: sha256 {got} does not match lock {f.sha256}")
        tmp.replace(dest)
        print(f"{how:<9} {name}  {f.path}")

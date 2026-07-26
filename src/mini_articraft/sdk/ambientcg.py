"""Fetch and cache ambientCG PBR material sets (offline after first pull).

ambientCG materials are CC0, so the maps can be bundled straight into an
exported USDZ. ambientCG ships one zip per resolution holding the whole map
set, so this adapter downloads the zip once, extracts it into the cache, and
locates the channels (Color / Roughness / NormalGL / Metalness) by filename
suffix, returning a :class:`TextureSet` the rest of the texture pipeline reads.

Download scheme (no API key)::

    https://ambientcg.com/get?file=<asset>_<res>-JPG.zip
      -> <asset>_<res>-JPG_Color.jpg / _Roughness.jpg / _NormalGL.jpg / _Metalness.jpg
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TextureSet:
    """Local paths to a fetched PBR texture set.

    Only ``base_color`` is guaranteed; ``roughness``/``normal``/``metalness``
    are present when the material provides them.
    """

    slug: str
    resolution: str
    base_color: Path
    roughness: Path | None = None
    normal: Path | None = None
    metalness: Path | None = None

    def maps(self) -> dict[str, Path]:
        found = {"base_color": self.base_color}
        for channel in ("roughness", "normal", "metalness"):
            path = getattr(self, channel)
            if path is not None:
                found[channel] = path
        return found


_URL = "https://ambientcg.com/get?file={asset}_{res}-JPG.zip"
_CACHE_ROOT = Path.home() / ".cache" / "mini_articraft" / "ambientcg"

# ambientCG filename suffix -> TextureSet channel.
_CHANNELS = {
    "_Color.": "base_color",
    "_Roughness.": "roughness",
    "_NormalGL.": "normal",
    "_Metalness.": "metalness",
}


def fetch_texture_set(
    asset: str,
    *,
    resolution: str = "1K",
    cache_root: Path | None = None,
) -> TextureSet:
    """Download (or reuse cached) an ambientCG material set for ``asset``.

    Raises ``RuntimeError`` if the zip cannot be fetched or has no color map, so
    callers can fall back to a parametric ``Material``.
    """

    cache = (cache_root or _CACHE_ROOT) / asset / resolution
    cache.mkdir(parents=True, exist_ok=True)

    found = _locate(cache)
    if "base_color" not in found:
        _download_and_extract(_URL.format(asset=asset, res=resolution), cache)
        found = _locate(cache)
    if "base_color" not in found:
        raise RuntimeError(f"ambientCG asset {asset!r} has no color map")

    return TextureSet(
        slug=asset,
        resolution=resolution,
        base_color=found["base_color"],
        roughness=found.get("roughness"),
        normal=found.get("normal"),
        metalness=found.get("metalness"),
    )


def _locate(cache: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in cache.iterdir():
        for suffix, channel in _CHANNELS.items():
            if suffix in path.name:
                found[channel] = path
    return found


def _download_and_extract(url: str, cache: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    zip_path = cache / "_download.zip"
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            zip_path.write_bytes(response.read())
    except Exception as exc:
        raise RuntimeError(f"could not download ambientCG asset: {url}") from exc
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if any(suffix in member for suffix in _CHANNELS):
                    archive.extract(member, cache)
    finally:
        zip_path.unlink(missing_ok=True)

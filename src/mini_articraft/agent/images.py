from __future__ import annotations

import base64
import hashlib
import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
PATCH_SIZE = 32
ImageDetail = Literal["high", "original"]


@dataclass(frozen=True)
class ImageLimits:
    max_dimension: int
    max_patches: int


LIMITS = {
    "high": ImageLimits(max_dimension=2_048, max_patches=2_500),
    "original": ImageLimits(max_dimension=6_000, max_patches=10_000),
}
FORMAT_MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}
MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    detail: ImageDetail

    @property
    def suffix(self) -> str:
        return MIME_SUFFIXES[self.mime_type]

    def content_item(self) -> dict[str, str]:
        data = base64.b64encode(self.data).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{self.mime_type};base64,{data}",
            "detail": self.detail,
        }

    def metadata(self, path: str) -> dict[str, object]:
        return {
            "path": path,
            "mime_type": self.mime_type,
            "bytes": len(self.data),
            "width": self.width,
            "height": self.height,
            "detail": self.detail,
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }


def prepare_image(path: Path, *, detail: ImageDetail = "original") -> PreparedImage:
    if not path.is_file():
        raise ValueError(f"image path is not a file: {path}")
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes: {path}")
    return _prepare_image(path.read_bytes(), detail=detail)


def _prepare_image(data: bytes, *, detail: ImageDetail) -> PreparedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                source_format = str(source.format or "").upper()
                if bool(getattr(source, "is_animated", False)):
                    raise ValueError("animated GIF images are not supported")
                source.load()
                orientation = source.getexif().get(274)
                image = ImageOps.exif_transpose(source).copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("image dimensions exceed the supported limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if str(exc) == "animated GIF images are not supported":
            raise
        raise ValueError("file is not a supported PNG, JPEG, GIF, or WebP image") from exc

    mime_type = FORMAT_MIME_TYPES.get(source_format)
    if mime_type is None:
        raise ValueError("image must be PNG, JPEG, GIF, or WebP")

    width, height = _output_dimensions(image.width, image.height, LIMITS[detail])
    if (
        source_format in {"PNG", "JPEG", "WEBP"}
        and (width, height) == image.size
        and orientation in (None, 1)
    ):
        return PreparedImage(data, mime_type, width, height, detail)

    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    output_format = "PNG" if source_format == "GIF" else source_format
    output = io.BytesIO()
    if output_format == "JPEG":
        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    elif output_format == "WEBP":
        image.convert("RGBA").save(
            output,
            format="WEBP",
            lossless=True,
            quality=100,
            method=0,
        )
    else:
        image.convert("RGBA").save(
            output,
            format="PNG",
            compress_level=9,
            optimize=False,
        )
    encoded = output.getvalue()
    return PreparedImage(
        encoded,
        FORMAT_MIME_TYPES[output_format],
        width,
        height,
        detail,
    )


def _output_dimensions(width: int, height: int, limits: ImageLimits) -> tuple[int, int]:
    width = max(1, width)
    height = max(1, height)
    if _fits(width, height, limits):
        return width, height

    scale = min(limits.max_dimension / max(width, height), 1.0)
    width = max(1, round(width * scale))
    height = max(1, round(height * scale))
    if _fits(width, height, limits):
        return width, height

    scale = math.sqrt(PATCH_SIZE**2 * limits.max_patches / width / height)
    scaled_patches_wide = width * scale / PATCH_SIZE
    scaled_patches_high = height * scale / PATCH_SIZE
    scale *= min(
        math.floor(scaled_patches_wide) / scaled_patches_wide,
        math.floor(scaled_patches_high) / scaled_patches_high,
    )
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _fits(width: int, height: int, limits: ImageLimits) -> bool:
    patches = math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE)
    return (
        width <= limits.max_dimension
        and height <= limits.max_dimension
        and patches <= limits.max_patches
    )

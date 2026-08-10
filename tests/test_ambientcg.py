from __future__ import annotations

import io
import zipfile

from articraft.sdk import ambientcg


def _archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("nested/Metal009_1K-JPG_Color.jpg", b"color")
        archive.writestr("nested/Metal009_1K-JPG_Roughness.jpg", b"roughness")
        archive.writestr("nested/unrelated.txt", b"ignored")
    return output.getvalue()


def test_fetch_texture_set_downloads_once_and_reuses_cache(monkeypatch, tmp_path) -> None:
    downloads = 0

    def urlopen(_request, *, timeout):
        nonlocal downloads
        downloads += 1
        assert timeout == 30
        return io.BytesIO(_archive())

    monkeypatch.setattr(ambientcg.urllib.request, "urlopen", urlopen)

    first = ambientcg.fetch_texture_set("Metal009", cache_root=tmp_path)
    second = ambientcg.fetch_texture_set("Metal009", cache_root=tmp_path)

    assert downloads == 1
    assert first == second
    assert first.base_color.read_bytes() == b"color"
    assert first.roughness is not None
    assert first.roughness.read_bytes() == b"roughness"
    assert not list(tmp_path.rglob("*.zip"))

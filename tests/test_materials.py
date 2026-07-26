from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pxr import Usd, UsdGeom, UsdShade  # pyright: ignore[reportAttributeAccessIssue]

from mini_articraft.sdk import (
    ArticulatedObject,
    BoxGeometry,
    Material,
    MotionLimits,
)
from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.export import export_object
from mini_articraft.viewer import _read_version


def _model() -> ArticulatedObject:
    model = ArticulatedObject("materialed")
    base = model.part("base")
    base.add(
        BoxGeometry([0.2, 0.2, 0.1]),
        name="body",
        material=Material.metal((0.8, 0.5, 0.2, 1.0), roughness=0.3),
    )
    lid = model.part("lid")
    lid.add(BoxGeometry([0.2, 0.2, 0.02]), name="cap", color=(0.2, 0.3, 0.8))
    model.articulation(
        "hinge",
        "revolute",
        base,
        lid,
        motion_limits=MotionLimits(lower=0.0, upper=1.5),
    )
    return model


def test_material_validates_ranges_and_presets() -> None:
    metal = Material.metal((0.8, 0.8, 0.85, 1.0))
    assert metal.metallic == 1.0
    assert metal.opacity == 1.0
    glass = Material.glass()
    assert glass.metallic == 0.0
    assert glass.opacity < 1.0

    with pytest.raises(ValidationError, match="metallic"):
        Material(base_color=(0.5, 0.5, 0.5, 1.0), metallic=1.5)
    with pytest.raises(ValidationError, match="roughness"):
        Material(base_color=(0.5, 0.5, 0.5, 1.0), roughness=-0.1)
    with pytest.raises(ValidationError, match="base_color"):
        Material(base_color=(2.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="3 numeric values"):
        Material(emissive=(0.1, 0.2, 0.3, 0.4))  # type: ignore[arg-type]


def test_color_shorthand_becomes_a_matte_dielectric() -> None:
    part = ArticulatedObject("o").part("p")
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="s", color=(0.2, 0.3, 0.8))
    shape = next(part._iter_shapes())
    assert shape.material == Material(base_color=(0.2, 0.3, 0.8, 1.0))
    assert shape.color == (0.2, 0.3, 0.8, 1.0)


def test_color_and_material_are_mutually_exclusive() -> None:
    part = ArticulatedObject("o").part("p")
    with pytest.raises(ValidationError, match="both color and material"):
        part.add(
            BoxGeometry([0.1, 0.1, 0.1]),
            name="s",
            color=(0.2, 0.3, 0.8),
            material=Material.metal(),
        )


def test_export_binds_usd_preview_surface(tmp_path) -> None:
    result = export_object(_model(), tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))

    mesh = stage.GetPrimAtPath("/World/materialed/parts/base/shapes/body")
    assert mesh.IsA(UsdGeom.Mesh)
    binding = UsdShade.MaterialBindingAPI(mesh).GetDirectBinding()
    material = UsdShade.Material(stage.GetPrimAtPath(binding.GetMaterialPath()))
    assert material

    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{binding.GetMaterialPath()}/surface"))
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("metallic").Get() == pytest.approx(1.0)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.3)
    diffuse = shader.GetInput("diffuseColor").Get()
    assert tuple(round(float(value), 3) for value in diffuse) == (0.8, 0.5, 0.2)

    # displayColor fallback is preserved for renderers that ignore UsdShade.
    assert mesh.GetAttribute("primvars:displayColor").Get() is not None


def test_export_payload_carries_material(tmp_path) -> None:
    result = export_object(_model(), tmp_path)
    manifest = json.loads(result.manifest.read_text())
    body = manifest["parts"][0]["shapes"][0]
    assert body["material"] == {
        "base_color": [0.8, 0.5, 0.2, 1.0],
        "metallic": 1.0,
        "roughness": 0.3,
        "emissive": None,
    }


def test_viewer_readback_exposes_shape_materials(tmp_path) -> None:
    result = export_object(_model(), tmp_path)
    version = _read_version(result.usdz)
    model = cast(dict[str, Any], version["model"])
    parts = {part["name"]: part for part in cast(list[dict[str, Any]], model["parts"])}

    body = parts["base"]["shapes"][0]
    assert body["usd_name"] == "body"
    assert body["material"]["metallic"] == pytest.approx(1.0)
    assert body["material"]["roughness"] == pytest.approx(0.3)
    assert body["material"]["base_color"] == pytest.approx([0.8, 0.5, 0.2])

    cap = parts["lid"]["shapes"][0]
    assert cap["material"]["metallic"] == pytest.approx(0.0)

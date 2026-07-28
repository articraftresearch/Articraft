from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pxr import Usd, UsdGeom, UsdShade  # pyright: ignore[reportAttributeAccessIssue]

from mini_articraft.sdk import (
    Appearance,
    ArticulatedObject,
    BoxGeometry,
    Material,
    MotionLimits,
    ambientcg,
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
        appearance=Appearance.metal((0.8, 0.5, 0.2, 1.0), roughness=0.3),
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
    metal = Appearance.metal((0.8, 0.8, 0.85, 1.0))
    assert metal.metallic == 1.0
    assert metal.opacity == 1.0
    glass = Appearance.glass()
    assert glass.metallic == 0.0
    assert glass.opacity < 1.0

    with pytest.raises(ValidationError, match="metallic"):
        Appearance(base_color=(0.5, 0.5, 0.5, 1.0), metallic=1.5)
    with pytest.raises(ValidationError, match="roughness"):
        Appearance(base_color=(0.5, 0.5, 0.5, 1.0), roughness=-0.1)
    with pytest.raises(ValidationError, match="base_color"):
        Appearance(base_color=(2.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="3 numeric values"):
        Appearance(emissive=(0.1, 0.2, 0.3, 0.4))  # type: ignore[arg-type]


def test_color_alone_becomes_a_matte_dielectric() -> None:
    part = ArticulatedObject("o").part("p")
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="s", color=(0.2, 0.3, 0.8))
    shape = next(part._iter_shapes())
    assert shape.resolved_appearance == Appearance(base_color=(0.2, 0.3, 0.8, 1.0))
    assert shape.color == (0.2, 0.3, 0.8, 1.0)


def test_a_material_supplies_its_own_look() -> None:
    part = ArticulatedObject("o").part("p")
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="s", material=Material.RUBBER)
    shape = next(part._iter_shapes())
    assert shape.appearance is None
    assert shape.resolved_appearance == Material.RUBBER.appearance


def test_color_recolors_a_material_without_changing_what_it_is() -> None:
    part = ArticulatedObject("o").part("p")
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="s", material=Material.STEEL, color=(0.2, 0.3, 0.8))
    shape = next(part._iter_shapes())

    assert shape.material is Material.STEEL
    assert shape.color == (0.2, 0.3, 0.8, 1.0)
    # Still reads as metal: recoloring keeps how the surface responds to light.
    assert shape.resolved_appearance is not None
    assert shape.resolved_appearance.metallic == Material.STEEL.appearance.metallic
    assert shape.resolved_appearance.roughness == Material.STEEL.appearance.roughness


def test_appearance_overrides_the_material_look_entirely() -> None:
    part = ArticulatedObject("o").part("p")
    part.add(
        BoxGeometry([0.1, 0.1, 0.1]),
        name="s",
        material=Material.ABS_PLASTIC,
        appearance=Appearance.metal(),
    )
    shape = next(part._iter_shapes())

    # Chrome-plated plastic: weighs like plastic, looks like metal.
    assert shape.material is Material.ABS_PLASTIC
    assert shape.resolved_appearance is not None
    assert shape.resolved_appearance.metallic == 1.0


def test_color_and_appearance_are_mutually_exclusive() -> None:
    part = ArticulatedObject("o").part("p")
    with pytest.raises(ValidationError, match="both color and appearance"):
        part.add(
            BoxGeometry([0.1, 0.1, 0.1]),
            name="s",
            color=(0.2, 0.3, 0.8),
            appearance=Appearance.metal(),
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


def test_export_payload_carries_material_and_appearance(tmp_path) -> None:
    model = ArticulatedObject("materialed")
    part = model.part("base")
    part.add(BoxGeometry([0.2, 0.2, 0.1]), name="body", material=Material.STEEL)

    result = export_object(model, tmp_path)
    manifest = json.loads(result.manifest.read_text())
    body = manifest["parts"][0]["shapes"][0]

    assert body["material"] == "steel"
    assert body["appearance"] == {
        "base_color": list(Material.STEEL.appearance.base_color),
        "metallic": 1.0,
        "roughness": Material.STEEL.appearance.roughness,
        "opacity": Material.STEEL.appearance.opacity,
        "emissive": None,
    }


def test_textured_export_resolves_each_explicit_kind_once(monkeypatch, tmp_path) -> None:
    model = ArticulatedObject("textures")
    part = model.part("part")
    # The material is the texture key, so two steel shapes share one fetch and a
    # shape with only a color asks for nothing.
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="warm_light", material=Material.STEEL)
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="showcase", material=Material.STEEL)
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="steel_by_name_only", color=(0.5, 0.5, 0.5))
    attempts = 0

    def fail_fetch(_kind):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("offline")

    monkeypatch.setattr("mini_articraft.sdk.ambientcg.fetch_material", fail_fetch)
    result = export_object(model, tmp_path, textured=True)

    assert attempts == 1
    assert result.textures.requested_shapes == 2
    assert result.textures.textured_shapes == 0
    assert len(result.textures.errors) == 1


def test_textured_export_applies_explicit_texture(monkeypatch, tmp_path) -> None:
    maps = tmp_path / "maps"
    maps.mkdir()
    color = maps / "Metal009_1K-JPG_Color.jpg"
    color.write_bytes(b"image")
    roughness_map = maps / "Metal009_1K-JPG_Roughness.jpg"
    roughness_map.write_bytes(b"image")
    texture_set = ambientcg.TextureSet("Metal009", "1K", color, roughness=roughness_map)
    spec = ambientcg.MaterialSpec("Metal009")
    monkeypatch.setattr(
        ambientcg,
        "fetch_material",
        lambda kind: (texture_set, spec),
    )
    model = ArticulatedObject("textured")
    model.part("part").add(
        BoxGeometry([0.1, 0.1, 0.1]),
        name="name_has_no_material_semantics",
        material=Material.STEEL,
    )

    result = export_object(model, tmp_path / "result", textured=True)
    stage = Usd.Stage.Open(str(result.usdz))
    mesh = stage.GetPrimAtPath("/World/textured/parts/part/shapes/name_has_no_material_semantics")

    assert result.textures.requested_shapes == 1
    assert result.textures.textured_shapes == 1
    assert result.textures.errors == ()
    assert mesh.GetAttribute("mini_articraft:material:textured").Get() == pytest.approx(1.0)
    points = mesh.GetAttribute("points").Get()
    uvs = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()  # pyright: ignore[reportAttributeAccessIssue]
    assert len(points) == len(uvs)
    assert all(0.0 <= float(component) <= 1.0 for uv in uvs for component in uv)

    binding = UsdShade.MaterialBindingAPI(mesh).GetDirectBinding()
    material = UsdShade.Material(stage.GetPrimAtPath(binding.GetMaterialPath()))
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material.GetPath()}/surface"))
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    diffuse_source = shader.GetInput("diffuseColor").GetConnectedSource()
    assert diffuse_source is not None
    assert UsdShade.Shader(diffuse_source[0].GetPrim()).GetIdAttr().Get() == "UsdUVTexture"
    diffuse = UsdShade.Shader(diffuse_source[0].GetPrim())
    # The texture is tinted by the material's own base color.
    assert tuple(diffuse.GetInput("scale").Get()) == pytest.approx(
        Material.STEEL.appearance.base_color
    )
    roughness_source = shader.GetInput("roughness").GetConnectedSource()
    assert roughness_source is not None
    roughness = UsdShade.Shader(roughness_source[0].GetPrim())
    assert tuple(roughness.GetInput("scale").Get()) == pytest.approx((0.35,) * 4)


def test_viewer_readback_exposes_shape_materials(tmp_path) -> None:
    result = export_object(_model(), tmp_path)
    version = _read_version(result.usdz)
    model = cast(dict[str, Any], version["model"])
    parts = {part["name"]: part for part in cast(list[dict[str, Any]], model["parts"])}

    body = parts["base"]["shapes"][0]
    assert body["usd_name"] == "body"
    assert body["appearance"]["metallic"] == pytest.approx(1.0)
    assert body["appearance"]["roughness"] == pytest.approx(0.3)
    assert body["appearance"]["base_color"] == pytest.approx([0.8, 0.5, 0.2])

    cap = parts["lid"]["shapes"][0]
    assert cap["appearance"]["metallic"] == pytest.approx(0.0)


def test_viewer_receives_the_mass_it_displays(tmp_path) -> None:
    """The parts panel was added in #68 but the viewer was never given the data."""
    model = ArticulatedObject("weighed")
    part = model.part("body")
    part.add(BoxGeometry([0.1, 0.1, 0.1]), name="shell", material=Material.STEEL)
    part.add(
        BoxGeometry([0.05, 0.02, 0.01]).translate(0.06, 0.0, 0.0),
        name="foot",
        material=Material.RUBBER,
    )

    result = export_object(model, tmp_path)
    version = _read_version(result.usdz)
    mass = cast(dict[str, Any], version["model"])["parts"][0]["mass"]

    assert mass is not None
    assert mass["kilograms"] == pytest.approx(
        0.1**3 * 7850.0 + 0.05 * 0.02 * 0.01 * 1200.0, rel=1e-3
    )
    assert mass["materials"] == ["rubber", "steel"]
    assert len(mass["center_of_mass"]) == 3

"""Publish articulated objects as validated USDZ packages."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import xatlas
from pxr import (  # pyright: ignore[reportAttributeAccessIssue]
    Gf,
    Sdf,
    Tf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,  # pyright: ignore[reportAttributeAccessIssue]
    UsdUtils,
    UsdValidation,
)

from mini_articraft.sdk import ambientcg
from mini_articraft.sdk._collision import MeshCollisionKernel, _rpy_matrix
from mini_articraft.sdk._mesh_core import geometry_to_trimesh
from mini_articraft.sdk.joints import Articulation, ArticulationType, MotionLimits
from mini_articraft.sdk.materials import Material, SurfaceKind
from mini_articraft.sdk.object import ArticulatedObject, Geometry
from mini_articraft.sdk.testing import DEFAULT_MESH_TOLERANCE

__all__ = ["ExportResult", "TextureExportReport", "export_object"]


@dataclass(frozen=True)
class TextureExportReport:
    requested_shapes: int = 0
    textured_shapes: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportResult:
    root: Path
    manifest: Path
    usdz: Path
    textures: TextureExportReport


@dataclass
class _TextureResolver:
    resolved: dict[SurfaceKind, ambientcg.TextureSet | None] = field(default_factory=dict)
    errors: dict[SurfaceKind, str] = field(default_factory=dict)

    def resolve(self, kind: SurfaceKind) -> ambientcg.TextureSet | None:
        if kind not in self.resolved:
            try:
                self.resolved[kind] = ambientcg.fetch_material(kind)[0]
            except Exception as exc:
                self.resolved[kind] = None
                self.errors[kind] = f"{kind.value}: {type(exc).__name__}: {exc}"
        return self.resolved[kind]


def export_object(
    obj: ArticulatedObject,
    output_dir: Path | str,
    *,
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE,
    textured: bool = False,
) -> ExportResult:
    """Publish ``obj`` as a validated USDZ package.

    With ``textured=True``, materials with an explicit ``SurfaceKind`` are
    upgraded to a tiling ambientCG PBR material with the maps embedded in the
    package. Materials without one -- or whose maps cannot be fetched -- stay
    parametric.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    obj.validate()

    usdz = _next_usdz_path(root / "usdz")
    manifest = root / "model.json"
    payload = _object_to_payload(obj) | {"files": {"usdz": usdz.relative_to(root).as_posix()}}
    manifest_temp = manifest.with_name(f".{manifest.name}.tmp")
    try:
        texture_report = _write_usdz(
            obj,
            usdz,
            mesh_tolerance,
            textured=textured,
        )
        manifest_temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_temp.replace(manifest)
    except BaseException:
        usdz.unlink(missing_ok=True)
        raise
    finally:
        manifest_temp.unlink(missing_ok=True)
    return ExportResult(root=root, manifest=manifest, usdz=usdz, textures=texture_report)


def _next_usdz_path(usdz_dir: Path) -> Path:
    indexes = [int(path.stem) for path in usdz_dir.glob("*.usdz") if path.stem.isdigit()]
    return usdz_dir / f"{(max(indexes) + 1) if indexes else 0:04d}.usdz"


def _write_usdz(
    obj: ArticulatedObject,
    path: Path,
    mesh_tolerance: float,
    *,
    textured: bool = False,
) -> TextureExportReport:
    if mesh_tolerance <= 0.0 or not math.isfinite(mesh_tolerance):
        raise ValueError("mesh_tolerance must be a positive finite number")

    with tempfile.TemporaryDirectory(prefix="mini-articraft-usd-") as temp_dir:
        stage_path = Path(temp_dir) / "model.usdc"
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        object_path = f"/World/{_safe_name(obj.name)}"
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(object_prim)
        _attrs(object_prim, {"name": obj.name, "units": "meters"})

        # Textured shapes copy their maps next to the layer (in temp_dir) so
        # CreateNewUsdzPackage bundles them into the .usdz.
        part_paths, texture_report = _write_parts(
            stage,
            f"{object_path}/parts",
            obj,
            mesh_tolerance,
            textured=textured,
            asset_dir=Path(temp_dir),
        )
        _write_articulations(stage, f"{object_path}/joints", obj, part_paths)

        path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Save()
        _validate_stage(stage)

        temp_path = path.with_name(f".{path.stem}.tmp.usdz")
        temp_path.unlink(missing_ok=True)
        try:
            if not UsdUtils.CreateNewUsdzPackage(str(stage_path), str(temp_path)):
                raise RuntimeError(f"failed to create USDZ package: {path}")
            _validate_usdz(temp_path)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return texture_report


def _write_parts(
    stage: Usd.Stage,
    scope_path: str,
    obj: ArticulatedObject,
    mesh_tolerance: float,
    *,
    textured: bool = False,
    asset_dir: Path | None = None,
) -> tuple[dict[str, str], TextureExportReport]:
    UsdGeom.Scope.Define(stage, scope_path)
    transforms = MeshCollisionKernel(obj, mesh_tolerance=mesh_tolerance).world_transforms({})
    safe_part_names = _safe_name_map(part.name for part in obj.parts)
    paths: dict[str, str] = {}
    resolver = _TextureResolver() if textured else None
    requested_shapes = 0
    textured_shapes = 0

    for part in obj.parts:
        part_path = f"{scope_path}/{safe_part_names[part.name]}"
        paths[part.name] = part_path
        part_prim = UsdGeom.Xform.Define(stage, part_path).GetPrim()
        _attrs(part_prim, {"name": part.name})
        UsdPhysics.RigidBodyAPI.Apply(part_prim)
        UsdGeom.Xformable(part_prim).AddTransformOp().Set(_gf_matrix(transforms[part.name]))

        shapes_path = f"{part_path}/shapes"
        UsdGeom.Scope.Define(stage, shapes_path)
        materials_path = f"{part_path}/materials"
        shape_entries = list(part._iter_shapes())
        safe_shape_names = _safe_name_map(shape.name for shape in shape_entries)
        if textured or any(shape.material is not None for shape in shape_entries):
            UsdGeom.Scope.Define(stage, materials_path)
        for shape in shape_entries:
            safe_shape = safe_shape_names[shape.name]
            mesh_path = f"{shapes_path}/{safe_shape}"
            material_path = f"{materials_path}/{safe_shape}"

            material = shape.material
            surface = material.surface if material is not None else None
            selection = resolver.resolve(surface) if resolver and surface else None
            if resolver is not None and surface is not None:
                requested_shapes += 1
            if selection is not None and material is not None and asset_dir is not None:
                _write_textured_shape(
                    stage,
                    mesh_path,
                    material_path,
                    shape,
                    selection,
                    material,
                    asset_dir,
                    mesh_tolerance,
                )
                textured_shapes += 1
                continue

            points, faces = _mesh(shape.geometry, mesh_tolerance)
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(points))
            _attrs(mesh.GetPrim(), {"name": shape.name})
            if shape.material is not None:
                # displayColor stays as a fallback for renderers that ignore
                # UsdShade; the bound UsdPreviewSurface carries the full material.
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*shape.material.base_color[:3])])
                mesh.CreateDisplayOpacityAttr([shape.material.opacity])
                _bind_material(stage, mesh, material_path, shape.material)
                _attrs(mesh.GetPrim(), _material_attrs(shape.material))
    errors = tuple(resolver.errors.values()) if resolver is not None else ()
    return paths, TextureExportReport(
        requested_shapes=requested_shapes,
        textured_shapes=textured_shapes,
        errors=errors,
    )


def _write_textured_shape(
    stage: Usd.Stage,
    mesh_path: str,
    material_path: str,
    shape,
    texture_set,
    material: Material,
    asset_dir: Path,
    mesh_tolerance: float,
) -> None:
    trimesh_obj = geometry_to_trimesh(shape.geometry, mesh_tolerance)
    points, faces, uvs, normals = _unwrap_mesh(trimesh_obj)
    gf_points = [Gf.Vec3f(*point) for point in points.tolist()]

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(gf_points)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(gf_points))
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals.tolist()])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(  # pyright: ignore[reportAttributeAccessIssue]
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(  # pyright: ignore[reportAttributeAccessIssue]
        [
            Gf.Vec2f(*uv)  # pyright: ignore[reportAttributeAccessIssue]
            for uv in uvs.tolist()
        ]
    )

    tint = material.base_color[:3]
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*tint)])
    mesh.CreateDisplayOpacityAttr([material.opacity])
    _bind_textured_material(stage, mesh, material_path, texture_set, material, asset_dir)
    _attrs(mesh.GetPrim(), {"name": shape.name})
    # The viewer keeps USDLoader's texture maps and layers the authored tint +
    # metalness on top (see viewer.html); these attrs carry that intent.
    _attrs(
        mesh.GetPrim(),
        {
            "material:metallic": material.metallic,
            "material:roughness": material.roughness,
            "material:baseColor": Gf.Vec3d(*tint),
            "material:opacity": material.opacity,
            "material:textured": 1.0,
        },
    )


def _unwrap_mesh(trimesh_obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a UV atlas while preserving source normals across seam vertices."""

    vertices = np.asarray(trimesh_obj.vertices)
    source_normals = np.asarray(trimesh_obj.vertex_normals)
    vertex_map, faces, uvs = xatlas.parametrize(vertices, np.asarray(trimesh_obj.faces))
    return (
        vertices[vertex_map].astype(np.float32),
        np.asarray(faces, dtype=np.int32),
        np.asarray(uvs, dtype=np.float32),
        source_normals[vertex_map].astype(np.float32),
    )


def _bind_textured_material(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    material_path: str,
    texture_set,
    authored: Material,
    asset_dir: Path,
) -> None:
    local: dict[str, str] = {}
    for channel, source in texture_set.maps().items():
        destination = asset_dir / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
        local[channel] = source.name

    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(name: str, filename: str, colorspace: str) -> UsdShade.Shader:
        node = UsdShade.Shader.Define(stage, f"{material_path}/{name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{filename}")
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        return node

    diffuse = texture("diffuseTex", local["base_color"], "sRGB")
    diffuse.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(*authored.base_color)  # pyright: ignore[reportAttributeAccessIssue]
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    if authored.opacity < 1.0:
        surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(authored.opacity)
    if "roughness" in local:
        rough = texture("roughTex", local["roughness"], "raw")
        rough.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*([authored.roughness] * 4))  # pyright: ignore[reportAttributeAccessIssue]
        )
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            rough.CreateOutput("r", Sdf.ValueTypeNames.Float)
        )
    if "normal" in local:
        normal = texture("normalTex", local["normal"], "raw")
        normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(2, 2, 2, 1)  # pyright: ignore[reportAttributeAccessIssue]
        )
        normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(-1, -1, -1, 0)  # pyright: ignore[reportAttributeAccessIssue]
        )
        surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
    if authored.metallic > 0.0 and "metalness" in local:
        metal = texture("metalTex", local["metalness"], "raw")
        metal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*([authored.metallic] * 4))  # pyright: ignore[reportAttributeAccessIssue]
        )
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            metal.CreateOutput("r", Sdf.ValueTypeNames.Float)
        )
    else:
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(authored.metallic)

    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _bind_material(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    material_path: str,
    material: Material,
) -> None:
    usd_material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    red, green, blue, alpha = material.base_color
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(red, green, blue))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material.metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material.roughness)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    if alpha < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(alpha)
    if material.emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*material.emissive)
        )
    usd_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(usd_material)


def _material_attrs(material: Material) -> dict[str, object]:
    red, green, blue, alpha = material.base_color
    values: dict[str, object] = {
        "material:metallic": material.metallic,
        "material:roughness": material.roughness,
        "material:baseColor": Gf.Vec3d(red, green, blue),
        "material:opacity": alpha,
    }
    if material.emissive is not None:
        values["material:emissive"] = Gf.Vec3d(*material.emissive)
    return values


def _write_articulations(
    stage: Usd.Stage,
    scope_path: str,
    obj: ArticulatedObject,
    part_paths: dict[str, str],
) -> None:
    UsdGeom.Scope.Define(stage, scope_path)
    safe_names = _safe_name_map(item.name for item in obj.articulations)
    for articulation in obj.articulations:
        schema = _articulation_schema(
            stage, f"{scope_path}/{safe_names[articulation.name]}", articulation
        )
        schema.CreateBody0Rel().SetTargets([part_paths[articulation.parent]])
        schema.CreateBody1Rel().SetTargets([part_paths[articulation.child]])
        _articulation_attrs(schema.GetPrim(), articulation)


def _articulation_schema(stage: Usd.Stage, path: str, articulation: Articulation):
    if articulation.articulation_type == ArticulationType.FIXED:
        schema = UsdPhysics.FixedJoint.Define(stage, path)
        _set_articulation_frames(schema, articulation)
        return schema

    schema_type = (
        UsdPhysics.PrismaticJoint
        if articulation.articulation_type == ArticulationType.PRISMATIC
        else UsdPhysics.RevoluteJoint
    )
    schema = schema_type.Define(stage, path)
    schema.CreateAxisAttr("X")
    _set_articulation_frames(schema, articulation, rotate_axis=True)
    limits = articulation.motion_limits
    if limits is not None and limits.lower is not None and limits.upper is not None:
        lower, upper = limits.lower, limits.upper
        if articulation.articulation_type == ArticulationType.REVOLUTE:
            lower, upper = math.degrees(lower), math.degrees(upper)
        schema.CreateLowerLimitAttr(lower)
        schema.CreateUpperLimitAttr(upper)
    return schema


def _set_articulation_frames(
    schema,
    articulation: Articulation,
    *,
    rotate_axis: bool = False,
) -> None:
    axis = _axis_matrix(articulation.axis) if rotate_axis else Gf.Matrix4d(1.0)
    # Gf uses row-vector composition while the SDK uses column vectors.
    frame = axis * _gf_matrix(_rpy_matrix(articulation.origin.rpy))
    schema.CreateLocalPos0Attr(Gf.Vec3f(*articulation.origin.xyz))
    schema.CreateLocalRot0Attr(_quat(frame))
    schema.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    schema.CreateLocalRot1Attr(_quat(axis))


def _articulation_attrs(prim: Usd.Prim, articulation: Articulation) -> None:
    values: dict[str, object] = {
        "name": articulation.name,
        "articulationType": cast(ArticulationType, articulation.articulation_type).value,
        "parent": articulation.parent,
        "child": articulation.child,
        "axis": Gf.Vec3d(*articulation.axis),
        "origin:xyz": Gf.Vec3d(*articulation.origin.xyz),
        "origin:rpy": Gf.Vec3d(*articulation.origin.rpy),
    }
    limits = articulation.motion_limits
    if limits is not None:
        values |= {
            "limits:effort": limits.effort,
            "limits:velocity": limits.velocity,
        }
        if limits.lower is not None and limits.upper is not None:
            values |= {"limits:lower": limits.lower, "limits:upper": limits.upper}
    _attrs(prim, values)


def _attrs(prim: Usd.Prim, values: dict[str, object]) -> None:
    types = {
        str: Sdf.ValueTypeNames.String,
        float: Sdf.ValueTypeNames.Double,
        Gf.Vec3d: Sdf.ValueTypeNames.Double3,
    }
    for name, value in values.items():
        prim.CreateAttribute(f"mini_articraft:{name}", types[type(value)], custom=True).Set(value)


def _mesh(
    geometry: Geometry,
    tolerance: float,
) -> tuple[list[Gf.Vec3f], list[tuple[int, int, int]]]:
    mesh = geometry_to_trimesh(geometry, tolerance)
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise TypeError("shape produced no USD mesh triangles")
    return (
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh.vertices],
        [(int(face[0]), int(face[1]), int(face[2])) for face in mesh.faces],
    )


def _object_to_payload(obj: ArticulatedObject) -> dict[str, object]:
    return {
        "name": obj.name,
        "units": "meters",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "parts": [
            {
                "name": part.name,
                "shapes": [
                    {
                        "name": shape.name,
                        "geometry_type": type(shape.geometry).__name__,
                        "color": shape.color,
                        "material": _material_payload(shape.material),
                    }
                    for shape in part._iter_shapes()
                ],
            }
            for part in obj.parts
        ],
        "articulations": [
            {
                "name": item.name,
                "type": cast(ArticulationType, item.articulation_type).value,
                "parent": item.parent,
                "child": item.child,
                "origin": {"xyz": item.origin.xyz, "rpy": item.origin.rpy},
                "axis": item.axis,
                "motion_limits": _limits(item.motion_limits),
            }
            for item in obj.articulations
        ],
    }


def _material_payload(material: Material | None) -> dict[str, object] | None:
    if material is None:
        return None
    return {
        "base_color": list(material.base_color),
        "metallic": material.metallic,
        "roughness": material.roughness,
        "emissive": list(material.emissive) if material.emissive is not None else None,
        "surface": material.surface.value if material.surface is not None else None,
    }


def _limits(limits: MotionLimits | None) -> dict[str, float | None] | None:
    if limits is None:
        return None
    return {
        "effort": limits.effort,
        "velocity": limits.velocity,
        "lower": limits.lower,
        "upper": limits.upper,
    }


def _axis_matrix(axis: tuple[float, float, float]) -> Gf.Matrix4d:
    length = math.hypot(*axis)
    if length <= 0.0:
        raise ValueError("articulation axis must be non-zero")
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(
        Gf.Rotation(
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(*(float(value) / length for value in axis)),
        )
    )
    return matrix


def _quat(matrix: Gf.Matrix4d) -> Gf.Quatf:
    return Gf.Quatf(matrix.ExtractRotationQuat())


def _gf_matrix(matrix) -> Gf.Matrix4d:
    rows = tuple(tuple(float(matrix[column, row]) for column in range(4)) for row in range(4))
    return Gf.Matrix4d(rows)


def _safe_name_map(names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for raw in names:
        base = _safe_name(str(raw))
        name = base
        index = 2
        while name in used:
            name = f"{base}_{index}"
            index += 1
        result[str(raw)] = name
        used.add(name)
    return result


def _safe_name(value: str) -> str:
    return Tf.MakeValidIdentifier(value.strip()) or "item"


def _validate_stage(stage: Usd.Stage) -> None:
    names = [
        "usdGeomValidators:StageMetadataChecker",
        "usdValidation:CompositionErrorTest",
        "usdPhysicsValidators:RigidBodyChecker",
        "usdPhysicsValidators:PhysicsJointChecker",
        "usdPhysicsValidators:ArticulationChecker",
    ]
    validators = UsdValidation.ValidationRegistry().GetOrLoadValidatorsByName(names)
    errors = UsdValidation.ValidationContext(validators).Validate(stage)
    if errors:
        raise RuntimeError(
            "OpenUSD validation failed: " + "; ".join(str(error) for error in errors)
        )


def _validate_usdz(path: Path) -> None:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError("OpenUSD could not open the generated USDZ package")
    validators = UsdValidation.ValidationRegistry().GetOrLoadValidatorsByName(
        ["usdUtilsValidators:UsdzPackageValidator"]
    )
    errors = UsdValidation.ValidationContext(validators).Validate(stage)
    if errors:
        raise RuntimeError(
            "OpenUSD USDZ validation failed: " + "; ".join(str(error) for error in errors)
        )

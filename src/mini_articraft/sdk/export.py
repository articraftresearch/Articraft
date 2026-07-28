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
import trimesh
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
from mini_articraft.sdk._mass_solver import ResolvedMass, resolve_mass
from mini_articraft.sdk._mesh_core import MeshGeometry, geometry_to_trimesh
from mini_articraft.sdk.joints import Articulation, ArticulationType, MotionLimits
from mini_articraft.sdk.materials import Appearance, Material
from mini_articraft.sdk.object import ArticulatedObject, Geometry
from mini_articraft.sdk.testing import DEFAULT_MESH_TOLERANCE

__all__ = ["ExportAudit", "ExportResult", "TextureExportReport", "export_object"]


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
    audit: ExportAudit


@dataclass(frozen=True)
class ExportAudit:
    part_count: int
    shape_count: int
    articulation_count: int
    triangle_count: int
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    meshes_with_normals: int
    material_bindings: int


@dataclass
class _TextureResolver:
    resolved: dict[Material, ambientcg.TextureSet | None] = field(default_factory=dict)
    errors: dict[Material, str] = field(default_factory=dict)

    def resolve(self, kind: Material) -> ambientcg.TextureSet | None:
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

    With ``textured=True``, shapes whose material has a texture set are
    upgraded to a tiling ambientCG PBR material with the maps embedded in the
    package. Materials without one -- or whose maps cannot be fetched -- stay
    parametric.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    obj.validate()

    usdz = _next_usdz_path(root / "usdz")
    manifest = root / "model.json"
    manifest_temp = manifest.with_name(f".{manifest.name}.tmp")
    try:
        texture_report, masses = _write_usdz(
            obj,
            usdz,
            mesh_tolerance,
            textured=textured,
        )
        audit = _audit_usdz(obj, usdz, mesh_tolerance)
        payload = _object_to_payload(obj, masses) | {
            "files": {"usdz": usdz.relative_to(root).as_posix()}
        }
        manifest_temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_temp.replace(manifest)
    except BaseException:
        usdz.unlink(missing_ok=True)
        raise
    finally:
        manifest_temp.unlink(missing_ok=True)
    return ExportResult(
        root=root,
        manifest=manifest,
        usdz=usdz,
        textures=texture_report,
        audit=audit,
    )


def _next_usdz_path(usdz_dir: Path) -> Path:
    indexes = [int(path.stem) for path in usdz_dir.glob("*.usdz") if path.stem.isdigit()]
    return usdz_dir / f"{(max(indexes) + 1) if indexes else 0:04d}.usdz"


def _write_usdz(
    obj: ArticulatedObject,
    path: Path,
    mesh_tolerance: float,
    *,
    textured: bool = False,
) -> tuple[TextureExportReport, dict[str, dict[str, object]]]:
    if mesh_tolerance <= 0.0 or not math.isfinite(mesh_tolerance):
        raise ValueError("mesh_tolerance must be a positive finite number")

    with tempfile.TemporaryDirectory(prefix="mini-articraft-usd-") as temp_dir:
        stage_path = Path(temp_dir) / "model.usdc"
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)  # pyright: ignore[reportAttributeAccessIssue]
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        object_path = f"/World/{_safe_name(obj.name)}"
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(object_prim)
        _attrs(object_prim, {"name": obj.name, "units": "meters"})

        # Textured shapes copy their maps next to the layer (in temp_dir) so
        # CreateNewUsdzPackage bundles them into the .usdz.
        part_paths, texture_report, masses = _write_parts(
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
    return texture_report, masses


def _write_parts(
    stage: Usd.Stage,
    scope_path: str,
    obj: ArticulatedObject,
    mesh_tolerance: float,
    *,
    textured: bool = False,
    asset_dir: Path | None = None,
) -> tuple[dict[str, str], TextureExportReport, dict[str, dict[str, object]]]:
    UsdGeom.Scope.Define(stage, scope_path)
    transforms = MeshCollisionKernel(obj, mesh_tolerance=mesh_tolerance).world_transforms({})
    safe_part_names = _safe_name_map(part.name for part in obj.parts)
    paths: dict[str, str] = {}
    masses: dict[str, dict[str, object]] = {}
    resolver = _TextureResolver() if textured else None
    # Contact behavior is a property of the substance, so one prim per material
    # is shared by every collider made of it.
    # A sibling of parts and joints, not a child of parts: these are shared by the
    # whole object rather than belonging to any one rigid body.
    physics_materials_path = f"{scope_path.rsplit('/', 1)[0]}/physics_materials"
    requested_shapes = 0
    textured_shapes = 0

    for part in obj.parts:
        part_path = f"{scope_path}/{safe_part_names[part.name]}"
        paths[part.name] = part_path
        part_prim = UsdGeom.Xform.Define(stage, part_path).GetPrim()
        _attrs(part_prim, {"name": part.name})
        UsdPhysics.RigidBodyAPI.Apply(part_prim)
        resolved_mass = _resolve_part_mass(part, mesh_tolerance)
        if resolved_mass is not None:
            _write_mass(part_prim, resolved_mass)
            masses[part.name] = _mass_entry(part, resolved_mass)
        UsdGeom.Xformable(part_prim).AddTransformOp().Set(_gf_matrix(transforms[part.name]))

        shapes_path = f"{part_path}/shapes"
        UsdGeom.Scope.Define(stage, shapes_path)
        materials_path = f"{part_path}/materials"
        shape_entries = list(part._iter_shapes())
        safe_shape_names = _safe_name_map(shape.name for shape in shape_entries)
        if textured or any(shape.resolved_appearance is not None for shape in shape_entries):
            UsdGeom.Scope.Define(stage, materials_path)
        for shape in shape_entries:
            safe_shape = safe_shape_names[shape.name]
            mesh_path = f"{shapes_path}/{safe_shape}"
            material_path = f"{materials_path}/{safe_shape}"

            appearance = shape.resolved_appearance
            surface = shape.material
            selection = resolver.resolve(surface) if resolver and surface else None
            if resolver is not None and surface is not None:
                requested_shapes += 1
            if selection is not None and appearance is not None and asset_dir is not None:
                _write_textured_shape(
                    stage,
                    mesh_path,
                    material_path,
                    shape,
                    selection,
                    appearance,
                    asset_dir,
                    mesh_tolerance,
                )
                textured_shapes += 1
                continue

            trimesh_obj = geometry_to_trimesh(shape.geometry, mesh_tolerance)
            points, faces = _mesh_arrays(trimesh_obj)
            normals, normal_interpolation = _normal_data(
                trimesh_obj,
                _normal_crease_angle(shape.geometry),
            )
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(points))
            mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals.tolist()])
            mesh.SetNormalsInterpolation(normal_interpolation)
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            _attrs(mesh.GetPrim(), {"name": shape.name})
            _write_collision(mesh, trimesh_obj)
            _write_physics_material(stage, mesh, shape.material, physics_materials_path)
            if appearance is not None:
                # displayColor stays as a fallback for renderers that ignore
                # UsdShade; the bound UsdPreviewSurface carries the full surface.
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*appearance.base_color[:3])])
                mesh.CreateDisplayOpacityAttr([appearance.opacity])
                _bind_material(stage, mesh, material_path, appearance)
                _attrs(mesh.GetPrim(), _material_attrs(appearance))
    errors = tuple(resolver.errors.values()) if resolver is not None else ()
    return (
        paths,
        TextureExportReport(
            requested_shapes=requested_shapes,
            textured_shapes=textured_shapes,
            errors=errors,
        ),
        masses,
    )


def _write_textured_shape(
    stage: Usd.Stage,
    mesh_path: str,
    material_path: str,
    shape,
    texture_set,
    material: Appearance,
    asset_dir: Path,
    mesh_tolerance: float,
) -> None:
    trimesh_obj = geometry_to_trimesh(shape.geometry, mesh_tolerance)
    points, faces, uvs, normals = _unwrap_mesh(
        trimesh_obj,
        _normal_crease_angle(shape.geometry),
    )
    gf_points = [Gf.Vec3f(*point) for point in points.tolist()]

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(gf_points)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(gf_points))
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals.tolist()])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _write_collision(mesh, trimesh_obj)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
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


def _unwrap_mesh(
    trimesh_obj,
    crease_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a UV atlas while preserving source normals across seam vertices."""

    vertices = np.asarray(trimesh_obj.vertices)
    faces = np.asarray(trimesh_obj.faces)
    source_normals, interpolation = _normal_data(trimesh_obj, crease_angle)
    if interpolation == UsdGeom.Tokens.faceVarying:
        vertices = vertices[faces].reshape((-1, 3))
        source_normals = source_normals.reshape((-1, 3))
        faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))
    vertex_map, faces, uvs = xatlas.parametrize(vertices, faces)
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
    authored: Appearance,
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
    material: Appearance,
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


def _material_attrs(material: Appearance) -> dict[str, object]:
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


def _resolve_part_mass(part, mesh_tolerance: float) -> ResolvedMass | None:
    """Measure the part's mass once, for both the USD prim and the manifest."""

    shapes_have_material = all(shape.material is not None for shape in part._iter_shapes())
    if part.mass_properties is None and not shapes_have_material:
        # Nothing to weigh with: mass stays absent rather than invented.
        return None
    shapes = [
        (geometry_to_trimesh(shape.geometry, mesh_tolerance), shape.material)
        for shape in part._iter_shapes()
    ]
    return resolve_mass(part.mass_properties, shapes, part_name=part.name)


def _mass_entry(part, resolved: ResolvedMass) -> dict[str, object]:
    """The manifest view of a resolved mass, for the viewer."""

    # Materials live on the shapes now, so a part reports the ones it is made of
    # rather than a single name it no longer has.
    materials = sorted(
        {shape.material.value for shape in part._iter_shapes() if shape.material is not None}
    )
    overrides = part.mass_properties
    return {
        "kilograms": round(resolved.mass, 6),
        "materials": materials,
        "density": None if overrides is None else overrides.density,
        "center_of_mass": [round(value, 6) for value in resolved.center_of_mass],
        "diagonal_inertia": [round(value, 9) for value in resolved.diagonal_inertia],
    }


def _write_mass(part_prim: Usd.Prim, resolved: ResolvedMass) -> None:
    """Author UsdPhysics.MassAPI from already-resolved mass values."""

    mass_api = UsdPhysics.MassAPI.Apply(part_prim)  # pyright: ignore[reportAttributeAccessIssue]
    mass_api.CreateMassAttr(float(resolved.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*resolved.center_of_mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*resolved.diagonal_inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*resolved.principal_axes))


def _write_collision(mesh: UsdGeom.Mesh, source: trimesh.Trimesh) -> None:
    """Make the shape's visible geometry its collider.

    Collision geometry is one-to-one with display geometry: what you see is what
    the simulator touches. Generating a separate, cheaper collider (a convex
    decomposition, a fitted proxy) is a job for a dedicated backend, not for the
    author of the model.

    The approximation is chosen from the mesh rather than declared. Engines
    generally cannot simulate a moving body against raw triangles, and a convex
    shape is both cheaper and exact as a hull, so only genuinely concave geometry
    pays for a decomposition.
    """

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)  # pyright: ignore[reportAttributeAccessIssue]
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)  # pyright: ignore[reportAttributeAccessIssue]
    mesh_collision.CreateApproximationAttr(_collision_approximation(source))


def _write_physics_material(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    material: Material | None,
    scope_path: str,
) -> None:
    """Bind how this collider behaves on contact, from what the shape is made of.

    Friction and restitution are surface properties, so they bind per collider
    rather than per part: a steel frame on rubber feet grips through the feet.
    Materials are shared, so every steel collider in the object points at one
    prim. A shape with no material gets none, and the engine applies its default.
    """

    if material is None:
        return
    path = f"{scope_path}/{material.value}"
    usd_material = UsdShade.Material.Get(stage, path)
    if not usd_material:
        UsdGeom.Scope.Define(stage, scope_path)
        usd_material = UsdShade.Material.Define(stage, path)
        physics = UsdPhysics.MaterialAPI.Apply(usd_material.GetPrim())  # pyright: ignore[reportAttributeAccessIssue]
        physics.CreateStaticFrictionAttr(material.static_friction)
        physics.CreateDynamicFrictionAttr(material.dynamic_friction)
        physics.CreateRestitutionAttr(material.restitution)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        usd_material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _collision_approximation(mesh: trimesh.Trimesh) -> str:
    """The USD approximation token that matches this mesh's shape."""

    return "convexHull" if mesh.is_convex else "convexDecomposition"


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


def _mesh_arrays(
    mesh,
) -> tuple[list[Gf.Vec3f], list[tuple[int, int, int]]]:
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise TypeError("shape produced no USD mesh triangles")
    return (
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh.vertices],
        [(int(face[0]), int(face[1]), int(face[2])) for face in mesh.faces],
    )


def _normal_crease_angle(geometry: Geometry) -> float:
    if isinstance(geometry, MeshGeometry) and geometry.normal_crease_angle is not None:
        return geometry.normal_crease_angle
    return math.radians(45.0)


def _normal_data(mesh, crease_angle: float) -> tuple[np.ndarray, str]:
    if crease_angle >= math.pi - 1e-10:
        return np.asarray(mesh.vertex_normals, dtype=np.float32), UsdGeom.Tokens.vertex
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if crease_angle <= 1e-10:
        return np.repeat(face_normals, 3, axis=0).astype(np.float32), UsdGeom.Tokens.faceVarying
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_faces = np.asarray(mesh.vertex_faces, dtype=np.int64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    cosine = math.cos(crease_angle)
    corner_normals = np.empty((len(faces), 3, 3), dtype=np.float64)
    for face_index, face in enumerate(faces):
        reference = face_normals[face_index]
        for corner_index, vertex_index in enumerate(face):
            adjacent = vertex_faces[vertex_index]
            adjacent = adjacent[adjacent >= 0]
            aligned = adjacent[(face_normals[adjacent] @ reference) >= cosine]
            weighted = (face_normals[aligned] * face_areas[aligned, None]).sum(axis=0)
            length = float(np.linalg.norm(weighted))
            corner_normals[face_index, corner_index] = (
                reference if length <= 1e-14 else weighted / length
            )
    return corner_normals.reshape((-1, 3)).astype(np.float32), UsdGeom.Tokens.faceVarying


def _object_to_payload(
    obj: ArticulatedObject, masses: dict[str, dict[str, object]] | None = None
) -> dict[str, object]:
    masses = masses or {}
    return {
        "name": obj.name,
        "units": "meters",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "parts": [
            {
                "name": part.name,
                "mass": masses.get(part.name),
                "shapes": [
                    {
                        "name": shape.name,
                        "geometry_type": type(shape.geometry).__name__,
                        "color": shape.color,
                        "material": None if shape.material is None else shape.material.value,
                        "appearance": _appearance_payload(shape.resolved_appearance),
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


def _appearance_payload(appearance: Appearance | None) -> dict[str, object] | None:
    if appearance is None:
        return None
    return {
        "base_color": list(appearance.base_color),
        "metallic": appearance.metallic,
        "roughness": appearance.roughness,
        "emissive": list(appearance.emissive) if appearance.emissive is not None else None,
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


def _audit_usdz(
    obj: ArticulatedObject,
    path: Path,
    mesh_tolerance: float,
) -> ExportAudit:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError("OpenUSD could not reopen the generated USDZ package for audit")

    expected_parts = {part.name for part in obj.parts}
    expected_shapes = {
        (part.name, shape.name) for part in obj.parts for shape in part._iter_shapes()
    }
    expected_joints = {joint.name: joint for joint in obj.articulations}
    found_parts: set[str] = set()
    found_shapes: set[tuple[str, str]] = set()
    found_joints: dict[str, Usd.Prim] = {}
    exported_points: list[np.ndarray] = []
    triangle_count = 0
    normal_meshes = 0
    material_bindings = 0
    source_meshes: dict[tuple[str, str], trimesh.Trimesh] = {}
    xforms = MeshCollisionKernel(obj, mesh_tolerance=mesh_tolerance).world_transforms({})
    source_points: list[np.ndarray] = []

    for part in obj.parts:
        for shape in part._iter_shapes():
            mesh = geometry_to_trimesh(shape.geometry, mesh_tolerance).copy()
            mesh.apply_transform(xforms[part.name])
            source_meshes[(part.name, shape.name)] = mesh
            source_points.append(np.asarray(mesh.vertices, dtype=np.float64))

    cache = UsdGeom.XformCache(  # pyright: ignore[reportAttributeAccessIssue]
        Usd.TimeCode.Default()  # pyright: ignore[reportAttributeAccessIssue]
    )
    for prim in stage.Traverse():
        path_text = str(prim.GetPath())
        authored_name = _custom_string(prim, "name")
        if "/parts/" in path_text and "/shapes/" not in path_text and authored_name:
            found_parts.add(authored_name)
        if "/joints/" in path_text and authored_name:
            found_joints[authored_name] = prim
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)  # pyright: ignore[reportCallIssue]
        shape_name = authored_name
        part_prim = prim.GetParent().GetParent()
        part_name = _custom_string(part_prim, "name")
        if not part_name or not shape_name:
            raise RuntimeError(f"USDZ audit found an unnamed mesh at {prim.GetPath()}")
        selector = (part_name, shape_name)
        found_shapes.add(selector)
        points = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or faces is None:
            raise RuntimeError(f"USDZ audit found an empty mesh at {prim.GetPath()}")
        triangle_count += len(faces) // 3
        matrix = cache.GetLocalToWorldTransform(prim)
        exported_points.append(
            np.asarray(
                [
                    tuple(
                        matrix.Transform(
                            Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                        )
                    )
                    for point in points
                ],
                dtype=np.float64,
            )
        )
        normals = mesh.GetNormalsAttr().Get()
        if normals is None or len(normals) == 0:
            raise RuntimeError(f"USDZ audit found a mesh without normals at {prim.GetPath()}")
        normal_meshes += 1
        orientation = mesh.GetOrientationAttr().Get()
        if orientation != UsdGeom.Tokens.rightHanded:
            raise RuntimeError(f"USDZ audit found non-right-handed winding at {prim.GetPath()}")
        # Purpose-scoped: a physics material also binds here, and it is not an
        # appearance, so counting it would make every collider look textured.
        if UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("").GetTargets():
            material_bindings += 1
        source = source_meshes.get(selector)
        if source is None:
            raise RuntimeError(f"USDZ audit found an unexpected mesh {selector!r}")
        if len(source.faces) != len(faces) // 3:
            raise RuntimeError(
                f"USDZ audit triangle mismatch for {selector!r}: "
                f"source={len(source.faces)} package={len(faces) // 3}"
            )
        source_sign = math.copysign(1.0, _signed_volume(source))
        exported_mesh = trimesh.Trimesh(
            vertices=np.asarray([tuple(point) for point in points], dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
            process=False,
        )
        exported_sign = math.copysign(1.0, _signed_volume(exported_mesh))
        if source.is_watertight and source_sign != exported_sign:
            raise RuntimeError(f"USDZ audit winding changed for {selector!r}")

    if found_parts != expected_parts:
        raise RuntimeError(
            f"USDZ audit part mismatch: expected={sorted(expected_parts)!r} "
            f"found={sorted(found_parts)!r}"
        )
    if found_shapes != expected_shapes:
        raise RuntimeError(
            f"USDZ audit shape mismatch: expected={sorted(expected_shapes)!r} "
            f"found={sorted(found_shapes)!r}"
        )
    if set(found_joints) != set(expected_joints):
        raise RuntimeError(
            f"USDZ audit articulation mismatch: expected={sorted(expected_joints)!r} "
            f"found={sorted(found_joints)!r}"
        )
    for name, joint in expected_joints.items():
        prim = found_joints[name]
        if (
            _custom_string(prim, "articulationType")
            != cast(ArticulationType, joint.articulation_type).value
        ):
            raise RuntimeError(f"USDZ audit joint type mismatch for {name!r}")
        if _custom_string(prim, "parent") != joint.parent:
            raise RuntimeError(f"USDZ audit joint parent mismatch for {name!r}")
        if _custom_string(prim, "child") != joint.child:
            raise RuntimeError(f"USDZ audit joint child mismatch for {name!r}")
        _expect_vector_attr(prim, "axis", joint.axis, joint_name=name)
        _expect_vector_attr(prim, "origin:xyz", joint.origin.xyz, joint_name=name)
        _expect_vector_attr(prim, "origin:rpy", joint.origin.rpy, joint_name=name)
        limits = joint.motion_limits
        if limits is not None:
            _expect_number_attr(prim, "limits:effort", limits.effort, joint_name=name)
            _expect_number_attr(prim, "limits:velocity", limits.velocity, joint_name=name)
            if limits.lower is not None:
                _expect_number_attr(prim, "limits:lower", limits.lower, joint_name=name)
            if limits.upper is not None:
                _expect_number_attr(prim, "limits:upper", limits.upper, joint_name=name)

    # Appearance drives the bound UsdPreviewSurface, and a material supplies one
    # when nothing overrides it, so count what the shape will actually look like.
    expected_materials = sum(
        shape.resolved_appearance is not None for part in obj.parts for shape in part._iter_shapes()
    )
    if material_bindings != expected_materials:
        raise RuntimeError(
            f"USDZ audit material binding mismatch: "
            f"expected={expected_materials} found={material_bindings}"
        )
    source_bounds = _point_bounds(np.concatenate(source_points, axis=0))
    package_bounds = _point_bounds(np.concatenate(exported_points, axis=0))
    tolerance = max(mesh_tolerance * 2.0, 1e-6)
    if not np.allclose(source_bounds, package_bounds, rtol=1e-6, atol=tolerance):
        raise RuntimeError(
            f"USDZ audit bounds mismatch: source={source_bounds!r} package={package_bounds!r}"
        )
    return ExportAudit(
        part_count=len(found_parts),
        shape_count=len(found_shapes),
        articulation_count=len(found_joints),
        triangle_count=triangle_count,
        bounds=package_bounds,
        meshes_with_normals=normal_meshes,
        material_bindings=material_bindings,
    )


def _custom_string(prim: Usd.Prim, name: str) -> str:
    value = prim.GetAttribute(f"mini_articraft:{name}").Get()
    return "" if value is None else str(value)


def _expect_vector_attr(
    prim: Usd.Prim,
    name: str,
    expected: tuple[float, float, float],
    *,
    joint_name: str,
) -> None:
    value = prim.GetAttribute(f"mini_articraft:{name}").Get()
    if value is None or not np.allclose(tuple(value), expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"USDZ audit joint {name} mismatch for {joint_name!r}: "
            f"expected={expected!r} found={value!r}"
        )


def _expect_number_attr(
    prim: Usd.Prim,
    name: str,
    expected: float,
    *,
    joint_name: str,
) -> None:
    value = prim.GetAttribute(f"mini_articraft:{name}").Get()
    if value is None or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"USDZ audit joint {name} mismatch for {joint_name!r}: "
            f"expected={expected!r} found={value!r}"
        )


def _signed_volume(mesh) -> float:
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
    return float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )


def _point_bounds(
    points: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(float(value) for value in points.min(axis=0)),
        tuple(float(value) for value in points.max(axis=0)),
    )  # type: ignore[return-value]

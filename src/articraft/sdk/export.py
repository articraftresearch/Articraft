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
    Kind,
    Sdf,
    Tf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,  # pyright: ignore[reportAttributeAccessIssue]
    UsdUtils,
    UsdValidation,
)

from articraft.sdk import ambientcg
from articraft.sdk._collision import MeshCollisionKernel, _rpy_matrix
from articraft.sdk._mesh.core import MeshGeometry, geometry_to_trimesh
from articraft.sdk.assembly import (
    WORLD,
    Joint,
    JointAxis,
    JointFrame,
    ResolvedRigidBodyAssembly,
    RigidBodyAssembly,
)
from articraft.sdk.bodies import RigidBody
from articraft.sdk.joints import (
    Articulation,
    ArticulationType,
    MotionLimits,
    partition_articulations,
)
from articraft.sdk.mass import ResolvedMass, resolve_mass
from articraft.sdk.materials import Material, is_library_material
from articraft.sdk.object import ArticulatedObject, Geometry, Part
from articraft.sdk.physics import BodyState, PhysicsScene
from articraft.sdk.testing import DEFAULT_MESH_TOLERANCE

__all__ = [
    "AssemblyExportAudit",
    "AssemblyExportResult",
    "ExportAudit",
    "ExportResult",
    "TextureExportReport",
    "export_assembly",
    "export_object",
]


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


@dataclass(frozen=True)
class AssemblyExportAudit:
    rigid_body_count: int
    shape_count: int
    joint_count: int
    articulation_count: int
    triangle_count: int
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    meshes_with_normals: int
    material_bindings: int


@dataclass(frozen=True)
class AssemblyExportResult:
    root: Path
    manifest: Path
    usdz: Path
    textures: TextureExportReport
    audit: AssemblyExportAudit


@dataclass
class _TextureResolver:
    resolved: dict[Material, ambientcg.TextureSet | None] = field(default_factory=dict)
    errors: dict[Material, str] = field(default_factory=dict)

    def resolve(self, kind: Material) -> ambientcg.TextureSet | None:
        if kind.texture is None:
            return None
        if kind not in self.resolved:
            try:
                self.resolved[kind] = ambientcg.fetch_material(kind)[0]
            except Exception as exc:
                self.resolved[kind] = None
                self.errors[kind] = f"{kind.name}: {type(exc).__name__}: {exc}"
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


def export_assembly(
    assembly: RigidBodyAssembly,
    output_dir: Path | str,
    *,
    mesh_tolerance: float = DEFAULT_MESH_TOLERANCE,
    textured: bool = False,
) -> AssemblyExportResult:
    """Publish a rigid-body graph as a validated USDZ package and manifest v2."""

    if not isinstance(assembly, RigidBodyAssembly):
        raise TypeError("export_assembly requires a RigidBodyAssembly")
    resolved = assembly.resolve()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    usdz = _next_usdz_path(root / "usdz")
    manifest = root / "model.json"
    manifest_temp = manifest.with_name(f".{manifest.name}.tmp")
    try:
        texture_report, masses = _write_assembly_usdz(
            resolved,
            usdz,
            mesh_tolerance,
            textured=textured,
        )
        audit = _audit_assembly_usdz(resolved, usdz, mesh_tolerance)
        payload = _assembly_to_payload(resolved, masses) | {
            "files": {"usdz": usdz.relative_to(root).as_posix()}
        }
        manifest_temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_temp.replace(manifest)
    except BaseException:
        usdz.unlink(missing_ok=True)
        raise
    finally:
        manifest_temp.unlink(missing_ok=True)
    return AssemblyExportResult(
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

    with tempfile.TemporaryDirectory(prefix="articraft-usd-") as temp_dir:
        stage_path = Path(temp_dir) / "model.usdc"
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)  # pyright: ignore[reportAttributeAccessIssue]
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        _write_scene(stage, "/World/physicsScene", obj.scene)

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


def _write_scene(stage: Usd.Stage, path: str, scene: PhysicsScene) -> None:
    """Author the one physics scene the whole stage is simulated in.

    There is exactly one, so no body needs a simulation owner relationship: a
    simulator that finds a single scene uses it for everything.
    """

    usd_scene = UsdPhysics.Scene.Define(stage, path)
    usd_scene.CreateGravityDirectionAttr(Gf.Vec3f(*scene.direction))
    usd_scene.CreateGravityMagnitudeAttr(scene.magnitude)


def _write_body_state(rigid_body, state: BodyState) -> None:
    """Author how the part starts, on the RigidBodyAPI already applied to it.

    USD measures angular velocity in degrees per second; the SDK uses radians
    everywhere, as it does for revolute joint limits.
    """

    rigid_body.CreateRigidBodyEnabledAttr(state.enabled)
    rigid_body.CreateKinematicEnabledAttr(state.kinematic)
    rigid_body.CreateStartsAsleepAttr(state.starts_asleep)
    rigid_body.CreateVelocityAttr(Gf.Vec3f(*state.linear_velocity))
    rigid_body.CreateAngularVelocityAttr(
        Gf.Vec3f(*(math.degrees(value) for value in state.angular_velocity))
    )


def _write_assembly_usdz(
    resolved: ResolvedRigidBodyAssembly,
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
        assembly_path = f"/World/{_safe_name(resolved.name)}"
        assembly_prim = UsdGeom.Xform.Define(stage, assembly_path).GetPrim()
        Usd.ModelAPI(assembly_prim).SetKind(Kind.Tokens.assembly)
        _attrs(assembly_prim, {"name": resolved.name, "units": "meters", "schemaVersion": 2})

        transforms = resolved.world_transforms()
        body_paths, texture_report, masses = _write_body_geometry(
            stage,
            f"{assembly_path}/rigid_bodies",
            resolved.rigid_bodies,
            transforms,
            mesh_tolerance,
            textured=textured,
            asset_dir=Path(temp_dir),
        )
        joint_paths = _write_graph_joints(
            stage,
            f"{assembly_path}/joints",
            resolved,
            body_paths,
        )
        _write_articulation_roots(resolved, body_paths, joint_paths, stage)

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
    # Bake at the authored zero pose, drives unresolved: joint value zero must
    # stay the baked pose so limits, sliders, and MJCF qpos0 line up.
    transforms = MeshCollisionKernel(obj, mesh_tolerance=mesh_tolerance)._place({})
    return _write_body_geometry(
        stage,
        scope_path,
        tuple(obj.parts),
        transforms,
        mesh_tolerance,
        textured=textured,
        asset_dir=asset_dir,
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


def _bind_shared_appearance(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    appearance: Material,
    scope_path: str,
    shared: dict[Material, str],
) -> None:
    """Bind the mesh to a prim for this appearance, defining it only once."""

    path = shared.get(appearance)
    if path is None:
        UsdGeom.Scope.Define(stage, scope_path)
        path = f"{scope_path}/appearance_{len(shared)}"
        _bind_material(stage, mesh, path, appearance)
        shared[appearance] = path
        return
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material.Get(stage, path))


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


def _substance_attrs(shape) -> dict[str, object]:
    """What the shape is made of, recorded on the prim for viewers and tools."""

    values: dict[str, object] = {}
    if shape.material is not None:
        values["material"] = shape.material.name
    if shape.coating is not None:
        values["coating"] = shape.coating.name
    return values


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
        {shape.material.name for shape in part._iter_shapes() if shape.material is not None}
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

    if material is None or material.friction is None:
        # No friction authored means we do not know it; the engine's default is
        # a better answer than a number we invented.
        return
    path = f"{scope_path}/{_safe_name(material.name)}"
    usd_material = UsdShade.Material.Get(stage, path)
    if not usd_material:
        UsdGeom.Scope.Define(stage, scope_path)
        usd_material = UsdShade.Material.Define(stage, path)
        physics = UsdPhysics.MaterialAPI.Apply(usd_material.GetPrim())  # pyright: ignore[reportAttributeAccessIssue]
        static, dynamic = material.friction
        physics.CreateStaticFrictionAttr(static)
        physics.CreateDynamicFrictionAttr(dynamic)
        if material.restitution is not None:
            physics.CreateRestitutionAttr(material.restitution)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        usd_material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _collision_approximation(mesh: trimesh.Trimesh) -> str:
    """The USD approximation token that matches this mesh's shape."""

    return "convexHull" if mesh.is_convex else "convexDecomposition"


def _body_name(endpoint) -> str:
    """The name of an endpoint the caller has already checked is not WORLD."""

    return cast(RigidBody, endpoint).name


def _write_graph_joints(
    stage: Usd.Stage,
    scope_path: str,
    resolved: ResolvedRigidBodyAssembly,
    body_paths: dict[str, str],
) -> dict[str, str]:
    UsdGeom.Scope.Define(stage, scope_path)
    safe_names = _safe_name_map(item.joint.name for item in resolved.joints)
    paths: dict[str, str] = {}
    for item in resolved.joints:
        joint = item.joint
        path = f"{scope_path}/{safe_names[joint.name]}"
        paths[joint.name] = path
        schema = _graph_joint_schema(stage, path, joint)
        if joint.body0 is not WORLD:
            schema.CreateBody0Rel().SetTargets([body_paths[_body_name(joint.body0)]])
        if joint.body1 is not WORLD:
            schema.CreateBody1Rel().SetTargets([body_paths[_body_name(joint.body1)]])
        schema.CreateExcludeFromArticulationAttr(item.exclude_from_articulation)
        _attrs(
            schema.GetPrim(),
            {
                "name": joint.name,
                "jointType": _graph_joint_type(joint),
                "body0": "WORLD" if joint.body0 is WORLD else _body_name(joint.body0),
                "body1": "WORLD" if joint.body1 is WORLD else _body_name(joint.body1),
                "frame0:xyz": Gf.Vec3d(*joint.frame0.xyz),
                "frame0:rpy": Gf.Vec3d(*joint.frame0.rpy),
                "frame1:xyz": Gf.Vec3d(*joint.frame1.xyz),
                "frame1:rpy": Gf.Vec3d(*joint.frame1.rpy),
                "dofs": json.dumps(
                    [
                        {
                            "axis": cast(JointAxis, dof.axis).value,
                            "limits": dof.limits,
                        }
                        for dof in joint.dofs
                    ],
                    separators=(",", ":"),
                ),
                "articulation": item.articulation or "",
                "excludeFromArticulation": item.exclude_from_articulation,
            },
        )
    return paths


def _graph_joint_schema(stage: Usd.Stage, path: str, joint: Joint):
    if joint.is_fixed:
        schema = UsdPhysics.FixedJoint.Define(stage, path)
    elif joint.is_revolute:
        schema = UsdPhysics.RevoluteJoint.Define(stage, path)
        axis = cast(JointAxis, joint.dofs[0].axis)
        schema.CreateAxisAttr(axis.value[-1].upper())
        _write_specialized_limits(schema, joint)
    elif joint.is_prismatic:
        schema = UsdPhysics.PrismaticJoint.Define(stage, path)
        axis = cast(JointAxis, joint.dofs[0].axis)
        schema.CreateAxisAttr(axis.value[-1].upper())
        _write_specialized_limits(schema, joint)
    else:
        schema = UsdPhysics.Joint.Define(stage, path)
        authored = {cast(JointAxis, dof.axis): dof for dof in joint.dofs}
        for axis in JointAxis:
            dof = authored.get(axis)
            if dof is not None and dof.limits is None:
                continue
            limit = UsdPhysics.LimitAPI.Apply(schema.GetPrim(), axis.value)
            if dof is None:
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)
                continue
            lower, upper = cast(tuple[float, float], dof.limits)
            if axis.is_rotational:
                lower, upper = math.degrees(lower), math.degrees(upper)
            limit.CreateLowAttr(lower)
            limit.CreateHighAttr(upper)
    _set_joint_frame_attrs(schema, joint.frame0, joint.frame1)
    return schema


def _write_specialized_limits(schema, joint: Joint) -> None:
    dof = joint.dofs[0]
    if dof.limits is None:
        return
    lower, upper = dof.limits
    if cast(JointAxis, dof.axis).is_rotational:
        lower, upper = math.degrees(lower), math.degrees(upper)
    schema.CreateLowerLimitAttr(lower)
    schema.CreateUpperLimitAttr(upper)


def _set_joint_frame_attrs(schema, frame0: JointFrame, frame1: JointFrame) -> None:
    schema.CreateLocalPos0Attr(Gf.Vec3f(*frame0.xyz))
    schema.CreateLocalRot0Attr(_quat(_gf_matrix(_rpy_matrix(frame0.rpy))))
    schema.CreateLocalPos1Attr(Gf.Vec3f(*frame1.xyz))
    schema.CreateLocalRot1Attr(_quat(_gf_matrix(_rpy_matrix(frame1.rpy))))


def _write_articulation_roots(
    resolved: ResolvedRigidBodyAssembly,
    body_paths: dict[str, str],
    joint_paths: dict[str, str],
    stage: Usd.Stage,
) -> None:
    for item in resolved.articulations:
        root = item.articulation.root
        path = joint_paths[root.name] if isinstance(root, Joint) else body_paths[root.name]
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(path))


def _graph_joint_type(joint: Joint) -> str:
    if joint.is_fixed:
        return "fixed"
    if joint.is_revolute:
        return "revolute"
    if joint.is_prismatic:
        return "prismatic"
    return "d6"


def _write_articulations(
    stage: Usd.Stage,
    scope_path: str,
    obj: ArticulatedObject,
    part_paths: dict[str, str],
) -> None:
    UsdGeom.Scope.Define(stage, scope_path)
    safe_names = _safe_name_map(item.name for item in obj.articulations)
    _tree, loops = partition_articulations(obj.articulations)
    loop_names = {item.name for item in loops}
    kernel = MeshCollisionKernel(obj, mesh_tolerance=DEFAULT_MESH_TOLERANCE)
    rest = kernel._place({})
    resolved = kernel._resolve_drives({})
    for articulation in obj.articulations:
        if articulation.drive is None:
            continue
        value = resolved.get(articulation.name, 0.0)
        if abs(value) > 1e-3:
            raise ValueError(
                f"driven articulation {articulation.name!r} solves to {value:.4f} at the "
                "rest pose; zero must be the assembled pose, so fold the rest angle into "
                "the joint origin's rpy or the rest gap into the drive's rest_length"
            )
    for articulation in obj.articulations:
        schema = _articulation_schema(
            stage, f"{scope_path}/{safe_names[articulation.name]}", articulation
        )
        schema.CreateBody0Rel().SetTargets([part_paths[articulation.parent]])
        schema.CreateBody1Rel().SetTargets([part_paths[articulation.child]])
        if articulation.name in loop_names:
            # USD articulations are trees, but a regular joint outside the
            # articulation may close a loop. The solver still enforces it.
            schema.CreateExcludeFromArticulationAttr(True)
            # A tree child's frame sits on its joint, so localPos1 = 0 there.
            # A loop child's frame belongs to its tree parent, so the pin must
            # be located in that frame explicitly or engines snap the child's
            # origin onto the pin.
            axis_rot = _axis_matrix(articulation.axis)
            local0 = axis_rot * _gf_matrix(_rpy_matrix(articulation.origin.rpy))
            frame0 = local0 * Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*articulation.origin.xyz))
            world = frame0 * _gf_matrix(rest[articulation.parent])
            frame1 = world * _gf_matrix(np.linalg.inv(rest[articulation.child]))
            schema.CreateLocalPos1Attr(Gf.Vec3f(frame1.ExtractTranslation()))
            schema.CreateLocalRot1Attr(_quat(frame1))
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
        "driven": "true" if articulation.drive is not None else "false",
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
        bool: Sdf.ValueTypeNames.Bool,
        int: Sdf.ValueTypeNames.Int,
        float: Sdf.ValueTypeNames.Double,
        Gf.Vec3d: Sdf.ValueTypeNames.Double3,
    }
    for name, value in values.items():
        prim.CreateAttribute(f"articraft:{name}", types[type(value)], custom=True).Set(value)


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
    loop_names = {loop.name for loop in partition_articulations(obj.articulations)[1]}
    return {
        "name": obj.name,
        "units": "meters",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "scene": {
            "gravity_direction": list(obj.scene.direction),
            "gravity_magnitude": obj.scene.magnitude,
        },
        "parts": [
            {
                "name": part.name,
                "mass": masses.get(part.name),
                "body_state": _body_state_payload(part),
                "shapes": [
                    {
                        "name": shape.name,
                        "geometry_type": type(shape.geometry).__name__,
                        "color": shape.color,
                        "material": _material_payload(shape.material),
                        "coating": _material_payload(shape.coating),
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
                "closes_loop": item.name in loop_names,
                "driven": item.drive is not None,
            }
            for item in obj.articulations
        ],
    }


def _body_state_payload(part: Part) -> dict[str, object]:
    """The manifest view of how a part starts. Angles stay in radians here."""

    state = part.body_state
    return {
        "enabled": state.enabled,
        "kinematic": state.kinematic,
        "linear_velocity": list(state.linear_velocity),
        "angular_velocity": list(state.angular_velocity),
        "starts_asleep": state.starts_asleep,
    }


def _assembly_to_payload(
    resolved: ResolvedRigidBodyAssembly,
    masses: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    masses = masses or {}
    return {
        "schema_version": 2,
        "name": resolved.name,
        "units": "meters",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "rigid_bodies": [
            {
                "name": body.name,
                "mass": masses.get(body.name),
                "shapes": [
                    {
                        "name": shape.name,
                        "geometry_type": type(shape.geometry).__name__,
                        "color": shape.color,
                        "material": _material_payload(shape.material),
                        "coating": _material_payload(shape.coating),
                    }
                    for shape in body._iter_shapes()
                ],
            }
            for body in resolved.rigid_bodies
        ],
        "joints": [
            {
                "name": item.joint.name,
                "type": _graph_joint_type(item.joint),
                "body0": None if item.joint.body0 is WORLD else _body_name(item.joint.body0),
                "body1": None if item.joint.body1 is WORLD else _body_name(item.joint.body1),
                "frame0": {
                    "xyz": item.joint.frame0.xyz,
                    "rpy": item.joint.frame0.rpy,
                },
                "frame1": {
                    "xyz": item.joint.frame1.xyz,
                    "rpy": item.joint.frame1.rpy,
                },
                "dofs": [
                    {
                        "axis": cast(JointAxis, dof.axis).value,
                        "limits": None if dof.limits is None else list(dof.limits),
                    }
                    for dof in item.joint.dofs
                ],
                "articulation": item.articulation,
                "exclude_from_articulation": item.exclude_from_articulation,
            }
            for item in resolved.joints
        ],
        "articulations": [
            {
                "name": item.articulation.name,
                "root": {
                    "type": (
                        "joint" if isinstance(item.articulation.root, Joint) else "rigid_body"
                    ),
                    "name": item.articulation.root.name,
                },
                "joints": [joint.name for joint in item.joints],
                "rigid_bodies": [body.name for body in item.rigid_bodies],
            }
            for item in resolved.articulations
        ],
        "reference_state": {
            "body_poses": {
                name: [list(row) for row in matrix]
                for name, matrix in resolved.reference_state.body_poses.items()
            },
            "dof_positions": dict(resolved.reference_state.dof_positions),
        },
    }


def _material_payload(material: Material | None) -> dict[str, object] | None:
    """The manifest view of a material.

    ``library`` records whether the numbers came from the checked library or were
    derived or invented, so a reviewer can tell them apart.
    """

    if material is None:
        return None
    return {
        "name": material.name,
        "library": is_library_material(material),
        "density": material.density,
        "friction": list(material.friction) if material.friction is not None else None,
        "restitution": material.restitution,
        "base_color": list(material.base_color),
        "metallic": material.metallic,
        "roughness": material.roughness,
        # The viewer reads opacity directly; without it, alpha in base_color was
        # silently ignored and glass rendered solid.
        "opacity": material.opacity,
        "emissive": list(material.emissive) if material.emissive is not None else None,
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
    physics_scenes = 0
    exported_points: list[np.ndarray] = []
    triangle_count = 0
    normal_meshes = 0
    material_bindings = 0
    source_meshes: dict[tuple[str, str], trimesh.Trimesh] = {}
    xforms = MeshCollisionKernel(obj, mesh_tolerance=mesh_tolerance)._place({})
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
        if prim.IsA(UsdPhysics.Scene):
            physics_scenes += 1
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

    # Bodies carry no simulation owner, so a second scene would leave which one
    # governs them up to the reader.
    if physics_scenes != 1:
        raise RuntimeError(f"USDZ audit expected exactly one physics scene, found {physics_scenes}")
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
    loop_names = {loop.name for loop in partition_articulations(obj.articulations)[1]}
    for name, joint in expected_joints.items():
        prim = found_joints[name]
        excluded_attr = prim.GetAttribute("physics:excludeFromArticulation")
        excluded = bool(excluded_attr.Get()) if excluded_attr else False
        if excluded != (name in loop_names):
            raise RuntimeError(f"USDZ audit loop exclusion mismatch for {name!r}")
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

    # Material drives the bound UsdPreviewSurface, and a material supplies one
    # when nothing overrides it, so count what the shape will actually look like.
    expected_materials = sum(
        shape.display_material is not None for part in obj.parts for shape in part._iter_shapes()
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


def _audit_assembly_usdz(
    resolved: ResolvedRigidBodyAssembly,
    path: Path,
    mesh_tolerance: float,
) -> AssemblyExportAudit:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError("OpenUSD could not reopen the generated USDZ package for audit")
    world = stage.GetDefaultPrim()
    assembly_prims = [prim for prim in world.GetChildren() if prim.GetChild("rigid_bodies")]
    if len(assembly_prims) != 1:
        raise RuntimeError("USDZ audit expected exactly one rigid-body assembly")
    if Usd.ModelAPI(assembly_prims[0]).GetKind() != Kind.Tokens.assembly:
        raise RuntimeError("USDZ audit assembly prim is not kind=assembly")

    expected_bodies = {body.name for body in resolved.rigid_bodies}
    expected_shapes = {
        (body.name, shape.name) for body in resolved.rigid_bodies for shape in body._iter_shapes()
    }
    expected_joints = {item.joint.name: item for item in resolved.joints}
    body_prims: dict[str, Usd.Prim] = {}
    joint_prims: dict[str, Usd.Prim] = {}
    found_shapes: set[tuple[str, str]] = set()
    source_meshes: dict[tuple[str, str], trimesh.Trimesh] = {}
    source_points: list[np.ndarray] = []
    exported_points: list[np.ndarray] = []
    triangle_count = 0
    normal_meshes = 0
    material_bindings = 0
    transforms = resolved.world_transforms()
    for body in resolved.rigid_bodies:
        for shape in body._iter_shapes():
            mesh = geometry_to_trimesh(shape.geometry, mesh_tolerance).copy()
            mesh.apply_transform(transforms[body.name])
            source_meshes[(body.name, shape.name)] = mesh
            source_points.append(np.asarray(mesh.vertices, dtype=np.float64))

    cache = UsdGeom.XformCache(  # pyright: ignore[reportAttributeAccessIssue]
        Usd.TimeCode.Default()  # pyright: ignore[reportAttributeAccessIssue]
    )
    for prim in stage.Traverse():
        path_text = str(prim.GetPath())
        authored_name = _custom_string(prim, "name")
        if "/rigid_bodies/" in path_text and "/shapes/" not in path_text and authored_name:
            body_prims[authored_name] = prim
        if "/joints/" in path_text and authored_name:
            joint_prims[authored_name] = prim
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)  # pyright: ignore[reportCallIssue]
        body_name = _custom_string(prim.GetParent().GetParent(), "name")
        selector = (body_name, authored_name)
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
        if mesh.GetOrientationAttr().Get() != UsdGeom.Tokens.rightHanded:
            raise RuntimeError(f"USDZ audit found non-right-handed winding at {prim.GetPath()}")
        if UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("").GetTargets():
            material_bindings += 1
        source = source_meshes.get(selector)
        if source is None:
            raise RuntimeError(f"USDZ audit found an unexpected mesh {selector!r}")
        if len(source.faces) != len(faces) // 3:
            raise RuntimeError(f"USDZ audit triangle mismatch for {selector!r}")

    if set(body_prims) != expected_bodies:
        raise RuntimeError(
            f"USDZ audit rigid body mismatch: expected={sorted(expected_bodies)!r} "
            f"found={sorted(body_prims)!r}"
        )
    if found_shapes != expected_shapes:
        raise RuntimeError(
            f"USDZ audit shape mismatch: expected={sorted(expected_shapes)!r} "
            f"found={sorted(found_shapes)!r}"
        )
    if set(joint_prims) != set(expected_joints):
        raise RuntimeError(
            f"USDZ audit joint mismatch: expected={sorted(expected_joints)!r} "
            f"found={sorted(joint_prims)!r}"
        )

    for name, item in expected_joints.items():
        joint = item.joint
        prim = joint_prims[name]
        schema = UsdPhysics.Joint(prim)  # pyright: ignore[reportCallIssue]
        expected0 = [] if joint.body0 is WORLD else [body_prims[_body_name(joint.body0)].GetPath()]
        expected1 = [] if joint.body1 is WORLD else [body_prims[_body_name(joint.body1)].GetPath()]
        if list(schema.GetBody0Rel().GetTargets()) != expected0:
            raise RuntimeError(f"USDZ audit body0 relationship mismatch for joint {name!r}")
        if list(schema.GetBody1Rel().GetTargets()) != expected1:
            raise RuntimeError(f"USDZ audit body1 relationship mismatch for joint {name!r}")
        if bool(schema.GetExcludeFromArticulationAttr().Get()) != item.exclude_from_articulation:
            raise RuntimeError(f"USDZ audit exclusion mismatch for joint {name!r}")
        if _custom_string(prim, "jointType") != _graph_joint_type(joint):
            raise RuntimeError(f"USDZ audit joint type mismatch for {name!r}")
        _expect_vector_attr(prim, "frame0:xyz", joint.frame0.xyz, joint_name=name)
        _expect_vector_attr(prim, "frame0:rpy", joint.frame0.rpy, joint_name=name)
        _expect_vector_attr(prim, "frame1:xyz", joint.frame1.xyz, joint_name=name)
        _expect_vector_attr(prim, "frame1:rpy", joint.frame1.rpy, joint_name=name)
        expected_schema = (
            UsdPhysics.FixedJoint
            if joint.is_fixed
            else UsdPhysics.RevoluteJoint
            if joint.is_revolute
            else UsdPhysics.PrismaticJoint
            if joint.is_prismatic
            else UsdPhysics.Joint
        )
        if not prim.IsA(expected_schema):
            raise RuntimeError(f"USDZ audit native schema mismatch for joint {name!r}")

    expected_roots = {item.articulation.root.name for item in resolved.articulations}
    found_roots = {
        _custom_string(prim, "name")
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    }
    if found_roots != expected_roots:
        raise RuntimeError(
            f"USDZ audit articulation root mismatch: expected={sorted(expected_roots)!r} "
            f"found={sorted(found_roots)!r}"
        )

    expected_materials = sum(
        shape.display_material is not None
        for body in resolved.rigid_bodies
        for shape in body._iter_shapes()
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
    return AssemblyExportAudit(
        rigid_body_count=len(body_prims),
        shape_count=len(found_shapes),
        joint_count=len(joint_prims),
        articulation_count=len(resolved.articulations),
        triangle_count=triangle_count,
        bounds=package_bounds,
        meshes_with_normals=normal_meshes,
        material_bindings=material_bindings,
    )


def _custom_string(prim: Usd.Prim, name: str) -> str:
    value = prim.GetAttribute(f"articraft:{name}").Get()
    return "" if value is None else str(value)


def _expect_vector_attr(
    prim: Usd.Prim,
    name: str,
    expected: tuple[float, float, float],
    *,
    joint_name: str,
) -> None:
    value = prim.GetAttribute(f"articraft:{name}").Get()
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
    value = prim.GetAttribute(f"articraft:{name}").Get()
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

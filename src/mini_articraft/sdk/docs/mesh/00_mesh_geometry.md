# Mesh geometry and solid builders

Use this page for direct triangle mesh editing, trimesh conversion, build123d
conversion, primitive solids, lathes, simple lofts, and extrusions.

All coordinates and lengths use meters. All angle arguments use radians. Mesh
faces are triangles with zero based vertex indices.

Import these names from the public SDK.

This page documents `BoxGeometry`, `CapsuleGeometry`, `ConeGeometry`,
`CylinderGeometry`, `DomeGeometry`, `ExtrudeGeometry`,
`ExtrudeWithHolesGeometry`, `LatheGeometry`, `LoftGeometry`, `MeshGeometry`,
`RoundedBoxGeometry`, `SphereGeometry`, `SuperellipsoidGeometry`,
`TorusGeometry`, and `build123d_to_mesh`.

```python
from mini_articraft.sdk import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    DomeGeometry,
    ExtrudeGeometry,
    ExtrudeWithHolesGeometry,
    LatheGeometry,
    LoftGeometry,
    MeshGeometry,
    RoundedBoxGeometry,
    SphereGeometry,
    SuperellipsoidGeometry,
    TorusGeometry,
)
from mini_articraft.sdk.mesh import build123d_to_mesh
```

## Choose a geometry type

Use build123d for exact solid modeling, cuts, fillets, and topology based work.
Add the resulting build123d shape directly to a part when no mesh operation is
needed.

Use `MeshGeometry` when you need direct vertex editing, a procedural mesh
builder, a mesh boolean, or a mesh repair helper. Use
`build123d_to_mesh(...)` only when a build123d shape must enter one of those
mesh workflows.

## MeshGeometry

```python
MeshGeometry(
    vertices: list[tuple[float, float, float]] = [],
    faces: list[tuple[int, int, int]] = [],
    normal_crease_angle: float | None = None,
)
```

`vertices` and `faces` are mutable lists. You may edit them directly. Call
`validate()` after direct edits and before depending on the result.
Each constructor call receives fresh empty lists. The displayed defaults are
not shared between mesh instances.

Validation applies these rules.

- Every vertex must contain three values that can be converted to finite
  floats. Constructor inputs are normalized to floats. After direct list
  edits, store numeric values because transforms perform numeric arithmetic on
  the stored entries.
- Every face must contain three integer indices.
- The three indices in one face must be distinct.
- Every index must refer to an existing vertex.

An empty `MeshGeometry()` is valid. It has no bounds and is not watertight.

### Properties and conversion

```python
geometry.validate() -> None
geometry.bounds -> tuple[tuple[float, float, float], tuple[float, float, float]]
geometry.is_watertight -> bool

MeshGeometry.from_trimesh(
    mesh: trimesh.Trimesh,
    *,
    process: bool = False,
) -> MeshGeometry

geometry.to_trimesh(*, process: bool = False) -> trimesh.Trimesh
geometry.to_obj() -> str
geometry.set_normal_crease(angle: float) -> MeshGeometry
geometry.use_smooth_normals() -> MeshGeometry
geometry.use_hard_normals() -> MeshGeometry
```

`bounds` returns `(minimum, maximum)` in mesh coordinates. It raises
`ValidationError` when the mesh has no vertices.

`is_watertight` asks trimesh whether the triangles form a closed surface. A
mesh can pass `validate()` and still be open or nonmanifold.

`from_trimesh(...)` copies the first three vertex properties and all triangle
indices. With `process=True`, it first asks trimesh to process and validate a
copy. `to_trimesh(...)` creates a new trimesh value. With `process=False`, it
does not merge or repair the mesh.

`to_obj()` returns OBJ text. It does not write a file or create a managed asset.

`normal_crease_angle` controls how the USDZ exporter blends normals across a
shared vertex. The angle uses radians and must be from zero through pi. Faces
whose angle is larger than this value keep separate corner normals. The
exporter uses 45 degrees when the value is `None`.

`set_normal_crease(...)` sets an explicit threshold. `use_smooth_normals()`
blends every shared vertex. `use_hard_normals()` keeps a separate normal for
every triangle corner. These methods change shading, not vertices or faces.

### Direct editing

```python
geometry.add_vertex(x: float, y: float, z: float) -> int
geometry.add_face(a: int, b: int, c: int) -> None
geometry.copy() -> MeshGeometry
geometry.merge(other: MeshGeometry) -> MeshGeometry
```

`add_vertex(...)` returns the new zero based index. `add_face(...)` validates
the indices before appending the face.

`copy()` returns an independent mesh with copied lists. `merge(...)` appends
the other mesh and adjusts its face indices. It mutates the receiver, returns
the receiver, and does not change the other mesh. A merge
does not perform a boolean union. The resulting mesh can contain separate
components or intersecting triangles.

```python
mesh = MeshGeometry()
a = mesh.add_vertex(0.0, 0.0, 0.0)
b = mesh.add_vertex(0.1, 0.0, 0.0)
c = mesh.add_vertex(0.0, 0.1, 0.0)
mesh.add_face(a, b, c)
```

### Transforms

```python
geometry.translate(dx: float, dy: float, dz: float) -> MeshGeometry
geometry.scale(
    sx: float,
    sy: float | None = None,
    sz: float | None = None,
) -> MeshGeometry
geometry.rotate(
    axis: tuple[float, float, float],
    angle: float,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MeshGeometry
geometry.rotate_x(angle: float) -> MeshGeometry
geometry.rotate_y(angle: float) -> MeshGeometry
geometry.rotate_z(angle: float) -> MeshGeometry
```

Every transform mutates the mesh and returns the same mesh so calls can be
chained. Call `copy()` first when you need to preserve the source.

If `sy` or `sz` is omitted, `scale(...)` uses `sx` for that axis. Scale values
must be finite and nonzero. A scale that reverses orientation also reverses
face winding.

`rotate(...)` normalizes `axis` and applies the right hand rule around the
given `origin`. The axis must be nonzero. The angle must be finite.

```python
base = BoxGeometry((0.20, 0.08, 0.04))
raised = base.copy().scale(0.8, 0.8, 1.2).rotate_z(0.35).translate(0.0, 0.0, 0.05)
base.merge(raised)
```

## Build123d conversion

```python
build123d_to_mesh(
    shape: build123d.Shape,
    *,
    tolerance: float = 0.0001,
    angular_tolerance: float = 0.1,
) -> MeshGeometry
```

The conversion tessellates the current shape, including any build123d `Pos`,
`Rot`, or `Location` already applied to it. The linear tolerance uses meters.
The angular tolerance uses radians. Both values must be finite and positive.

The input must be a valid build123d `Shape` that produces triangles. The
helper processes the tessellated mesh, removes unused vertices, and fixes face
normals. It raises `TypeError`, `ValueError`, or `ValidationError` for an
unsupported, invalid, or empty result.

```python
from build123d import Box, Pos

placed = Pos(X=0.20, Z=0.05) * Box(0.10, 0.08, 0.04)
mesh = build123d_to_mesh(placed)
```

Do not convert a build123d shape just to add it to a part. `Part.add(...)`
accepts the build123d shape directly and preserves its location.

## Primitive builders

All primitive builders return `MeshGeometry`. The default closed forms are
watertight. Segment counts control smoothness. Increasing them creates more
vertices and faces.

### BoxGeometry

```python
BoxGeometry(size: tuple[float, float, float])
```

`size` contains the full X, Y, and Z extents. Each value must be positive. The
box is centered at the origin.

### RoundedBoxGeometry

```python
RoundedBoxGeometry(
    size: tuple[float, float, float],
    radius: float,
    *,
    tolerance: float = 0.0005,
    angular_tolerance: float = 0.08,
)
```

This builder creates flat faces with tangent rounded edges and corners. `size`
contains the full centered extents. `radius` must be positive and less than
half the smallest size.

The two tolerances control the final mesh density. Linear tolerance uses
meters. Angular tolerance uses radians. Smaller values follow the rounded
surface more closely and produce more triangles.

```python
housing = RoundedBoxGeometry(
    (0.12, 0.075, 0.028),
    0.006,
    tolerance=0.0006,
    angular_tolerance=0.08,
)
```

### CylinderGeometry

```python
CylinderGeometry(
    radius: float,
    height: float,
    *,
    radial_segments: int = 24,
    closed: bool = True,
)
```

The cylinder is centered at the origin and follows the Z axis. Its Z range is
`-height / 2` through `height / 2`. Radius and height must be positive.
`radial_segments` is clamped to at least 3. With `closed=False`, only the side
wall is present.

### ConeGeometry

```python
ConeGeometry(
    radius: float,
    height: float,
    *,
    radial_segments: int = 24,
    closed: bool = True,
)
```

The cone is centered along Z. Its base is at `z = -height / 2` and its tip is
at `z = height / 2`. Radius and height must be positive. The segment count is
clamped to at least 3. With `closed=False`, the base cap is removed.

### SphereGeometry

```python
SphereGeometry(
    radius: float,
    *,
    width_segments: int = 24,
    height_segments: int = 16,
)
```

The sphere is centered at the origin. Radius must be positive. Width segments
are clamped to at least 8. Height segments are clamped to at least 4.

### SuperellipsoidGeometry

```python
SuperellipsoidGeometry(
    radius: float | tuple[float, float, float],
    *,
    latitude_exponent: float = 1.0,
    longitude_exponent: float = 1.0,
    radial_segments: int = 48,
    height_segments: int = 24,
)
```

A scalar radius creates an equal sided form. Three values set the X, Y, and Z
radii. An exponent of `1.0` creates an ellipsoid. Values below one make the
form fuller and boxier. Values above one make it more pinched. Separate
latitude and longitude exponents let the vertical and horizontal shape change
independently.

All radii and exponents must be finite and positive. `radial_segments` must be
at least 8. `height_segments` must be at least 4. Increase both values for a
smoother surface.

```python
soft_housing = SuperellipsoidGeometry(
    (0.06, 0.04, 0.018),
    latitude_exponent=0.55,
    longitude_exponent=0.45,
    radial_segments=64,
    height_segments=32,
)
```

### DomeGeometry

```python
DomeGeometry(
    radius: float | tuple[float, float, float],
    *,
    radial_segments: int = 24,
    height_segments: int = 12,
    closed: bool = True,
)
```

A scalar radius creates a hemisphere. Three radii create an ellipsoid dome
with separate X, Y, and Z radii. Every radius must be positive. The base lies
on `z = 0` and the dome extends toward positive Z. `closed=True` adds the flat
base. Radial segments are clamped to at least 3. Height segments are clamped
to at least 1.

### CapsuleGeometry

```python
CapsuleGeometry(
    radius: float,
    length: float,
    *,
    radial_segments: int = 24,
    height_segments: int = 8,
)
```

The capsule follows the Z axis and is centered at the origin. `length` is the
length of the cylindrical middle section. The full end to end length is
`length + 2 * radius`. Radius must be positive and length must be nonnegative.
A zero length creates a sphere.

### TorusGeometry

```python
TorusGeometry(
    radius: float,
    tube: float,
    *,
    radial_segments: int = 16,
    tubular_segments: int = 32,
)
```

The torus lies in the XY plane and is centered at the origin. `radius` is the
distance from the origin to the center of the tube. `tube` is the tube radius.
Both must be positive. Each segment count is clamped to at least 3.

## LatheGeometry

```python
LatheGeometry(
    profile: Iterable[tuple[float, float]],
    *,
    segments: int = 32,
    closed: bool = True,
    angle: float = 2 * pi,
    start_angle: float = 0.0,
    axis: str = "z",
    cap_ends: bool = True,
)
```

Each profile point is `(radius, z)`. The helper revolves the profile around the
selected axis. `axis` can be `"x"`, `"y"`, or `"z"`. Radius values must be
nonnegative. Segments are clamped to at least 3.

`angle` must be greater than zero and at most a full turn. `start_angle`
rotates the first section around the selected axis. A partial revolution of a
closed profile gets flat end caps when `cap_ends=True`. Partial end caps require
a closed profile.

With `closed=True`, the profile must contain at least three distinct points and
must enclose a nonzero area in the radius and Z plane. The helper closes the
profile before revolving it. With `closed=False`, the profile needs at least
two distinct points and produces an open surface.

Use `from_shell_profiles(...)` for a thin revolved shell.

```python
LatheGeometry.from_shell_profiles(
    outer_profile: Iterable[tuple[float, float]],
    inner_profile: Iterable[tuple[float, float]],
    *,
    segments: int = 32,
    start_cap: str = "flat",
    end_cap: str = "flat",
    lip_samples: int = 6,
) -> LatheGeometry
```

The outer and inner profiles each need at least two points. Their authored
order defines the start and end. Cap modes are `"flat"` and `"round"`.
`lip_samples` controls the curved connector used by a round cap and is clamped
to at least 2.

```python
cup = LatheGeometry.from_shell_profiles(
    outer_profile=[(0.045, 0.0), (0.050, 0.09)],
    inner_profile=[(0.040, 0.004), (0.044, 0.085)],
    end_cap="round",
    segments=48,
)
```

## LoftGeometry

```python
LoftGeometry(
    profiles: Iterable[Iterable[tuple[float, float, float]]],
    *,
    cap: bool = True,
    closed: bool = True,
    interpolation: str = "linear",
    samples_per_span: int = 1,
    close_path: bool = False,
    parameterization: str = "uniform",
    tension: float = 0.0,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
)
```

This is the direct mesh loft. Use `section_loft(...)` when sections have
different point counts, need path offsets, need symmetry, or need repair.

At least two profiles are required. Every profile must have the same point
count. With `closed=True`, each profile needs at least three distinct points
and must enclose a nonzero planar area. Adjacent profile centers must be
different. The helper corrects reversed loop winding by comparing each profile
with the previous one, but corresponding points should still describe the same
feature around each profile.

With `closed=False`, profiles are treated as open polylines with at least two
points. Only adjacent points are connected. Caps are only added when both
`cap` and `closed` are true. Use planar first and last profiles when you need
reliable caps.

`interpolation="linear"` connects the authored profiles directly. Set
`interpolation="catmull_rom"` to fit a smooth path through corresponding
profile points. `samples_per_span` controls how many rings the helper creates
from one authored profile to the next. It must be at least one. Smooth
interpolation needs a value greater than one to add a visible curve between
authored profiles.

Smooth interpolation passes through every authored profile. It can extend
outside the straight span when profile size or position changes sharply. Add
another authored profile or use linear interpolation when the surface must
stay within a strict boundary.

For Catmull Rom spans, `parameterization` can be `"uniform"`, `"chord"`, or
`"centripetal"`. Chord and centripetal modes account for the distance between
profile centers. Centripetal mode is a good starting point when the profile
spacing is uneven. `tension` ranges from `0.0` to `1.0` and reduces tangent
strength as it increases.

The default `cap_style="flat"` triangulates each end profile. `"round"` grows
smaller rings toward a rounded tip. `cap_segments` controls the number of rings.
`cap_length` sets the distance from the end profile to the tip. When it is
omitted, the loft uses the largest profile radius at that end.

With `close_path=True`, the helper connects the final profile back to the first
profile and does not add caps. A closed path needs at least three profiles.
This supports ring forms whose profile changes around a complete loop.

```python
loft = LoftGeometry(
    profiles=[
        [(-0.05, -0.04, 0.0), (0.05, -0.04, 0.0), (0.05, 0.04, 0.0), (-0.05, 0.04, 0.0)],
        [(-0.03, -0.02, 0.12), (0.03, -0.02, 0.12), (0.03, 0.02, 0.12), (-0.03, 0.02, 0.12)],
    ],
    interpolation="catmull_rom",
    samples_per_span=6,
    parameterization="centripetal",
    tension=0.1,
    cap_style="round",
    cap_segments=8,
    cap_length=0.015,
)
```

## ExtrudeGeometry

```python
ExtrudeGeometry(
    profile: Iterable[tuple[float, float]],
    height: float,
    *,
    cap: bool = True,
    center: bool = True,
    closed: bool = True,
)

ExtrudeGeometry.centered(
    profile,
    height,
    *,
    cap: bool = True,
    closed: bool = True,
) -> ExtrudeGeometry

ExtrudeGeometry.from_z0(
    profile,
    height,
    *,
    cap: bool = True,
    closed: bool = True,
) -> ExtrudeGeometry
```

The profile is a two dimensional loop in XY. Height must be positive. A
centered extrusion spans `-height / 2` through `height / 2`. With
`center=False`, or with `from_z0(...)`, it spans `0` through `height`.

The helper accepts either clockwise or counterclockwise profile input and
normalizes it. The profile must contain at least three distinct points and
must have nonzero area. Caps are only added when both `cap` and `closed` are
true. With `closed=False`, the last profile point is not connected back to the
first, so the result is an open strip.

## ExtrudeWithHolesGeometry

```python
ExtrudeWithHolesGeometry(
    outer_profile: Iterable[tuple[float, float]],
    hole_profiles: Iterable[Iterable[tuple[float, float]]],
    height: float,
    *,
    cap: bool = True,
    center: bool = True,
    closed: bool = True,
)
```

The outer loop and every hole loop are in XY. Each loop needs at least three
distinct points and nonzero area. The helper normalizes loop winding. For a
capped solid, every hole must lie inside the outer loop and the profiles must
be valid simple polygons. Manifold performs the cap triangulation.

Height and centering follow `ExtrudeGeometry`. If `hole_profiles` is empty, the
helper delegates to `ExtrudeGeometry`.

```python
plate = ExtrudeWithHolesGeometry(
    outer_profile=[(-0.10, -0.06), (0.10, -0.06), (0.10, 0.06), (-0.10, 0.06)],
    hole_profiles=[
        [(-0.02, -0.02), (-0.02, 0.02), (0.02, 0.02), (0.02, -0.02)],
    ],
    height=0.008,
)
```

## Related references

- Read [profiles and curve sampling](10_profiles.md) for profile generation and
  curve sampling.
- Read [wires and sweeps](20_wires_and_sweeps.md) for paths, pipes, and tube
  networks.
- Read [section lofts and repair](30_section_lofts.md) for the section loft and
  repair surface.
- Read [booleans and shells](40_booleans_and_shells.md) for Manifold booleans,
  openings, and shell partitioning.

# Wires, pipes, and sweeps

Use this page for swept profiles, circular wires, smooth tubes, authored paths,
arcs, and tube networks.

All coordinates, radii, and tolerances use meters. All angles use radians.
Path points are three dimensional values in the final mesh coordinate frame.

This page documents `SweepSection`, `SweepGeometry`, `PipeGeometry`,
`ArcPipeGeometry`, `WirePath`, `WirePolylineGeometry`, `wire_from_points`,
`tube_from_spline_points`, `sweep_profile_along_spline`, and
`tube_network_from_paths`.

```python
from articraft.sdk import (
    ArcPipeGeometry,
    PipeGeometry,
    SweepGeometry,
    WirePolylineGeometry,
)
from articraft.sdk.mesh import (
    SweepSection,
    WirePath,
    sweep_profile_along_spline,
    tube_from_spline_points,
    tube_network_from_paths,
    wire_from_points,
)
```

## Choose a helper

- Use `WirePolylineGeometry` or `wire_from_points(...)` for straight segments
  with optional bevels or rounded corners.
- Use `tube_from_spline_points(...)` for a smooth circular tube fitted through
  control points.
- Use `sweep_profile_along_spline(...)` for a smooth rail with a custom two
  dimensional profile.
- Use `PipeGeometry` when you already have the final sampled path.
- Use `ArcPipeGeometry` for one exact circular arc.
- Use `tube_network_from_paths(...)` when several circular paths must be joined
  into one mesh.

Use `SweepSection` with any single path sweep when the profile should change
along the path. A section can change the profile shape, size, local rotation,
or local offset.

## SweepSection

```python
SweepSection(
    position: float,
    profile: Sequence[tuple[float, float]] | None = None,
    scale: float | Sequence[float] = 1.0,
    rotation: float = 0.0,
    offset: Sequence[float] = (0.0, 0.0),
    interpolation: str | None = None,
    tension: float | None = None,
)
```

`position` is the normalized distance along the complete path. Zero is the
start and one is the end. The sweep measures distance along the path, so a
section at `0.5` lies halfway along its length rather than at the middle path
index.

Each field changes the local two dimensional profile at that position.

- `profile` replaces the base profile. When omitted, the section uses the base
  profile passed to the sweep.
- `scale` accepts one positive value for uniform scale or two positive values
  for separate profile X and Y scale.
- `rotation` rotates the profile in its local plane. Angles use radians.
- `offset` moves the profile in its local X and Y directions.
- `interpolation` can override the outgoing span with `"linear"` or
  `"catmull_rom"`.
- `tension` can override the outgoing span tension from zero through one. Zero
  gives the fullest smooth curve. One eases into and out of the section without
  using neighboring section slopes.

The sweep applies scale first, then rotation, then offset. Profiles may have
different point counts. The sweep samples them to one shared count and aligns
their point order before blending. Every authored section position creates an
actual path ring, even when it lies between two input path points.

Sweep constructors also accept `section_interpolation` and `section_tension`.
These values apply to spans whose starting `SweepSection` does not provide an
override. Linear interpolation connects corresponding profile points directly.
Catmull Rom interpolation uses the neighboring sections to make the profile
change smoothly.

Section positions must be unique. When position zero or one is omitted, the
sweep inserts the unchanged base profile at that end. A closed path requires
the final profile to match the first profile. When no final section is given,
the sweep uses the first result at the seam.

```python
blade = PipeGeometry(
    rounded_rect_profile(0.012, 0.006, 0.001),
    [(0.0, 0.0, 0.0), (0.04, 0.0, 0.02), (0.10, 0.0, 0.01)],
    cap=True,
    sections=(
        SweepSection(0.0, scale=0.7),
        SweepSection(0.45, scale=(1.4, 0.8), rotation=0.25),
        SweepSection(1.0, scale=0.25, rotation=0.6),
    ),
)
```

## Path frames

Single path sweeps accept `frame_mode="parallel_transport"` or
`frame_mode="fixed_up"`.

The default parallel transport mode carries the initial profile orientation
along the path while limiting unwanted roll. On a closed path, the sweep
spreads the remaining frame rotation around the complete loop and shares the
first profile ring at the seam.

Fixed up mode uses `up_hint` at every path point. This is useful when a flat
profile should stay upright as a path turns. When the path tangent becomes
parallel to the up hint, the sweep continues from the previous frame at that
point.

## SweepGeometry

```python
SweepGeometry(
    profile: Iterable[tuple[float, float]],
    path: Iterable[tuple[float, float, float]],
    *,
    cap: bool = False,
    closed: bool = True,
    path_closed: bool = False,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
)
```

`SweepGeometry` is a short form of `PipeGeometry`. The path needs at least two
distinct finite points. Its path closure, frame, and section behavior match
`PipeGeometry`.

`closed` describes the profile, not the path. With `closed=True`, the profile
must contain at least three distinct points and have nonzero area. With
`closed=False`, it is an open polyline with at least two points. Caps are only
added when both `cap` and `closed` are true.

## PipeGeometry

```python
PipeGeometry(
    profile: Iterable[tuple[float, float]],
    path: Iterable[tuple[float, float, float]],
    *,
    cap: bool = False,
    closed: bool = True,
    path_closed: bool = False,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
)
```

The helper sweeps a two dimensional profile along a sampled path. It transports
a local frame along the path to reduce sudden profile roll.

At the first path point, profile X follows the local normal and profile Y
follows the local binormal. The helper derives those directions from the first
path tangent and `up_hint`. If the up hint is almost parallel to the tangent,
the helper selects a different fallback direction. A different up hint can
change the profile roll without changing the path.

The path needs at least two distinct finite points. Consecutive duplicate path
points are removed. The up hint must be nonzero.

With `path_closed=True`, the helper removes a repeated final path point when
needed, joins the last ring to the shared first ring, and disables caps. A
closed path needs at least three distinct points. With an open path, `cap=True`
adds a cap at each end when the profile is closed.

Set `max_segment_length` to add path rings until no centerline segment exceeds
that length. This controls surface density without changing the path shape.

`cap_style="flat"` triangulates the end profiles. `cap_style="round"` extends
the profile along the endpoint tangent while scaling it into one tip. The
default round cap length is the largest profile radius. Set `cap_length` to use
a different length. `cap_segments` controls the number of curved spans and must
be at least two.

Flat caps triangulate the profile boundary. Concave profiles are supported
when their boundary does not cross itself.

```python
rail_profile = [
    (-0.006, -0.003),
    (0.006, -0.003),
    (0.006, 0.003),
    (-0.006, 0.003),
]
rail = PipeGeometry(
    rail_profile,
    [(0.0, 0.0, 0.0), (0.05, 0.0, 0.03), (0.10, 0.0, 0.0)],
    cap=True,
)
```

## ArcPipeGeometry

```python
ArcPipeGeometry(
    profile: Iterable[tuple[float, float]],
    *,
    start_point: tuple[float, float, float],
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    angle: float,
    arc_segments: int = 20,
    cap: bool = False,
    closed: bool = True,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
)
```

This helper samples a circular arc with `sample_arc_3d(...)`, then passes that
path to `PipeGeometry`. `normal` defines the rotation axis through `center`.
The angle follows the right hand rule around that normal.

The start point must differ from the center. The radius direction must not be
collinear with the normal. Arc segments are clamped to at least 2. Profile,
cap, frame, and section behavior match `PipeGeometry`.

```python
elbow = ArcPipeGeometry(
    rail_profile,
    start_point=(0.10, 0.0, 0.0),
    center=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
    angle=1.57079632679,
    arc_segments=24,
    cap=True,
)
```

## WirePath

`WirePath` stores an editable list of path points. Its methods mutate the path
and return the same path so calls can be chained. Duplicate consecutive points
are skipped.

```python
WirePath(start: tuple[float, float, float])

WirePath.from_points(
    points: Iterable[tuple[float, float, float]],
) -> WirePath

path.line_to(point: tuple[float, float, float]) -> WirePath
path.line_by(dx: float, dy: float, dz: float) -> WirePath
path.bezier_to(
    control1: tuple[float, float, float],
    control2: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    samples: int = 12,
) -> WirePath
path.arc(
    *,
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    angle: float,
    segments: int = 16,
) -> WirePath
path.extend(points: Iterable[tuple[float, float, float]]) -> WirePath
path.to_points() -> list[tuple[float, float, float]]
```

`from_points(...)` needs at least one point. `line_by(...)` adds an offset to
the current endpoint. `bezier_to(...)` makes one cubic Bezier segment from the
current endpoint. `arc(...)` starts at the current endpoint and uses the same
rules as `sample_arc_3d(...)`. `to_points()` returns a copy of the point list.

```python
path = (
    WirePath((-0.06, 0.0, 0.0))
    .line_by(0.02, 0.0, 0.0)
    .bezier_to(
        (-0.02, 0.0, 0.05),
        (0.02, 0.0, 0.05),
        (0.04, 0.0, 0.0),
        samples=16,
    )
    .line_to((0.06, 0.0, 0.0))
)
```

## Polyline wires

The class and function below have the same behavior. The function returns a
`MeshGeometry` and the class returns a `WirePolylineGeometry`.

```python
WirePolylineGeometry(
    points: Iterable[tuple[float, float, float]],
    *,
    radius: float,
    radial_segments: int = 16,
    closed_path: bool = False,
    cap_ends: bool = False,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
)

wire_from_points(
    points: Iterable[tuple[float, float, float]],
    *,
    radius: float,
    radial_segments: int = 16,
    closed_path: bool = False,
    cap_ends: bool = False,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry
```

The radius must be positive. `radial_segments` must be at least 6. The input
needs at least two distinct finite points. After that first check, segments
shorter than `min_segment_length` are removed. Use a positive minimum length
that is small relative to the visible wire.

Corner modes are `"miter"`, `"bevel"`, and `"fillet"`.

- Miter keeps the original corner point.
- Bevel replaces the corner with two trimmed points.
- Fillet adds a sampled quadratic bend between two trimmed points.

`corner_radius` is limited to 45 percent of each adjacent segment. A zero
corner radius leaves the centerline unchanged for every mode. `corner_segments`
is clamped to at least 2 when a fillet is built.

With `closed_path=True`, the helper closes the path and ignores `cap_ends`.
With an open path, `cap_ends=True` creates a watertight tube. The default
`cap_ends=False` leaves both ends open.

Sweep sections scale the circular profile by default. A two value scale can
turn it into an ellipse. A section may also provide a different profile.

```python
guard = wire_from_points(
    [(-0.08, 0.0, 0.0), (0.0, 0.0, 0.06), (0.08, 0.0, 0.0)],
    radius=0.003,
    radial_segments=16,
    cap_ends=True,
    corner_mode="fillet",
    corner_radius=0.012,
    corner_segments=8,
)
```

## Smooth circular tubes

```python
tube_from_spline_points(
    points: Iterable[tuple[float, float, float]],
    *,
    radius: float,
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    radial_segments: int = 16,
    cap_ends: bool = True,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry
```

The helper samples the requested spline, then passes the sampled path to
`wire_from_points(...)` with miter corners. Since the sampled points already
follow a smooth curve, the miter setting preserves that curve.

Supported spline names are `"catmull_rom"` and `"bezier"`. Catmull Rom fits
through the input points and uses `alpha`. Bezier input must contain `3 * n + 1`
control points. A closed Bezier chain must already end where it starts.

`closed_spline=True` closes the tube and disables end caps. With an open
spline, `cap_ends=True` creates closed ends. Radius, radial segment, up hint,
frame, section, and minimum segment behavior match the polyline wire helper.

```python
handle = tube_from_spline_points(
    path.to_points(),
    radius=0.003,
    samples_per_segment=8,
    radial_segments=18,
    cap_ends=True,
)
```

Increase `samples_per_segment` when the centerline looks coarse. Increase
`radial_segments` when the circular section looks faceted.

## Smooth custom profile sweeps

```python
sweep_profile_along_spline(
    points: Iterable[tuple[float, float, float]],
    *,
    profile: Iterable[tuple[float, float]],
    samples_per_segment: int = 12,
    closed_spline: bool = False,
    spline: str = "catmull_rom",
    alpha: float = 0.5,
    cap_profile: bool = True,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    frame_mode: str = "parallel_transport",
    sections: Sequence[SweepSection] = (),
    section_interpolation: str = "linear",
    section_tension: float = 0.0,
    max_segment_length: float | None = None,
    cap_style: str = "flat",
    cap_segments: int = 6,
    cap_length: float | None = None,
    min_segment_length: float = 1e-6,
) -> MeshGeometry
```

Spline selection follows `tube_from_spline_points(...)`. The profile is a
closed two dimensional loop. `cap_profile=True` caps an open spline. A closed
spline never receives end caps.

Unlike `wire_from_points(...)`, this helper raises `ValueError` if the sampled
centerline contains a segment shorter than `min_segment_length`. Lower the
minimum only after checking that the control points do not contain accidental
duplicates.

The profile orientation follows the selected path frame. Use `up_hint` to
control its initial roll, or use fixed up mode to keep the profile aligned to
the hint along the complete path. Sweep sections use normalized distance along
the sampled spline.

```python
from articraft.sdk.mesh import rounded_rect_profile

trim = sweep_profile_along_spline(
    [(-0.05, 0.0, 0.0), (0.0, 0.02, 0.03), (0.05, 0.0, 0.0)],
    profile=rounded_rect_profile(0.006, 0.003, 0.0008),
    samples_per_segment=16,
    cap_profile=True,
)
```

## Tube networks

```python
tube_network_from_paths(
    paths: Iterable[Iterable[tuple[float, float, float]]],
    *,
    radius: float,
    radial_segments: int = 16,
    cap_ends: bool = True,
    corner_mode: str = "fillet",
    corner_radius: float = 0.0,
    corner_segments: int = 8,
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    min_segment_length: float = 1e-6,
    shared_node_radius: float | None = None,
) -> MeshGeometry
```

At least one path is required, and each path needs at least two distinct finite
points. Tube radius must be positive. `shared_node_radius` defaults to the tube
radius and may be zero to disable shared node spheres.

The helper builds one `WirePolylineGeometry` for every path. It counts point
occurrences across the complete input, including repeated points in one path.
When a point occurs more than once, it can add a sphere at that point. Points
are considered the same after rounding by `min_segment_length`.

With `cap_ends=True`, every path is capped and Manifold unions the tubes and
shared node spheres. Intersecting paths can form one watertight body. Disjoint
paths can remain separate bodies inside one mesh. If Manifold cannot form a
valid closed result, the helper raises `ValueError`.

With `cap_ends=False`, the helper skips the boolean union and merges the
triangle lists. The result normally has open ends and can contain separate
components.

```python
frame = tube_network_from_paths(
    [
        [(-0.06, 0.0, 0.0), (0.0, 0.0, 0.0), (0.06, 0.0, 0.0)],
        [(0.0, -0.06, 0.0), (0.0, 0.0, 0.0), (0.0, 0.06, 0.0)],
    ],
    radius=0.004,
    radial_segments=12,
    cap_ends=True,
    shared_node_radius=0.005,
)
```

## Validation and common failures

- A path with fewer than two distinct points raises `ValueError`.
- A zero up hint or a zero arc normal raises `ValueError`.
- An unknown frame mode raises `ValueError`.
- An invalid corner mode raises `ValueError`.
- A circular wire radius must be positive.
- A custom profile must contain enough distinct points for its `closed` mode.
- A network boolean requires closed input tubes, so keep `cap_ends=True` when
  you need a closed network solid.
- Sweep section positions must be unique and between zero and one.
- Sweep section scale values must be positive.
- A closed path requires matching section results at positions zero and one.

## Related references

- Read [profiles and curve sampling](10_profiles.md) for spline and arc
  sampling details.
- Read [booleans and shells](40_booleans_and_shells.md) for the Manifold rules
  used by closed tube networks.

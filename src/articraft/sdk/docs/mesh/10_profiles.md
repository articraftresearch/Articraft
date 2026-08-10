# Profiles and curve sampling

Use this page to create two dimensional profiles, sample two or three
dimensional curves, build side lofts, and resample side sections.

Coordinates and lengths use meters. Arc angles use radians. Profile helpers
return ordinary Python lists, so you can inspect or edit the points before
building a mesh.

This page documents `rounded_rect_profile`, `superellipse_profile`,
`sample_cubic_bezier_spline_2d`, `sample_cubic_bezier_spline_3d`,
`sample_catmull_rom_spline_2d`, `sample_catmull_rom_spline_3d`,
`sample_arc_3d`, `superellipse_side_loft`, `split_superellipse_side_loft`, and
`resample_side_sections`.

```python
from articraft.sdk.mesh import (
    resample_side_sections,
    rounded_rect_profile,
    sample_arc_3d,
    sample_catmull_rom_spline_2d,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_2d,
    sample_cubic_bezier_spline_3d,
    split_superellipse_side_loft,
    superellipse_profile,
    superellipse_side_loft,
)
```

## Profile conventions

A two dimensional profile is a sequence of `(x, y)` points. Mesh builders
close a profile when their `closed` argument is true, so you do not need to
repeat the first point at the end. Consecutive duplicate points are removed by
the builders.

The public profile generators on this page return centered counterclockwise
loops. Their first point is not repeated at the end.

Use finite coordinates and parameters. The curve samplers convert values to
floats but do not reject every `NaN` or infinite input before sampling. A later
`MeshGeometry` constructor rejects nonfinite vertices.

## Rounded rectangle profiles

```python
rounded_rect_profile(
    width: float,
    height: float,
    radius: float,
    *,
    corner_segments: int = 6,
) -> list[tuple[float, float]]
```

The result is centered at the origin in XY. Width and height must be positive.
Radius must be between zero and half of the smaller dimension. A zero radius
returns four rectangle corners. `corner_segments` is clamped to at least 1.

```python
profile = rounded_rect_profile(
    width=0.12,
    height=0.08,
    radius=0.01,
    corner_segments=8,
)
```

## Superellipse profiles

```python
superellipse_profile(
    width: float,
    height: float,
    exponent: float = 2.6,
    *,
    segments: int = 48,
) -> list[tuple[float, float]]
```

The result is centered at the origin in XY. Width, height, and exponent must be
finite and positive for a usable profile. The helper rejects zero and negative
values, but it does not reject `NaN` before sampling. An exponent of 2 creates
an ellipse. Larger values make the sides flatter and the corners squarer.
Segments are clamped to at least 12.

Use a segment count that matches the final object scale. More segments make a
smoother profile and create more loft or extrusion faces.

## Cubic Bezier sampling

```python
sample_cubic_bezier_spline_2d(
    control_points: Sequence[tuple[float, float]],
    *,
    samples_per_segment: int = 12,
) -> list[tuple[float, float]]

sample_cubic_bezier_spline_3d(
    control_points: Sequence[tuple[float, float, float]],
    *,
    samples_per_segment: int = 12,
) -> list[tuple[float, float, float]]
```

The control point count must be `3 * n + 1` for one or more cubic segments.
The first four points define the first segment. Each later segment reuses the
previous endpoint and adds three control points.

`samples_per_segment` is clamped to at least 2. The result includes the first
point and the final endpoint. A shared endpoint appears once between adjacent
segments.

This is an authored Bezier chain. It does not fit a curve through an arbitrary
point list.

```python
curve = sample_cubic_bezier_spline_3d(
    [
        (0.00, 0.00, 0.00),
        (0.03, 0.00, 0.04),
        (0.07, 0.00, 0.04),
        (0.10, 0.00, 0.00),
    ],
    samples_per_segment=16,
)
```

## Catmull Rom sampling

```python
sample_catmull_rom_spline_2d(
    points: Sequence[tuple[float, float]],
    *,
    samples_per_segment: int = 12,
    closed: bool = False,
    alpha: float = 0.5,
) -> list[tuple[float, float]]

sample_catmull_rom_spline_3d(
    points: Sequence[tuple[float, float, float]],
    *,
    samples_per_segment: int = 12,
    closed: bool = False,
    alpha: float = 0.5,
) -> list[tuple[float, float, float]]
```

An open spline needs at least two distinct points. A closed spline needs at
least three distinct points. Consecutive duplicates are removed. If a closed
input repeats its first point at the end, the helper removes that duplicate
before sampling.

`alpha` must be from 0 through 1. The default value of 0.5 is a good starting
point for most authored paths. `samples_per_segment` is clamped to at least 2.

An open result includes both endpoints. A closed result repeats its first
sample at the end so path builders can close the final segment.

With exactly two open points, the helper returns straight linear samples.

```python
centerline = sample_catmull_rom_spline_3d(
    [
        (-0.05, 0.00, 0.00),
        (-0.02, 0.00, 0.04),
        (0.02, 0.00, 0.04),
        (0.05, 0.00, 0.00),
    ],
    samples_per_segment=12,
)
```

## Arc sampling

```python
sample_arc_3d(
    *,
    start_point: tuple[float, float, float],
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    angle: float,
    segments: int = 16,
) -> list[tuple[float, float, float]]
```

The helper rotates `start_point` around the axis that passes through `center`
and follows `normal`. It uses the right hand rule. A positive angle turns in
the positive direction around the normal.

The start point must differ from the center. The radius direction must not be
collinear with the normal. The normal must be nonzero. Segments are clamped to
at least 2. The result contains `segments + 1` points, including both ends.

```python
quarter_arc = sample_arc_3d(
    start_point=(0.10, 0.00, 0.00),
    center=(0.00, 0.00, 0.00),
    normal=(0.00, 0.00, 1.00),
    angle=1.57079632679,
    segments=20,
)
```

Use `ArcPipeGeometry` when you want to sample the arc and sweep a profile in
one call.

## Superellipse side lofts

```python
superellipse_side_loft(
    sections: Sequence[tuple[float, float, float, float]],
    *,
    exponents: float | Sequence[float] = 2.8,
    segments: int = 56,
    cap: bool = True,
    closed: bool = True,
    min_height: float = 0.0001,
    min_width: float = 0.0001,
) -> MeshGeometry
```

Each section is `(y, z_min, z_max, width)`. The loft axis follows the authored
Y values. Each section is a superellipse in the XZ plane and is centered at
`x = 0` and at the midpoint of `z_min` and `z_max`.

At least two sections are required. The helper keeps the input order, so author
the sections in the order that the surface should follow. A scalar `exponents`
value applies to all sections. A sequence must have one value for every
section. Exponents below 0.2 are clamped to 0.2.

`segments` is clamped to at least 12. A section height smaller than
`min_height` is expanded around its Z center. A width smaller than `min_width`
is raised to that value. Use positive minimum values.

The `cap` and `closed` arguments pass through to `LoftGeometry`. Caps are only
created for a closed loft.

```python
housing = superellipse_side_loft(
    [
        (-0.10, -0.03, 0.03, 0.12),
        (0.00, -0.05, 0.05, 0.16),
        (0.12, -0.035, 0.035, 0.10),
    ],
    exponents=(2.4, 3.2, 2.6),
    segments=48,
)
```

## Split a side loft

```python
split_superellipse_side_loft(
    sections: Sequence[tuple[float, float, float, float]],
    *,
    split_y: float,
    exponents: float | Sequence[float] = 2.8,
    **loft_options,
) -> tuple[
    MeshGeometry,
    MeshGeometry,
    tuple[float, float, float, float],
]
```

This helper sorts sections by Y, inserts or reuses a section at `split_y`, and
builds one loft on each side. It returns `(lower_y_mesh, upper_y_mesh, seam)`.
The seam tuple uses the same `(y, z_min, z_max, width)` format.

At least three sections are required. `split_y` must lie strictly between the
smallest and largest Y values. If no section already lies at the split, the
helper linearly interpolates the seam and its exponent.

Extra keyword arguments pass to `superellipse_side_loft(...)`, so you may set
`segments`, `cap`, `closed`, `min_height`, or `min_width`.

```python
rear, front, seam = split_superellipse_side_loft(
    sections,
    split_y=0.03,
    segments=48,
)
```

## Resample side sections

```python
resample_side_sections(
    sections: Sequence[tuple[float, float, float, float]],
    *,
    samples_per_span: int = 2,
    smooth_passes: int = 0,
    min_height: float = 0.0001,
    min_width: float = 0.0001,
) -> list[tuple[float, float, float, float]]
```

The helper sorts sections by Y. When two rows have the same Y, it replaces the
stored row with the pairwise average of that row and the next row. With more
than two duplicate rows, this repeated averaging is order dependent. It then
adds linear samples between each pair of remaining rows. `samples_per_span` is
clamped to at least 1.

Each smoothing pass keeps the Y values and endpoints fixed. It replaces the
interior Z and width values with a weighted average of neighboring rows.
Negative smoothing counts act like zero.

The result always orders each Z pair from low to high. A short height is
expanded around its center to `min_height`. Width is clamped to `min_width`.
Use positive minimum values.

For `n` unique input rows, the unsmoothed output has
`(n - 1) * samples_per_span + 1` rows after the minimum clamp on the sample
count.

```python
dense_sections = resample_side_sections(
    sections,
    samples_per_span=3,
    smooth_passes=1,
)
smooth_housing = superellipse_side_loft(dense_sections, segments=48)
```

## Related references

- Read [mesh geometry and solid builders](00_mesh_geometry.md) for extrusion
  and low level loft builders.
- Read [wires and sweeps](20_wires_and_sweeps.md) to turn sampled three
  dimensional curves into tubes and swept profiles.
- Read [section lofts and repair](30_section_lofts.md) for general section loops
  with optional path offsets and repair.

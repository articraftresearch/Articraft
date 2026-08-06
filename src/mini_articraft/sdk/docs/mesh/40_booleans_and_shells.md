# Booleans, openings, and shell partitions

Use this page for Manifold mesh booleans, hollow mesh shells, face opening
throats, and axis aligned shell partitions.

All coordinates, dimensions, gaps, and depths use meters.

This page documents `boolean_union`, `boolean_difference`,
`boolean_intersection`, `cut_opening_on_face`, `ShellPartitionRegion`,
`ShellPartitionSpec`, and `partition_shell`.

```python
from mini_articraft.sdk import BoxGeometry
from mini_articraft.sdk.mesh import (
    ShellPartitionRegion,
    ShellPartitionSpec,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    cut_opening_on_face,
    partition_shell,
    rounded_rect_profile,
)
```

## Mesh boolean rules

The boolean helpers use Manifold. Each input must be a nonempty valid closed
`MeshGeometry` solid. A mesh that only contains a surface, an uncapped pipe, or
separate open triangles is not a valid boolean input.

The helpers validate both inputs and return a new `MeshGeometry`. They do not
mutate either input.

A valid boolean can have no shared or remaining volume. The helpers raise an
error for an empty result.

A union of separated solids does not invent a bridge between them. Difference
and intersection reject an output with more than one positive volume body.
Keep intended separate pieces as separate named shapes. This also catches small
fragments left by a bad cut.

The helpers inspect every output for bad triangles and invalid topology. An
unhealthy result raises an error that includes the issue count and affected
bounds. Repair the source geometry or operation before using that result.

## boolean_union

```python
boolean_union(
    a: MeshGeometry,
    b: MeshGeometry,
) -> MeshGeometry
```

The result contains the volume occupied by either input.

```python
left = BoxGeometry((0.10, 0.08, 0.06)).translate(-0.03, 0.0, 0.0)
right = BoxGeometry((0.10, 0.08, 0.06)).translate(0.03, 0.0, 0.0)
joined = boolean_union(left, right)
```

Use `merge(...)` instead when you only need to combine triangle lists and do
not need one solved solid boundary.

## boolean_difference

```python
boolean_difference(
    a: MeshGeometry,
    b: MeshGeometry,
) -> MeshGeometry
```

The result contains the volume of `a` after removing the volume of `b`.

### Hollow shell example

```python
outer = BoxGeometry((0.30, 0.24, 0.20))

# The inner box extends through the top of the outer box. The remaining mesh
# has side walls and a floor, with a visible opening at the top.
inner = BoxGeometry((0.26, 0.20, 0.19)).translate(0.0, 0.0, 0.025)

shell = boolean_difference(outer, inner)
```

The example leaves a side wall thickness of `0.02` and a floor thickness of
`0.03`. The inner cutter extends above the outer top, so Manifold creates a real
opening instead of a sealed internal void.

Keep a positive wall thickness. A cutter that is exactly tangent to the outer
surface can create a fragile result. Extend a through cutter slightly beyond
the surface that it must open.

## boolean_intersection

```python
boolean_intersection(
    a: MeshGeometry,
    b: MeshGeometry,
) -> MeshGeometry
```

The result contains only the volume shared by both inputs. Shell partitioning
uses this operation to capture the part of a shell inside an axis aligned
region box.

## Boolean errors

The helpers raise `TypeError` when an input is not `MeshGeometry`.
`MeshGeometry.validate()` can raise `ValidationError` for invalid vertices or
faces. The helpers raise `ValueError` when an input is empty, open, nonmanifold,
or rejected by Manifold. They also raise when an output has bad triangles,
invalid edges, bad winding, inward orientation, or unintended separate solids.
The error names the operation and the affected bounds so you can inspect the
failing region.

Check these properties before retrying a failed boolean.

```python
geometry.validate()
print(geometry.is_watertight)
print(geometry.bounds)
```

Do not retry the same operation with arbitrary offsets until it happens to
pass. Find the open edge, zero thickness wall, duplicate face, or invalid input
builder first.

## Add a throat to an existing face opening

```python
cut_opening_on_face(
    shell_geometry: MeshGeometry,
    *,
    face: str,
    opening_profile: Iterable[tuple[float, float]],
    depth: float,
    offset: tuple[float, float] = (0.0, 0.0),
    taper: float = 0.0,
) -> MeshGeometry
```

This helper adds the interior side walls of an opening throat. It mutates
`shell_geometry`, returns that same mesh, and does not subtract a closed face.
The opening boundary must already exist in the shell surface.

Supported face values are `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, and
`"-z"`. The helper places the outer profile on the selected extreme plane of
the mesh bounds and moves the inner profile inward by `depth`.

The two profile coordinates map to the face like this.

- On an X face, the profile coordinates map to Y and Z.
- On a Y face, the profile coordinates map to X and Z.
- On a Z face, the profile coordinates map to X and Y.

`offset` moves the profile in those two face coordinates. Depth must be
positive. The opening profile must contain at least three distinct points and
have nonzero area.

The inner profile scale is `1 - taper`, measured around the profile center. A
positive taper narrows the throat inward. A negative taper widens it. The
absolute taper value must be less than 0.95.

```python
opening = rounded_rect_profile(0.08, 0.04, 0.006)
cut_opening_on_face(
    shell_with_existing_top_hole,
    face="+z",
    opening_profile=opening,
    depth=0.015,
    offset=(0.0, 0.01),
    taper=0.10,
)
```

If the face is still closed, use `boolean_difference(...)` with a closed cutter
to make the opening. Do not use `cut_opening_on_face(...)` as a substitute for
that subtraction.

## ShellPartitionRegion

```python
ShellPartitionRegion(
    name: str,
    side: Literal["full", "left", "right", "center"] = "full",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
)
```

A region defines an axis aligned cutter in the shell coordinate frame. Its name
must be nonempty. Every supplied bound must be finite. A minimum must be less
than its matching maximum.

Omitted bounds extend past the shell bounds by the spec `padding` value.

The `side` field adds an X constraint around the model origin.

- `"full"` adds no side constraint.
- `"left"` limits the high X value to `-center_gap / 2`.
- `"right"` limits the low X value to `center_gap / 2`.
- `"center"` limits X to the center gap and requires a positive center gap.

Explicit X bounds still apply. The side constraint can make a region empty,
which raises `ValidationError` during partitioning.

## ShellPartitionSpec

```python
ShellPartitionSpec(
    shell: MeshGeometry,
    regions: tuple[ShellPartitionRegion, ...],
    splitters: tuple[MeshGeometry, ...] = (),
    remainder_name: str | None = None,
    center_gap: float = 0.0,
    padding: float = 0.01,
)
```

`shell` must be a nonempty closed `MeshGeometry` solid because partitioning
uses Manifold.

At least one region is required. Region names must be unique. Every splitter
must also be a nonempty closed `MeshGeometry` solid.

`remainder_name` optionally stores all geometry left after the named regions.
It must be nonempty and different from every region name.

`center_gap` must be finite and nonnegative. `padding` must be finite and
positive.

## partition_shell

```python
partition_shell(
    spec: ShellPartitionSpec | MeshGeometry,
    /,
    **overrides,
) -> dict[str, MeshGeometry]
```

You may pass a complete spec, or pass a shell and provide `regions=...` plus
other spec fields as keyword arguments. Keyword overrides replace fields on an
existing spec for this call.

The operation follows this order.

1. It copies the shell.
2. It subtracts every splitter from the working shell in authored order.
3. It processes regions in authored order.
4. For each region, it intersects the remaining shell with the region box.
5. It stores that piece under the region name and subtracts it from the
   remaining shell.
6. If `remainder_name` is set, it stores the final remainder under that name.

Region order matters because each region can only capture geometry that remains
after earlier regions. A broad `"full"` region with no explicit bounds can
capture the whole shell and leave later regions empty.

The helper raises `ValidationError` when a region captures no geometry. It can
also raise a boolean `ValueError` for an open or invalid solid.

```python
spec = ShellPartitionSpec(
    shell=BoxGeometry((0.20, 0.30, 0.10)),
    regions=(
        ShellPartitionRegion("left_front", side="left", y_max=0.0),
        ShellPartitionRegion("right_front", side="right", y_max=0.0),
    ),
    remainder_name="body",
    center_gap=0.01,
)

pieces = partition_shell(spec)
left_front = pieces["left_front"]
right_front = pieces["right_front"]
body = pieces["body"]
```

The result meshes are independent. Add them as named shapes when they belong to
one rigid part, or add them to separate parts when the prompt requires separate
motion.

## weld and snap_to: blend a protrusion into a form

```python
weld(
    *geometries: MeshGeometry,
    radius: float = 0.006,
    tolerance: float | None = None,
    profile: Literal["tight", "round", "soft"] = "round",
    max_gap: float = 0.0,
    trim: MeshGeometry | None = None,
) -> MeshGeometry
```

The public `weld` function fuses closed solids into one smooth mesh. Use it to grow a molded
transition where a spout meets a shell, a boss meets a panel, or a handle meets
a body. Place the pieces so they overlap, then weld them and add the single
result to the part. Use `boolean_union(...)` instead when the joint should stay
sharp and preserve the exact input surfaces.

Weld samples the full bounds of every input and rebuilds their surfaces. It is
well suited to one continuous freeform result. It is not a local fillet for an
otherwise exact solid. Use build123d when exact faces, wall thickness,
openings, or rims must remain unchanged.

`radius` controls how far the smooth transition reaches. `tolerance` controls
the generated triangle size. It defaults to one quarter of the radius. Smaller
values preserve more surface detail and produce more triangles. Large objects
with very small tolerance values are rejected with a suggested minimum, so an
accidental setting cannot create an unbounded field.

The profile changes the fullness of the transition while keeping it smooth.
`"tight"` adds the least volume, `"round"` is the neutral shape, and `"soft"`
adds the fullest transition.

Inputs must overlap by default. Set `max_gap` to a small positive distance only
when the blend is meant to bridge that gap. The weld checks input connectivity
first and still fails if the selected radius cannot form one solid.

```python
body = LatheGeometry.from_shell_profiles(outer_profile=..., inner_profile=...)
spout = LoftGeometry(sections).translate(0.06, 0.0, 0.10)
molded = weld(
    body,
    spout,
    radius=0.008,
    tolerance=0.002,
    profile="round",
)
kettle.add(molded, name="body_with_molded_spout", color=(0.80, 0.82, 0.83))
```

When the body is a hollow shell and the protrusion pokes through the wall into
the cavity, pass `trim`, which is the solid that fills the cavity. The trim
field is applied during the same surface extraction as the weld. This avoids a
second near coincident boolean against the rebuilt surface.

```python
molded = weld(shell, spout, trim=cavity_solid)
```

If a protrusion was placed with a small gap to the form it should meet, call
`snap_to(anchor, piece, overlap=0.004, max_move=0.02, axis=None)` first. It moves the
piece toward `anchor` until they overlap by about `overlap` and returns it, so you do
not have to hit the exact coordinate by hand. Snap the piece BEFORE you add it:

```python
spout = snap_to(body, spout, max_move=0.01)   # close the gap
kettle.add(
    weld(body, spout, radius=0.008, tolerance=0.002),
    name="body_with_molded_spout",
    color=(0.80, 0.82, 0.83),
)
```

`snap_to` only translates the whole piece, so it fits a single freely-placeable
attachment (a boss, a foot, a spout root). It raises `SnapRefused` when closing the
gap would move the piece further than `max_move` -- a large required move means the
piece is constrained (a hinge barrel on its axis) or must meet the body at more than
one place (a handle with two ends), and those are fixed by their own shape or path,
not by snapping. Pass `axis` to constrain the motion to one direction.

## smooth_difference: generate rounded cuts

```python
smooth_difference(
    geometry: MeshGeometry,
    *cutters: MeshGeometry,
    radius: float = 0.006,
    tolerance: float | None = None,
    profile: Literal["tight", "round", "soft"] = "round",
    max_gap: float = 0.0,
) -> MeshGeometry
```

The public `smooth_difference` function subtracts one or more closed cutters and generates a
smooth transition where each cut meets the remaining surface. It uses the same
radius, tolerance, and profile controls as `weld(...)`.

Use `boolean_difference(...)` for an exact sharp cut. Use
`smooth_difference(...)` for a recessed control, vent, opening, or socket whose
edge should be part of the rounded generated surface.

```python
panel = BoxGeometry((0.12, 0.08, 0.012))
opening = CylinderGeometry(0.018, 0.04, radial_segments=48)
rounded_panel = smooth_difference(
    panel,
    opening,
    radius=0.004,
    tolerance=0.0015,
    profile="tight",
)
```

Every cutter must overlap the base or connect to it through another cutter.
`max_gap` can allow a nearby cutter to form a shallow recess without an exact
intersection. At least one cutter is required. The operation fails when the
cutters remove the entire solid.

## Decision guide

- Use `weld(...)` for one smooth generated transition. Set its radius, tolerance,
  and profile directly. Pass `trim` to remove a stub left inside a hollow body.
  Remember that it rebuilds every input surface.
- Use build123d for exact shells, openings, bores, rims, mating faces, and local
  fillets.
- Use `boolean_union(...)` when the joint should be sharp and keep the exact input
  surfaces.
- Use `snap_to(...)` to close a small gap before welding when the whole piece can
  move safely.
- Use `smooth_difference(...)` for a cut with a generated rounded transition.
- Use `boolean_difference(...)` for an exact sharp cavity, hole, or trim.
- Note: `boolean_difference` against a thin hollow shell removes only a wall-thick
  slice and can fragment the input into slivers; subtract a solid to remove
  everything inside a surface.
- Use `cut_opening_on_face(...)` only when the outer opening boundary already
  exists and you need its throat walls.
- Use `partition_shell(...)` to divide one closed solid into named axis aligned
  pieces.
- Use direct `MeshGeometry.merge(...)` when separate triangle components are
  acceptable and you do not need one solved solid boundary.

## Related references

- Read [mesh geometry and solid builders](00_mesh_geometry.md) for closed
  primitive inputs and build123d mesh conversion.
- Read [wires and sweeps](20_wires_and_sweeps.md) for closed tube networks that
  use Manifold union.
- See `docs/sdk/examples/hollow_shell.py` for an executable hollow shell built
  with `BoxGeometry` and `boolean_difference(...)`.

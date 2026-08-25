# Testing geometry and assemblies

Use `TestContext` to record checks for a `RigidBodyAssembly`. The checks use the same shapes,
body transforms, and units as export.

Every `main.py` must define `run_tests()` and return a `TestReport`.

```python
from articraft.sdk import TestContext, TestReport


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.expect_contact("base", "arm", shape_a="top_plate", shape_b="arm_foot")
    ctx.expect_gap(
        "shade",
        "bulb",
        axis="z",
        positive_shape="rim",
        negative_shape="glass",
        min_gap=0.002,
        max_gap=0.008,
    )
    return ctx.report()
```

Add checks for facts that are important to the requested object. The compiler adds the baseline
checks on this page.

Read [USDZ export](50_usdz_export.md) for the output from a passing report.

## Units and geometry

Positions, distances, gaps, tolerances, and translation values use meters. Rotation values use
radians.

Collision and distance checks convert each shape to a triangle mesh. `MeshGeometry` uses its
authored vertices and faces. A build123d shape is tessellated with the context's
`mesh_tolerance`, which defaults to `0.001` meter.

World bounds are axis aligned. Projection checks therefore measure world bounds, not the exact
curved surface between those bounds.

## Create a context

```python
ctx = TestContext(object_model, mesh_tolerance=0.001)
```

`model` must be a `RigidBodyAssembly`. `mesh_tolerance` must be positive and finite.

Select a body with a `RigidBody` or its name. Select a shape with its unique name within the
body.

A missing body or shape raises `ValidationError`.

`ctx.pose({...})` applies temporary tree positions. It also solves unspecified positions in
closed loops.

`ctx.state(physics_state)` uses a complete `PhysicsState`. Use it for multiple articulation
trees or poses from a physics engine.

## Check frame intent

Frame checks consume the same `BodyFrame` values as `model.joint(...)` and use
the current pose.

```python
ctx.expect_coincident(base.at(base_pin), arm.at(arm_pin))
ctx.expect_coaxial(base.at(base_axis), arm.at(arm_axis), axis="z")
```

`expect_coincident(first, second, *, position_tol=1e-6, angle_tol=1e-6, name=None)` checks both
origins and complete orientations.

`expect_coaxial(first, second, *, axis="z", position_tol=1e-6, angle_tol=1e-6, name=None)` checks
that two selected local axes share one line.

Opposite directions count as aligned. Linear tolerances use meters, and angular tolerances use
radians.

Use these checks for pins, bearings, hinge edges, and loop closures. Use
`expect_contact(...)` to check surface contact.

Use `expect_within(...)` or `expect_gap(...)` for support relationships. Those functions check
meshes, not frames.

## Reports

### `TestFailure`

```python
TestFailure(name: str, details: str, kind: FailureKind = FailureKind.AUTHORED)
```

Each blocking failure has a check name and details. The check method assigns a
`FailureKind`, such as `OVERLAP` or `CONTACT`.

`check()` and `fail()` use `FailureKind.AUTHORED`. The compile worker adds `source="tests"` or
`source="compiler"` to each serialized failure.

### `AllowedOverlap`

```python
AllowedOverlap(
    part_a: str,
    part_b: str,
    reason: str,
    shape_a: str,
    shape_b: str,
)
```

The report uses this record for each overlap allowance. The context sorts the two part names so
the same pair has one stable representation. It swaps the shape names at the same time.

### `AllowedMeshIssues`

```python
AllowedMeshIssues(
    part: str,
    shape: str,
    issues: tuple[MeshHealthIssue, ...],
    reason: str,
)
```

The report uses this record for a mesh health allowance. It always names one part, one shape, and
one or more exact issue types.

### `DistanceFinding`

```python
DistanceFinding(
    part_a: str,
    part_b: str,
    distance: float,
    collided: bool,
    shape_a: str | None = None,
    shape_b: str | None = None,
    nearest_a: tuple[float, float, float] | None = None,
    nearest_b: tuple[float, float, float] | None = None,
    contacts: tuple[ContactInfo, ...] = (),
)
```

`distance_between()` returns this record. A collision has distance zero. The nearest points can
be absent when the mesh collision library does not provide them.

Collision assertion methods return `bool`. They put collision details in the
recorded failure instead of returning another result type.

### `TestReport`

```python
TestReport(
    passed: bool,
    checks_run: int,
    checks: tuple[str, ...],
    failures: tuple[TestFailure, ...],
    warnings: tuple[str, ...] = (),
    allowances: tuple[str, ...] = (),
    allowed_isolated_parts: tuple[str, ...] = (),
    allowed_overlaps: tuple[AllowedOverlap, ...] = (),
    allowed_mesh_issues: tuple[AllowedMeshIssues, ...] = (),
    metrics: tuple[TestMetric, ...] = (),
    artifacts: tuple[TestArtifact, ...] = (),
)
```

`passed` is true when there are no blocking failures. Warnings and allowances do not make a
report fail. `checks_run` counts recorded checks. Calling `warn()` or an allowance method does
not increase that count.

### `MeshHealthIssue`, `MeshHealthFinding`, and `MeshHealthReport`

Use `analyze_mesh_health` to inspect a `MeshGeometry`, a build123d shape, or a
`trimesh.Trimesh`. It returns a `MeshHealthReport`.

The report includes triangle and vertex counts, disconnected component count, watertight and
winding status, signed volume, and a tuple of `MeshHealthFinding` records. Each finding has a
`MeshHealthIssue`, a count, details, and the bounds of the affected region when available.

The issue types cover degenerate and sliver faces, duplicate faces and vertices, unused vertices,
boundary and nonmanifold edges, disconnected components, inconsistent winding, and inward
orientation.

### `TestMetric`, `TestArtifact`, and `GeometryMetrics`

`record_metric(...)` and `expect_metric(...)` add a `TestMetric` to the report.
An expected metric also records a blocking authored check.

`attach_artifact(...)` adds a `TestArtifact` for a file that already exists.
Read [visual evidence and artifacts](45_visual_evidence.md) for the preview
workflow and supported file types.

`measure_geometry(...)` returns `GeometryMetrics`.

```python
GeometryMetrics(
    bounds: tuple[Vec3, Vec3],
    dimensions: Vec3,
    centroid: Vec3,
    triangle_count: int,
    component_count: int,
    watertight: bool,
    surface_area: float,
    signed_volume: float,
    orientation: str,
)
```

`orientation` is `outward`, `inward`, `open`, or `degenerate`.

### Reporting methods

```python
ctx.check(name, ok, details="")
ctx.fail(name, details)
ctx.warn(text)
ctx.record_metric(name, value, unit="", details="")
ctx.expect_metric(name, value, minimum=None, maximum=None, unit="", details="")
report = ctx.report()
```

`check()` records a blocking failure when `ok` is false. `fail()` always records a blocking
failure and returns false. `warn()` records a nonblocking message and ignores an exact duplicate.
`report()` returns the checks recorded so far.

## Shape measurements

```python
ctx.measure_geometry(part=None, *, shape=None) -> GeometryMetrics
ctx.expect_bounds(part=None, *, shape=None, minimum=None, maximum=None, tolerance=0.0)
ctx.expect_radial_extent(
    part=None,
    *,
    shape=None,
    axis=(0.0, 0.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    minimum=None,
    maximum=None,
)
ctx.expect_component_count(expected, part=None, *, shape=None)
ctx.expect_watertight(part=None, *, shape=None)
ctx.expect_positive_volume(part=None, *, shape=None, minimum=0.0)
ctx.expect_symmetry(
    part=None,
    *,
    shape=None,
    plane_origin=(0.0, 0.0, 0.0),
    plane_normal=(1.0, 0.0, 0.0),
    tolerance=0.001,
)
```

Leave out `part` to measure the full model at the current pose. A shape selector
requires a part. Signed volume is positive when a closed mesh has outward face
winding. `expect_positive_volume(...)` requires the signed volume to be greater
than its `minimum`.

## Poses

Use `pose()` to apply temporary joint positions.

```python
rest = ctx.part_world_position("slider")
with ctx.pose({slide: 0.12}):
    extended = ctx.part_world_position("slider")
    ctx.expect_gap("slider", "stop", axis="x", min_gap=0.003)

assert ctx.part_world_position("slider") == rest
```

The signature is:

```python
ctx.pose(
    articulation_positions: Mapping[object, float] | None = None,
    **positions: float,
)
```

Keys name joints, not articulations. Use one of these forms:

- A `Joint` object.
- A joint name such as `"hinge"` for a joint with one degree of freedom.
- A qualified identifier such as `"hinge.rotZ"` or `"slide.transX"`.

Use the qualified form for a joint with several free axes. The same identifiers work with
`forward_kinematics(...)` and `PhysicsState.dof_positions`.

Qualified identifiers contain a dot, so pass them through the mapping argument. Each value must
be finite and within its limits.

An invalid position raises `ValidationError` at the `pose()` call. You cannot pose a fixed joint
or a closing joint.

The context restores the previous pose when the `with` block ends. Nested blocks restore the
pose that was active before each block.

In a closed loop, direct measurements raise `LoopClosureError` for an unreachable pose. Sweep
checks record an unreachable sample as a failure and continue.

The mechanism determines the position of a closing joint, so you cannot pose it directly.

### `PoseSample`

`sample_joint(...)` returns `PoseSample` records. It requires a joint with one degree of freedom
and rejects closing joints.

The default sweep spans the authored limits. A closed loop can have a smaller reachable range,
so sweep checks record unreachable samples as failures.

```python
poses = ctx.sample_joint("door_hinge", samples=5)
poses = ctx.sample_joint("door_hinge", positions=(0.0, 0.4, 0.8))
```

Use `sample_dof(...)` when a joint has several degrees of freedom. Pass an axis name or the
qualified DOF identifier:

```python
lift_poses = ctx.sample_dof(
    "removable_lid",
    "removable_lid.transZ",
    positions=(0.0, 0.04, 0.10),
)
```

The other degrees of freedom keep their current values during the sweep.

Use these records with:

```python
ctx.expect_no_collision_at_poses(part_a, part_b, poses, ...)
ctx.expect_distance_at_poses(part_a, part_b, poses, minimum=0.0, maximum=None, ...)
ctx.expect_contact_at_poses(part_a, part_b, poses, contact_tol=1e-6, ...)
ctx.expect_within_at_poses(inner_part, outer_part, poses, ...)
ctx.track_point(part, local_point, poses) -> tuple[Vec3, ...]
```

`part_world_point(part, point)` transforms one point from a part's local frame
into the current world pose.

## World inspection

These methods read the current pose and do not record a check.

### `part_world_position(part) -> Vec3`

Returns the part origin in world coordinates. This is the articulation frame position, not the
center of the part geometry.

### `part_world_bounds(part) -> tuple[Vec3, Vec3]`

Returns the minimum and maximum world coordinates for all named shapes in the part.

### `shape_world_bounds(part, shape) -> tuple[Vec3, Vec3]`

Returns the world bounds for one named shape.

### `distance_between(part_a, part_b, *, shape_a=None, shape_b=None) -> DistanceFinding`

Returns the smallest mesh distance among the selected shapes. You can scope either side or both
sides. With no shape names, the query checks every shape pair between the two parts.

The two part arguments can name the same part. The selected sets must contain at least two
different shapes. Naming both shapes is the clearest form. A query that selects one shape against
itself raises `ValidationError`.

These methods are useful in a short `exec_command` inspection.

```python
from main import object_model
from articraft.sdk import TestContext

ctx = TestContext(object_model)
print(ctx.shape_world_bounds("housing", "shell"))
print(ctx.distance_between("housing", "door", shape_a="rim", shape_b="panel"))
```

## Exact collision and distance checks

Every method in this section records one check and returns true on pass or false on failure. The
optional `name` replaces the generated check name.

### `expect_no_collision(part_a, part_b, *, shape_a=None, shape_b=None, name=None)`

Passes when the selected triangle meshes do not collide. This is a direct collision test. It does
not use the compiler's meaningful overlap thresholds.

### `expect_collision(part_a, part_b, *, shape_a=None, shape_b=None, name=None)`

Passes when at least one selected shape pair collides. If several pairs collide, the report uses a
representative pair with the largest bounds overlap.

### `expect_contact(part_a, part_b, *, shape_a=None, shape_b=None, contact_tol=1e-6, name=None)`

Passes when the selected geometry collides or its minimum distance is at most `contact_tol`.
`contact_tol` must be nonnegative and uses meters.

### `expect_distance(part_a, part_b, *, shape_a=None, shape_b=None, min_distance=0.0, max_distance=None, name=None)`

Passes when the minimum mesh distance is at least `min_distance` and no more than
`max_distance`, when an upper bound is provided. Both bounds are inclusive and nonnegative. A
collision has distance zero. An upper bound below the lower bound records a failed check.

Shape selectors are independent for these methods. For example, `shape_a="shaft"` can be checked
against every shape in `part_b`.

## Exact bounds checks

These checks use world axis aligned bounds. They do not prove mesh contact or mesh collision.

### `expect_gap(positive_part, negative_part, *, axis, positive_shape=None, negative_shape=None, min_gap=None, max_gap=None, max_penetration=None, name=None)`

The signed gap is:

```text
positive bounds minimum on axis minus negative bounds maximum on axis
```

`axis` must be `"x"`, `"y"`, or `"z"`. A positive result is a gap. A negative result is bounds
penetration.

When `min_gap` is omitted, the lower bound is zero. If `max_penetration` is provided, the lower
bound is `-max_penetration`. A provided `min_gap` takes precedence. `max_gap` is an optional
inclusive upper bound. An upper bound below the lower bound records a failed check.

### `expect_within(inner_part, outer_part, *, inner_shape=None, outer_shape=None, axes="xy", margin=0.0, name=None)`

Passes when the inner bounds stay inside the outer bounds on every requested axis. `margin`
allows the inner bounds to extend that far beyond the outer bounds and must be nonnegative.

### `expect_overlap(part_a, part_b, *, shape_a=None, shape_b=None, axes="xy", min_overlap=0.0, name=None)`

Passes when the projected bounds overlap by at least `min_overlap` on every requested axis. A
minimum of zero allows exact bounds contact. `min_overlap` must be nonnegative.

For `axes`, use a string such as `"xy"` or a sequence such as `("x", "z")`. Repeated axes are
ignored. At least one axis is required.

`expect_overlap()` proves projected overlap only. It does not declare an allowance and does not
suppress the compiler's mesh overlap check.

## Allowances

Allowances describe intentional exceptions to compiler owned physical checks. Each reason must be
nonempty.

### `allow_overlap(part_a, part_b, *, reason, shape_a, shape_b)`

Both shape names are required. The allowance covers only that exact named shape
pair.

```python
ctx.allow_overlap(
    "shaft",
    "hub",
    shape_a="steel_shaft",
    shape_b="bore_liner",
    reason="The shaft is captured inside the bearing liner.",
)
```

An allowance does not hide another collision between the same two parts.

An overlap allowance affects only `fail_if_parts_overlap_in_current_pose()`, including the copy of
that check which the compiler runs. It does not make `expect_no_collision()` or another authored
check pass.

Pair an allowance with an exact check that explains the intended relationship. For a captured
shaft, an `expect_contact()`, `expect_within()`, or bounds gap check can provide that evidence.

### `allow_isolated_part(part, *, reason)`

Allows one named part in `fail_if_isolated_parts()`. If several touching parts form one floating
group, every part in that group must have an allowance. The check still records a nonblocking
warning that the group was allowed.

An isolation allowance does not affect disconnected shapes inside one rigid part.

### `allow_mesh_issues(part, *, shape, issues, reason)`

Allows exact mesh health issue types on one named shape. It does not hide other issue types on the
same shape or any issue on another shape.

```python
ctx.allow_mesh_issues(
    "shade",
    shape="fabric_panel",
    issues=(MeshHealthIssue.BOUNDARY_EDGES,),
    reason="This named fabric panel is an intentional open surface.",
)
```

Use this only for a shape that is meant to have the reported property. Repair accidental holes,
bad triangles, disconnected debris, and invalid topology.

## Compiler owned checks

The compile worker runs `run_tests()` first. It then copies authored overlap, isolation, and mesh
health allowances into a new context and runs these baseline checks before export:

1. `check_model_valid()`
2. `fail_if_mesh_unhealthy()`
3. `fail_if_parts_have_no_mass()` (only while the physics lane is enabled)
4. `fail_if_isolated_parts()`
5. `warn_if_part_contains_disconnected_geometry_islands()`
6. `warn_if_absurd_dimensions()`
7. `fail_if_parts_overlap_in_current_pose()`
8. `fail_if_articulation_separates_child()`
9. `fail_if_loop_limits_contradict()` (only reaches a closed loop the articulation tree spans)

If model validity fails, the worker stops the rest of the baseline pass. When the object is valid
enough to inspect, model validity, mesh health, missing mass properties, and USDZ validation can
block the compiler. The other baseline methods appear as nonblocking diagnostics.
Add an authored check when one of those findings is important to the requested object.
The failed check still makes the compile fail and prevents final publication.

The merged report keeps one copy of each check name. A compiler failure replaces an authored
failure with the same name. A passing compiler check never erases an authored failure. Exact
duplicate failures, warnings, and allowance strings are also merged.

### Model validity and root policy

`check_model_valid()` calls `object_model.validate()`. The assembly must contain valid named
geometry and one connected physical joint graph.

Each articulation validates its root and selected tree. An assembly with several articulations
does not have one global root.

### Physical isolation

`fail_if_isolated_parts(*, contact_tol=1e-6, name=None)` builds a physical contact graph at the
current pose. Two parts are connected when their meshes collide or their minimum distance is no
more than `contact_tol`. An articulation by itself does not count as physical support.

The graph starts at the root part from the articulation tree. A physically separate group fails
unless every part in that group has an isolation allowance. Parent and child parts are checked in
the same way as any other pair.

### Disconnected geometry within a part

`warn_if_part_contains_disconnected_geometry_islands(*, contact_tol=1e-6, name=None)` splits all
named shapes into mesh components. Components are connected when they collide or are within the
contact tolerance. The compiler records a warning when one rigid part contains more than one
physical group.

Disconnected geometry is nonblocking in the baseline pass. Use the blocking form when one of your
design requirements needs it:

```python
ctx.fail_if_part_contains_disconnected_geometry_islands(contact_tol=1e-6)
```

Nested closed solids with positive volume intersection count as connected geometry.

### Loop limits

`fail_if_loop_limits_contradict(*, samples=9, tolerance=1e-6, overrun_fraction=0.25,
max_solves=1000, name=None)` sweeps every bounded coordinate on a closed loop. The sweep walks
outward from the rest pose. Each sample continues from its neighbour's solution, with the
coordinates the solver derives unbounded, so the walk follows the motion the linkage asks for.
Each side of the sweep ends at the first pose where the ring will not close. A bisection toward
that edge measures the travel the mechanism covers. It also probes the motion close to the edge,
where a follower moves fastest. The check names two defects.

A follower whose limits exclude motion the linkage requires. Where the walk leaves a follower's
authored limits, the solve that respects those limits is asked for the same pose. Only its
failure to close the ring within them makes a case. When it solves every accused pose instead, the
limits blocked nothing. The disagreement then indicts the unbounded walk, which has no assembly
branch guarantee near a singular fold. The report gives the declared range, the range the
mechanism needs, and the driver poses where the two disagree. The needed range is rounded
outward, so its numbers can be copied back as the new limits.

A driver whose declared range is wider than the linkage can follow. This is reported once more
than `overrun_fraction` of the declared travel lies past where the ring stops closing. The
fraction measures travel rather than samples, so slack smaller than that stays quiet at every
sweep density.

A full-circle coordinate is never reported as a defect. A hinge authored `(-pi, pi)` says
unconstrained. It does not claim that every angle is reachable. When such a ring closes over less
than a hundredth of its declared range, the check warns that the loop barely moves.

The walk costs about one loop solve per sample per coordinate. Each unreachable edge adds its
bisection, and each pose a limit excludes adds one bounded solve. A coordinate on two rings is
driven once. `max_solves` bounds that budget. When a ring has more bounded coordinates than fit,
the check warns and names the ones it did not drive.

### Scale warnings

`warn_if_absurd_dimensions(*, max_dimension=1000.0, outlier_ratio=100.0, name=None)` checks the
largest world bounds span of every named shape. It warns when a shape is larger than
`max_dimension` meters. It also warns when a shape is more than
`outlier_ratio` times the median positive shape span.

This check is always nonblocking.

### Meaningful overlap

`fail_if_parts_overlap_in_current_pose(*, overlap_tol=0.005, overlap_volume_tol=5e-7, name=None)`
checks every named shape pair from different parts at the current pose, including adjacent parent
and child parts.

A pair can be a blocking overlap only when all of these facts are true:

1. Its world bounds overlap by more than `overlap_tol` on all three axes.
2. Its bounds overlap volume is greater than `overlap_volume_tol` cubic meters.
3. The triangle meshes collide.
4. No matching overlap allowance exists.

For watertight meshes, a pair blocks only when its intersection volume exceeds
`overlap_volume_tol`.

Without solid volume, the pair blocks for penetration greater than `1e-6` meters. It also blocks
when no penetration depth is available.

Mere contact passes. Bounds penetration at or below either physical threshold also passes. A
shape allowance suppresses only its exact pair. The report includes at most one representative
unallowed shape pair for each part pair, chosen by overlap depth and volume.

The baseline check uses the rest pose. Use `with ctx.pose(...)` and an authored exact check when a
different pose is important to the requested mechanism.

## Choose checks

Checks are design evidence. Do not delete or simplify visible geometry only to make a check pass.

Use shape names when a part contains unrelated regions. Use a small number of checks that prove
the important support, clearance, insertion, or motion. If a check finds a real defect, repair the
geometry, articulation, pose, or exact selector. Add an allowance only when the physical
relationship is intentional.

# Mesh refinement and smoothing

Use this page to add triangle density or smooth vertex positions after you
create a mesh.

All lengths use meters. Every function returns a new `MeshGeometry` and leaves
the input mesh unchanged.

This page documents `refine_mesh`, `subdivide_mesh`, and `smooth_mesh`.

```python
from articraft.sdk.mesh import refine_mesh, smooth_mesh, subdivide_mesh
```

## Choose an operation

Use `refine_mesh(...)` when you need a maximum edge length. It adds triangles
without changing the surface.

Use `subdivide_mesh(...)` when you want a fixed number of subdivision levels.
Plain subdivision keeps the surface unchanged. Smooth subdivision rounds the
surface.

Use `smooth_mesh(...)` when the mesh already has enough triangles and you want
to smooth uneven vertex positions.

## Edge length refinement

```python
refine_mesh(
    mesh: MeshGeometry,
    *,
    max_edge_length: float,
    max_iterations: int = 10,
) -> MeshGeometry
```

The function splits triangles until every edge is no longer than
`max_edge_length`. New vertices lie on the existing triangles. The bounds and
represented surface therefore stay unchanged.

`max_edge_length` must be finite and positive. `max_iterations` must be a
nonnegative integer. The function raises `ValueError` if it cannot reach the
requested edge length within that limit.

```python
coarse = SphereGeometry(0.05, width_segments=12, height_segments=8)
dense = refine_mesh(coarse, max_edge_length=0.004)
```

Refinement does not make a coarse curved surface rounder by itself. It only
adds vertices to the current triangles. Use smooth subdivision when you want
the surface shape to change.

## Fixed subdivision levels

```python
subdivide_mesh(
    mesh: MeshGeometry,
    *,
    levels: int = 1,
    smooth: bool = False,
) -> MeshGeometry
```

Each level splits every triangle into four triangles. A level of zero returns
an independent copy.

With `smooth=False`, the function uses midpoint subdivision. All new vertices
remain on the input triangles, so the represented surface stays unchanged.

With `smooth=True`, the function uses Loop subdivision. It adds triangles and
moves vertices to create a smoother surface. It can round sharp edges and
reduce volume. Use it for forms that should become softer. Do not use it when
precise corners or dimensions must stay fixed.

```python
soft_form = subdivide_mesh(coarse, levels=2, smooth=True)
```

Triangle count grows by a factor of four at each level. Keep the level count
small and inspect the result before adding another level.

## Vertex smoothing

```python
smooth_mesh(
    mesh: MeshGeometry,
    *,
    iterations: int = 10,
    preserve_boundary: bool = True,
) -> MeshGeometry
```

The function applies Taubin smoothing. This method alternates contraction and
expansion to limit surface shrinkage. It keeps the original face topology.

When `preserve_boundary=True`, the function pins every vertex on an edge used
by only one triangle. This keeps the outline of an open surface fixed while
interior vertices move. A closed mesh has no open boundary, so this option has
no effect on it.

An iteration count of zero returns an independent copy. The count must be a
nonnegative integer.

```python
smoothed = smooth_mesh(dense, iterations=6, preserve_boundary=True)
```

Smoothing can still change volume and soften internal creases. Start with a
small iteration count. Use plain refinement when the shape must remain exact.

## Input rules

Every operation requires a nonempty `MeshGeometry` with vertices and triangle
faces. The mesh may be open or closed. When an operation runs, the returned
mesh does not include vertices that no face uses. A zero level or iteration
count returns a direct copy instead.

These functions do not repair nonmanifold topology or self intersections. Fix
those defects before smoothing when they affect the intended surface.

## Related references

Read [mesh geometry and solid builders](00_mesh_geometry.md) for the base mesh
type and primitive builders. Read [section lofts and repair](30_section_lofts.md)
for cleanup of loft results.

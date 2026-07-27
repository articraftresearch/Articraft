# Collision shapes

`ShapeRole` says whether a shape is drawn, collided against, or both.
`CollisionApproximation` says how a simulator may simplify a mesh collider.

Every part is a rigid body, so it needs geometry a simulator can push against. By
default every shape you add is both drawn and collided against, and you do not have to
think about this at all:

```python
body = model.part("body")
body.add(CylinderGeometry(0.05, 0.12), name="shell")  # drawn and collided against
```

## Splitting the two jobs

The visible surface and the collision surface usually coincide. They should not when
detail that matters visually is wasted on contact, or when a stand-in collides better
than the real thing:

```python
from mini_articraft.sdk import ShapeRole

# A fine ridged grip that nothing needs to collide against precisely.
handle.add(ridges, name="grip_ridges", role=ShapeRole.VISUAL)

# A simple capsule that stands in for it, never drawn.
handle.add(CapsuleGeometry(0.012, 0.09), name="grip_collider", role=ShapeRole.COLLISION)
```

| Role | Drawn | Collides |
| --- | --- | --- |
| `ShapeRole.VISUAL_AND_COLLISION` (default) | yes | yes |
| `ShapeRole.VISUAL` | yes | no |
| `ShapeRole.COLLISION` | no | yes |

A collision-only shape is exported as an invisible mesh rather than dropped, so the
collider still resolves on the stage while nothing draws it.

**Every part must keep at least one collider.** A part whose shapes are all
`ShapeRole.VISUAL` renders but falls through the world, so the compile fails naming it.

## Approximation

Engines generally cannot simulate a moving body against raw triangles, so a mesh
collider records how it may be simplified:

```python
lid.add(panel, name="panel", collision_approximation=CollisionApproximation.CONVEX_HULL)
```

| Approximation | Use it for |
| --- | --- |
| `CONVEX_DECOMPOSITION` (default) | anything with a cavity or a concave profile -- a tin, a mug, a drawer |
| `CONVEX_HULL` | a genuinely convex part, where it is cheaper and exact |
| `BOUNDING_CUBE`, `BOUNDING_SPHERE` | coarse stand-ins where contact detail does not matter |
| `MESH_SIMPLIFICATION` | a dense mesh whose silhouette matters more than its detail |
| `NONE` | raw triangles; only appropriate for geometry that never moves |

The default is `CONVEX_DECOMPOSITION` because articulated objects are mostly not convex.
A convex hull of a hollow tin fills the cavity, and the lid can then never close into it.

## Export

A collider gets `UsdPhysics.CollisionAPI` and `UsdPhysics.MeshCollisionAPI` with
`physics:approximation` on its mesh prim, alongside the `RigidBodyAPI` already applied
to the part. The manifest records each shape's `role` and `collision_approximation`.
See `docs/sdk/common/50_usdz_export.md` for the rest of the USD layout.

# SDK quickstart

Work in meters and radians.

Each `RigidBody` is one rigid body holding named shapes, either `build123d.Shape` or
`MeshGeometry`. Apply build123d `Pos`, `Rot`, or `Location` before adding; there is no second
shape transform.

```python
from build123d import Box, Pos

from articraft.sdk import RigidBodyAssembly, TestContext, TestReport


model = RigidBodyAssembly("small_table")
body = model.rigid_body("body")
body.add(Box(0.8, 0.5, 0.04), name="top", color=(0.45, 0.24, 0.10))
body.add(Pos(X=0.36, Y=0.21, Z=-0.36) * Box(0.04, 0.04, 0.7), name="leg_1")

object_model = model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    return ctx.report()
```

Every shape needs a unique name within its part. A color can contain RGB or RGBA values from
zero through one.

Within one rigid body, overlapping shapes count as connected -- `leg_1` overlaps up into `top`
above. To attach a handle or spout, extend its end a few millimeters into the form it meets; it
then reads as one molded piece.

Prefer build123d for exact solids, wall thickness, openings, bores, rims and local fillets. Use
mesh helpers when the form is better described by freeform sections or paths. One body can hold
both.

Motion is two steps. `model.joint(...)` connects two bodies at a frame on each, and its `dofs`
say which axes are free: none is fixed, one rotational is a hinge, one linear is a slide, three
rotational is a ball. `model.articulation(...)` then names the tree the simulator solves.

```python
from articraft.sdk import JointAxis, JointDOF, JointFrame


lid = model.rigid_body("lid")
lid.add(Box(0.8, 0.5, 0.02), name="panel")
model.joint(
    "lid_hinge",
    body0=body,
    frame0=JointFrame(xyz=(0.0, 0.25, 0.02)),
    body1=lid,
    frame1=JointFrame(xyz=(0.0, 0.25, -0.01)),
    dofs=(JointDOF(JointAxis.ROT_Y, limits=(0.0, 1.9)),),
)
model.articulation("main", root=body, joints=["lid_hinge"])
```


Each frame is where the joint sits *in that body's own coordinates*, so the two coincide at rest.
Limits are radians for rotation, meters for travel, and must contain zero.

**Count the pivots.** A body pinned in two places takes two joints, which makes the mechanism a
ring -- linkages, four-bars, grippers and scissors all are. Author every joint it has, then leave
the ring-closing one out of the articulation. See `docs/sdk/common/35_joints.md`.

Read only the reference that applies to the next piece of geometry:

- Errors and validation: `docs/sdk/common/10_errors.md`.
- Shared units and types: `docs/sdk/common/20_core_types.md`.
- Named shapes and parts: `docs/sdk/common/30_assembly.md`.
- Articulations: `docs/sdk/common/35_joints.md`.
- Materials and mass: `docs/sdk/common/37_materials.md`.
- Simulation settings, gravity and initial motion:
  `docs/sdk/common/38_simulation_settings.md`.
- Checks and geometry inspection: `docs/sdk/common/40_testing.md`.
- Visual views and report artifacts: `docs/sdk/common/45_visual_evidence.md`.
- USDZ output: `docs/sdk/common/50_usdz_export.md`.
- Mesh editing, primitives, lathes, lofts, and extrusions:
  `docs/sdk/mesh/00_mesh_geometry.md`.
- Profiles and curve sampling: `docs/sdk/mesh/10_profiles.md`.
- Wires, pipes, and sweeps, including changing profiles and frame control:
  `docs/sdk/mesh/20_wires_and_sweeps.md`.
- Section lofts, smooth interpolation, rounded caps, and guide paths:
  `docs/sdk/mesh/30_section_lofts.md`.
- Mesh booleans, rounded cuts, shell partitioning, and configurable smooth welds:
  `docs/sdk/mesh/40_booleans_and_shells.md`.
- Mesh refinement and smoothing:
  `docs/sdk/mesh/50_refinement_and_smoothing.md`.

Detailed build123d pages are under `docs/sdk/build123d/`: `key_concepts_algebra.md` for object
algebra, `moving_objects.md` for placement, `operations.md` for solid operations, and
`topology_selection.md` for selecting faces and edges. Their examples use arbitrary dimensions;
convert every one to meters.

Use the reference pages for API discovery, and short `exec_command` inspections after authoring
to measure bounds, distances, collisions, and posed geometry.

Read only the executable example closest to the current task:

- Hollow shell: `docs/sdk/examples/hollow_shell.py`.
- Section loft with a swept wire: `docs/sdk/examples/section_loft_with_wires.py`.
- Mixed build123d and mesh assembly: `docs/sdk/examples/mixed_articulated_assembly.py`.
- Molding a handle/protrusion into a body (no mounting pads):
  `docs/sdk/examples/molded_mug.py`.
- Mass properties from materials and geometry:
  `docs/sdk/examples/mass_properties.py`.
- A closed-loop linkage, and which joint to leave out of the articulation:
  `docs/sdk/examples/closed_loop_linkage.py`.
- Variable profile sweep and smooth section loft:
  `docs/sdk/examples/variable_sweep_and_loft.py`.

Run `compile` after meaningful edits. Treat checks as design evidence: a failed check is not a
reason to remove geometry the prompt requires.

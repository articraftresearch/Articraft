# SDK quickstart

Use meters for lengths and radians for angles. Build123d rotations are the only exception.
They use degrees.

A `RigidBody` contains named shapes that always move together. A shape can be a
`build123d.Shape` or `MeshGeometry`.

Apply a build123d location before you add the shape. The SDK does not apply a second shape
transform.

```python
from build123d import Box, Pos

from articraft.sdk import RigidBodyAssembly, TestContext, TestReport

model = RigidBodyAssembly("small_table")
body = model.rigid_body("body")
top = body.add(Box(0.8, 0.5, 0.04), name="top", color=(0.45, 0.24, 0.10))
body.add(Pos(X=0.36, Y=0.21, Z=-0.36) * Box(0.04, 0.04, 0.7), name="leg_1")

object_model = model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    return ctx.report()
```

Give each shape a unique name within its body. A color contains RGB or RGBA values from zero
through one.

Make connected shapes overlap within one rigid body. For example, extend a leg into the table
top by a few millimeters.

Use build123d for exact boundaries, wall thickness, openings, bores, rims, and local fillets.
Use mesh helpers for freeform sections or paths.

## Decide whether the object moves

Many objects have no independent motion. A loose teapot or saucepan lid is a separate seated
part, not a reason to invent a hinge. Add a retained mechanism only when the request,
reference, or ordinary construction calls for one.

## Add retained motion

First, use `model.joint(...)` to connect two bodies. Then use `model.articulation(...)` to
select the solver tree.

```python
from articraft.sdk import JointAxis, JointDOF

moving_model = RigidBodyAssembly("wall_cabinet")
frame = moving_model.rigid_body("cabinet_frame")
frame.add(Box(0.8, 0.08, 0.6), name="frame")
door = moving_model.rigid_body("cabinet_door")
door.add(Box(0.8, 0.02, 0.6), name="panel")

moving_model.joint(
    "door_hinge",
    frame.at((-0.4, 0.0, 0.0)),
    door.at((-0.4, 0.0, 0.0)),
    dofs=(JointDOF(JointAxis.ROT_Z, limits=(0.0, 1.9)),),
)
moving_model.articulation("main", root=frame, joints=["door_hinge"])
```

Use `body.at(...)` to bind a point or build123d feature to its body. Pass the same feature to
both endpoints when possible.

The endpoint frames coincide at rest. Each joint limit uses radians or meters and must include
zero.

Count every physical pivot. A body with two pivots needs two joints, which creates a closed
loop.

Author every joint in a closed loop. Omit one closing joint from the articulation tree.

## Read the applicable reference

Read only the reference that applies to your next task:

- Read `docs/sdk/common/10_errors.md` for errors and validation.
- Read `docs/sdk/common/20_core_types.md` for shared units and types.
- Read `docs/sdk/common/30_assembly.md` for assemblies, bodies, and shapes.
- Read `docs/sdk/common/35_joints.md` for joints and articulations.
- Read `docs/sdk/common/37_materials.md` for materials and mass.
- Read `docs/sdk/common/38_simulation_settings.md` for gravity and initial motion.
- Read `docs/sdk/common/40_testing.md` for checks and geometry inspection.
- Visual views and report artifacts: `docs/sdk/common/45_visual_evidence.md`.
- Read `docs/sdk/common/50_usdz_export.md` for USDZ output.
- Read `docs/sdk/mesh/00_mesh_geometry.md` for mesh editing and solid builders.
- Read `docs/sdk/mesh/10_profiles.md` for profiles and curve samples.
- Read `docs/sdk/mesh/20_wires_and_sweeps.md` for wires, pipes, and sweeps.
- Read `docs/sdk/mesh/30_section_lofts.md` for section lofts and guide paths.
- Read `docs/sdk/mesh/40_booleans_and_shells.md` for booleans, shells, and welds.
- Read `docs/sdk/mesh/50_refinement_and_smoothing.md` for mesh refinement.

The detailed build123d pages are under `docs/sdk/build123d/`. Their examples use arbitrary
dimensions, so convert all dimensions to meters.

Read only the example that is closest to your task:

- `docs/sdk/examples/hollow_shell.py`
- `docs/sdk/examples/section_loft_with_wires.py`
- `docs/sdk/examples/mixed_articulated_assembly.py`
- `docs/sdk/examples/molded_mug.py`
- `docs/sdk/examples/mass_properties.py`
- `docs/sdk/examples/closed_loop_linkage.py`
- `docs/sdk/examples/variable_sweep_and_loft.py`

Run `compile` after meaningful changes. Use failed checks to improve the design. Do not remove
geometry that the prompt requires.

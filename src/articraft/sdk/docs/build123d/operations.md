# Operations

Source:

- https://build123d.readthedocs.io/en/latest/_sources/operations.rst.txt

Use this page for build123d operations such as add, extrude, fillet, chamfer, loft, mirror, offset, revolve, sweep, and split.
Operations are functions that take objects as inputs and transform them into new objects. For example, a 2D Sketch can be extruded to create a 3D Part. All operations are Python functions which can be applied using both the Algebra and Builder APIs. It's important to note that objects created by operations are not affected by `Locations`, meaning their position is determined solely by the input objects used in the operation.

Here are a couple ways to use `extrude`, in Builder and Algebra mode:

```python
with BuildPart() as cylinder:
    with BuildSketch():
        Circle(radius)
    extrude(amount=height)
```

```python
cylinder = extrude(Circle(radius), amount=height)
```

The following table summarizes all of the available operations. Operations marked as 1D are
applicable to BuildLine and Algebra Curve, 2D to BuildSketch and Algebra Sketch, 3D to
BuildPart and Algebra Part.

+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| Operation                                    | Description                        | 0D | 1D | 2D | 3D | Example                           |
+==============================================+====================================+====+====+====+====+===================================+
| `add`              | Add object to builder              |    | ✓  | ✓  | ✓  | `16`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `bounding_box`     | Add bounding box as Shape          |    | ✓  | ✓  | ✓  |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `chamfer`          | Bevel Vertex or Edge               |    |    | ✓  | ✓  | `9`                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `draft`               | Add a draft taper to a part        |    |    |    | ✓  | `examples-cast_bearing_unit` |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `extrude`             | Draw 2D Shape into 3D              |    |    |    | ✓  | `3`                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `fillet`           | Radius Vertex or Edge              |    |    | ✓  | ✓  | `9`                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `full_round`        | Round-off Face along given Edge    |    |    | ✓  |    | `ttt-24-spo-06`              |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `loft`                | Create 3D Shape from sections      |    |    |    | ✓  | `24`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `make_brake_formed`   | Create sheet metal parts           |    |    |    | ✓  |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `make_face`         | Create a Face from Edges           |    |    | ✓  |    | `4`                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `make_hull`         | Create Convex Hull from Edges      |    |    | ✓  |    |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `mirror`           | Mirror about Plane                 |    | ✓  | ✓  | ✓  | `15`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `offset`           | Inset or outset Shape              |    | ✓  | ✓  | ✓  | `25`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `project`          | Project points, lines or Faces     | ✓  | ✓  | ✓  |    |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `project_workplane`   | Create workplane for projection    |    |    |    |    |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `revolve`             | Swing 2D Shape about Axis          |    |    |    | ✓  | `23`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `scale`            | Change size of Shape               |    | ✓  | ✓  | ✓  |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `section`             | Generate 2D slices from 3D Shape   |    |    |    | ✓  |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `split`            | Divide object by Plane             |    | ✓  | ✓  | ✓  | `27`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `sweep`            | Extrude 1/2D section(s) along path |    |    | ✓  | ✓  | `14`                 |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `thicken`             | Expand 2D section(s)               |    |    |    | ✓  |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+
| `trace`             | Convert lines to faces             |    |    | ✓  |    |                                   |
+----------------------------------------------+------------------------------------+----+----+----+----+-----------------------------------+

The following table summarizes all of the selectors that can be used within
the scope of a Builder. Note that they will extract objects from the builder that is
currently within scope without it being explicitly referenced.

+---------------------------------+--------------------------------------+----------------------+
|                                                                        |        Builder       |
+---------------------------------+--------------------------------------+------+--------+------+
| Selector                        | Description                          | Line | Sketch | Part |
+=================================+======================================+======+========+======+
| `edge`      | Select edge from current builder     |  ✓   |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `edges`     | Select edges from current builder    |  ✓   |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `face`      | Select face from current builder     |      |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `faces`     | Select faces from current builder    |      |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `solid`     | Select solid from current builder    |      |        |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `solids`    | Select solids from current builder   |      |        |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `vertex`    | Select vertex from current builder   |  ✓   |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `vertices`  | Select vertices from current builder |  ✓   |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `wire`      | Select wire from current builder     |  ✓   |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+
| `wires`     | Select wires from current builder    |  ✓   |   ✓    |   ✓  |
+---------------------------------+--------------------------------------+------+--------+------+

#### Reference

Function reference: `operations_generic.add`.

Function reference: `operations_generic.bounding_box`.

Function reference: `operations_generic.chamfer`.

Function reference: `operations_part.draft`.

Function reference: `operations_part.extrude`.

Function reference: `operations_generic.fillet`.

Function reference: `operations_sketch.full_round`.

Function reference: `operations_part.loft`.

Function reference: `operations_part.make_brake_formed`.

Function reference: `operations_sketch.make_face`.

Function reference: `operations_sketch.make_hull`.

Function reference: `operations_generic.mirror`.

Function reference: `operations_generic.offset`.

Function reference: `operations_generic.project`.

Function reference: `operations_part.project_workplane`.

Function reference: `operations_part.revolve`.

Function reference: `operations_generic.scale`.

Function reference: `operations_part.section`.

Function reference: `operations_generic.split`.

Function reference: `operations_generic.sweep`.

Function reference: `operations_part.thicken`.

Function reference: `operations_sketch.trace`.

Function reference: `build_common.edge`.

Function reference: `build_common.edges`.

Function reference: `build_common.face`.

Function reference: `build_common.faces`.

Function reference: `build_common.solid`.

Function reference: `build_common.solids`.

Function reference: `build_common.vertex`.

Function reference: `build_common.vertices`.

Function reference: `build_common.wire`.

Function reference: `build_common.wires`.

# Introduction

Source:

- https://build123d.readthedocs.io/en/latest/_sources/introduction.rst.txt
- https://build123d.readthedocs.io/en/latest/_sources/advantages.rst.txt

Use this page for orientation. It explains why build123d is a Python BREP CAD
library and how its style differs from mesh tools, GUI CAD, OpenSCAD, and
CadQuery. It is not an API reference.

## Key Aspects

build123d is meant for precise, parametric, source-controlled CAD models written
in normal Python. Its core advantages come from BREP geometry, Python code,
open-source distribution, source control, tests, and generated documentation.

### BREP Modeling

BREP, or boundary representation, models solids with mathematical topology:
faces, edges, vertices, and their relationships. This is different from mesh
modeling, where surfaces are usually represented by triangles.

BREP systems are useful when a model needs:

- Precise geometric definitions.
- Stable topology for booleans, fillets, chamfers, holes, and other CAD
  features.
- Analytical operations such as collision checks, mass properties, and finite
  element analysis.
- Feature-based edits that keep the design parametric.
- Compact storage compared with dense triangle meshes.

Mesh systems are useful when a model needs:

- Simple surface representation.
- Real-time rendering.
- Easy interchange with animation and visualization tools.
- Freeform or organic surfaces.
- Large point-cloud or scan-derived datasets.

For articraft, prefer build123d BREP geometry for authored parts. Meshes are
an export and collision representation, not the primary authoring model.

### Parametric Models

Parametric CAD models are driven by named dimensions, relationships, and
constraints. A generated build script should make important dimensions explicit
instead of burying numeric constants in repeated geometry calls.

Parametric models help with:

- Reusing a design by changing dimensions.
- Exploring proportions quickly.
- Preserving relationships between features.
- Automating repeated modeling tasks.
- Keeping documentation and generated outputs connected to the design source.

### Python

build123d uses Python as the modeling language. That means generated CAD code can
use ordinary Python names, functions, loops, conditionals, helper data, and tests.

For generated articraft scripts, this matters more than broad Python
cleverness. Keep the code readable, direct, and easy to debug.

### Open Source

build123d is open-source software. The source can be inspected, modified, and
distributed according to its license. This makes it suitable for reproducible
reference projects, examples, and agent-authored code where the implementation
should be inspectable.

### Source Code Control

Source-based CAD works well with Git because the design is represented as text.
That enables normal software workflows:

- Version history.
- Branching and merging.
- Reviewable diffs.
- Remote backup.
- Collaboration.
- Auditing and debugging.

### Automated Testing

Source-based CAD can be tested. Tests can catch geometry regressions, document
expected behavior, and provide a safety net for refactors.

In articraft, every generated script must define `run_tests()` and return a
`TestReport`. Use those tests to check prompt-specific motion, contact,
clearance, and collision behavior.

### Automated Documentation

The upstream build123d docs are built with Sphinx. Automated docs keep reference
material close to source code and can produce HTML, PDF, and other outputs.

For this repo, build123d docs should stay small and agent-readable. Prefer a
clear source-linked Markdown page over a large scraped artifact.

## Advantages Over CadQuery

build123d is designed to feel more like ordinary Python than CadQuery's fluent
method-chain style.

### Standard Python Context Managers

build123d uses Python `with` blocks to define active modeling contexts.

CadQuery style:

```python
pillow_block = (
    cq.Workplane("XY")
    .box(height, width, thickness)
    .edges("|Z")
    .fillet(fillet)
    .faces(">Z")
    .workplane()
)
```

build123d style:

```python
with BuildPart() as pillow_block:
    with BuildSketch() as plan:
        Rectangle(width, height)
        fillet(plan.vertices(), radius=fillet)
    extrude(thickness)
```

The `with` block style allows ordinary Python statements between modeling steps,
including assignments, loops, `print()` calls, and debugger hooks.

### Instantiated Objects

Objects and operations are Python instantiations that interact with the active
context. They can also be assigned to variables for direct use.

```python
with BuildSketch() as plan:
    rectangle = Rectangle(width, height)
    print(rectangle.area)
```

### Operators

build123d adds operators for extracting information from edges and wires:

- `edge @ t` returns a position along an edge or wire.
- `edge % t` returns a tangent along an edge or wire.

The parameter `t` is a float from `0.0` to `1.0`, where `0.0` is the beginning
and `1.0` is the end.

```python
with BuildLine() as outline:
    l5 = Polyline(...)
    l6 = Polyline(...)
    Spline(l5 @ 1, l6 @ 0, tangents=(l5 % 1, l6 % 0))
```

These operators let new features snap to existing geometry without manually
copying coordinates.

### Last Operation Selection

Builder selection methods such as `vertices()`, `edges()`, `faces()`, and
`solids()` can return all matching objects or just the objects changed by the
last operation. This makes local refinement easier, such as filleting only the
edges added by the previous modeling step.

### Extensions

build123d can be extended with custom objects and operations as new classes.
This avoids monkey-patching core classes and keeps custom functionality visible
to IDEs.

### Enums

build123d replaces many string literals with enums. Enums make valid options
more discoverable in editors and reduce typo-prone string arguments.

### Selectors Replaced By Lists

build123d uses Python list-style filtering and sorting instead of CadQuery-style
selector strings.

```python
top = rail.faces().filter_by(Axis.Z)[-1]

outside_vertices = filter(
    lambda v: (v.Y == 0.0 or v.Y == height) and -width / 2 < v.X < width / 2,
    din.vertices(),
)
```


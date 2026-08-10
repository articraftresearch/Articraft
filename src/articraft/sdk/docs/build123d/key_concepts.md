# Key Concepts

Source:

- https://build123d.readthedocs.io/en/latest/_sources/key_concepts.rst.txt
- https://build123d.readthedocs.io/en/latest/_sources/selectors.rst.txt

Use this page to understand the core objects and selection patterns that appear
throughout build123d examples. For API syntax, read the page for the specific
object, builder, or operation.

## Topology

build123d models are made from topological objects. Topology describes how
geometric pieces connect to each other, which is what makes booleans, fillets,
selection, and analysis reliable.

The common topology stack is:

- `Vertex`: a precise point in 3D space.
- `Edge`: a one-dimensional curve or line segment.
- `Wire`: a connected sequence of edges, open or closed.
- `Face`: a bounded two-dimensional surface.
- `Shell`: a connected set of faces.
- `Solid`: a closed, watertight volume.
- `Compound`: a container for multiple shapes.
- `Shape`: the base class for build123d topology objects.

`Shape.show_topology()` can print a tree of a shape's solids, shells, faces,
wires, edges, and vertices. Use it when a selector is hard to reason about.

## Location

A `Location` combines translation and rotation. Shapes, `Axis`, and `Plane`
objects have a `location` property.

`Location` has:

- `position`: translation.
- `orientation`: rotation.

These can be read or assigned directly:

```python
box = Box(1, 1, 1)
box_location = box.location
box_location.position = (1, 2, 3)
box_location.orientation = (30, 40, 50)
box_location.position += (3, 2, 1)
```

## Moving Existing Shapes

Four common methods change a shape location:

- `shape.locate(location)`: absolute move of this object.
- `shape.located(location)`: absolute move of a copy.
- `shape.move(location)`: relative move of this object.
- `shape.moved(location)`: relative move of a copy.

Locations can be combined with `*` and inverted with unary `-`.

## Selectors

Code-based CAD needs selectors instead of mouse clicks. In build123d, selectors
usually mean extracting a `ShapeList` and then filtering, sorting, grouping, or
indexing it.

Builder extraction methods:

| Selector | Applies To | Returns |
| --- | --- | --- |
| `vertices()` | `BuildLine`, `BuildSketch`, `BuildPart` | vertices |
| `edges()` | `BuildLine`, `BuildSketch`, `BuildPart` | edges |
| `wires()` | `BuildLine`, `BuildSketch`, `BuildPart` | wires |
| `faces()` | `BuildSketch`, `BuildPart` | faces |
| `solids()` | `BuildPart` | solids |

ShapeList operators and equivalent methods:

| Operator | Method | Meaning | Example |
| --- | --- | --- | --- |
| `>` | `sort_by` | Sort by an axis or sort key. | `part.vertices() > Axis.Z` |
| `<` | `sort_by` | Reverse sort. | `part.faces() < Axis.Z` |
| `>>` | `group_by` | Group and return the last group. | `part.solids() >> Axis.X` |
| `<<` | `group_by` | Group and return the first group. | `part.faces() << Axis.Y` |
| `\|` | `filter_by` | Filter by axis, plane, geometry type, or predicate. | `part.faces() \| Axis.Z` |
| `[]` | list access | Use normal Python indexing and slicing. | `part.faces()[-2:]` |

Common selector operands:

- `Axis`: predefined axes such as `Axis.X`, `Axis.Y`, and `Axis.Z`, or custom
  axes.
- `Plane`: coordinate systems; filtering by a plane finds parallel faces or
  edges.
- `GeomType`: geometry kinds such as line, circle, plane, cylinder, sphere, and
  torus.
- `SortBy`: properties such as length, radius, area, volume, and distance.

## ShapeList

Builder selection methods return `ShapeList`, a `list` subclass with CAD-specific
sorting and filtering helpers.

You can also use normal Python filtering for custom selectors:

```python
outside_vertices = filter(
    lambda v: (v.Y == 0.0 or v.Y == height) and -overall_width / 2 < v.X < overall_width / 2,
    din.vertices(),
)
```

`filter_by()` also accepts predicates:

```python
obj = Box(1, 1, 1) - Cylinder(0.2, 1)
faces_with_holes = obj.faces().filter_by(lambda f: f.inner_wires())
```


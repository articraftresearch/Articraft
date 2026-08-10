# Surface Modeling

Source:

- https://build123d.readthedocs.io/en/latest/_sources/tutorial_surface_modeling.rst.txt

Use this page for an overview of direct surface modeling and build123d face construction tools.
Surface modeling refers to the direct creation and manipulation of the skin of a 3D
object—its bounding faces—rather than starting from volumetric primitives or solid
operations.

Instead of defining a shape by extruding or revolving a 2D profile to fill a volume,
surface modeling focuses on building the individual curved or planar faces that together
define the outer boundary of a part. This approach allows for precise control of complex
freeform geometry such as aerodynamic surfaces, boat hulls, or organic transitions that
cannot easily be expressed with simple parametric solids.

In build123d, as in other CAD kernels based on BREP (Boundary Representation) modeling,
all solids are ultimately defined by their boundaries: a hierarchy of faces, edges, and
vertices. Each face represents a finite patch of a geometric surface (plane, cylinder,
Bézier patch, etc.) bounded by one or more edge loops or wires. When adjacent faces share
edges consistently and close into a continuous boundary, they form a manifold
`Shell`—the watertight surface of a volume. If this shell is properly
oriented and encloses a finite region of space, the model becomes a solid.

Surface modeling therefore operates at the most fundamental level of BREP construction.
Rather than relying on higher-level modeling operations to implicitly generate faces,
it allows you to construct and connect those faces explicitly. This provides a path to
build geometry that blends analytical and freeform shapes seamlessly, with full control
over continuity, tangency, and curvature across boundaries.

This section provides:
- A concise overview of surface‑building tools in build123d
- Hands‑on tutorials, from fundamentals to advanced techniques like Gordon surfaces

Methods on `Face` for creating non‑planar surfaces:

* `make_bezier_surface`
* `make_gordon_surface`
* `make_surface`
* `make_surface_from_array_of_points`
* `make_surface_from_curves`
* `make_surface_patch`

**Note**

Surface modeling is an advanced technique. Robust results usually come from
reusing the same `Edge` objects across adjacent faces and
ensuring the final `Shell` is *water‑tight* or *manifold* (no gaps).

   :maxdepth: 1

   tutorial_surface_heart_token.rst
   tutorial_spitfire_wing_gordon.rst

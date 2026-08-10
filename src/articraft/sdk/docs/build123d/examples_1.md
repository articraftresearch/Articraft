# The build123d Examples

Source:

- https://build123d.readthedocs.io/en/latest/_sources/examples_1.rst.txt

Use this page for a catalog of upstream build123d examples, with local verbatim copies under docs/sdk/build123d/examples/.

Local example files:

- `docs/sdk/build123d/examples/benchy.py`
- `docs/sdk/build123d/examples/bicycle_tire.py`
- `docs/sdk/build123d/examples/boxes_on_faces.py`
- `docs/sdk/build123d/examples/boxes_on_faces_algebra.py`
- `docs/sdk/build123d/examples/bracelet.py`
- `docs/sdk/build123d/examples/build123d_customizable_logo.py`
- `docs/sdk/build123d/examples/build123d_customizable_logo_algebra.py`
- `docs/sdk/build123d/examples/build123d_logo.py`
- `docs/sdk/build123d/examples/build123d_logo_algebra.py`
- `docs/sdk/build123d/examples/canadian_flag.py`
- `docs/sdk/build123d/examples/canadian_flag_algebra.py`
- `docs/sdk/build123d/examples/cast_bearing_unit.py`
- `docs/sdk/build123d/examples/circuit_board.py`
- `docs/sdk/build123d/examples/circuit_board_algebra.py`
- `docs/sdk/build123d/examples/clock.py`
- `docs/sdk/build123d/examples/clock_algebra.py`
- `docs/sdk/build123d/examples/custom_sketch_objects.py`
- `docs/sdk/build123d/examples/custom_sketch_objects_algebra.py`
- `docs/sdk/build123d/examples/din_rail.py`
- `docs/sdk/build123d/examples/din_rail_algebra.py`
- `docs/sdk/build123d/examples/dual_color_3mf.py`
- `docs/sdk/build123d/examples/extrude.py`
- `docs/sdk/build123d/examples/extrude_algebra.py`
- `docs/sdk/build123d/examples/fast_grid_holes.py`
- `docs/sdk/build123d/examples/handle.py`
- `docs/sdk/build123d/examples/handle_algebra.py`
- `docs/sdk/build123d/examples/heat_exchanger.py`
- `docs/sdk/build123d/examples/heat_exchanger_algebra.py`
- `docs/sdk/build123d/examples/holes.py`
- `docs/sdk/build123d/examples/holes_algebra.py`
- `docs/sdk/build123d/examples/intersecting_chamfers.py`
- `docs/sdk/build123d/examples/intersecting_chamfers_algebra.py`
- `docs/sdk/build123d/examples/intersecting_pipes.py`
- `docs/sdk/build123d/examples/intersecting_pipes_algebra.py`
- `docs/sdk/build123d/examples/joints.py`
- `docs/sdk/build123d/examples/joints_algebra.py`
- `docs/sdk/build123d/examples/key_cap.py`
- `docs/sdk/build123d/examples/key_cap_algebra.py`
- `docs/sdk/build123d/examples/lego.py`
- `docs/sdk/build123d/examples/lego_algebra.py`
- `docs/sdk/build123d/examples/loft.py`
- `docs/sdk/build123d/examples/loft_algebra.py`
- `docs/sdk/build123d/examples/low_poly_benchy.stl`
- `docs/sdk/build123d/examples/maker_coin.py`
- `docs/sdk/build123d/examples/mixed_algebra_context.py`
- `docs/sdk/build123d/examples/multiple_workplanes.py`
- `docs/sdk/build123d/examples/multiple_workplanes_algebra.py`
- `docs/sdk/build123d/examples/packed_boxes.py`
- `docs/sdk/build123d/examples/pegboard_j_hook.py`
- `docs/sdk/build123d/examples/pegboard_j_hook_algebra.py`
- `docs/sdk/build123d/examples/pillow_block.py`
- `docs/sdk/build123d/examples/pillow_block_algebra.py`
- `docs/sdk/build123d/examples/platonic_solids.py`
- `docs/sdk/build123d/examples/playing_cards.py`
- `docs/sdk/build123d/examples/playing_cards_algebra.py`
- `docs/sdk/build123d/examples/projection.py`
- `docs/sdk/build123d/examples/projection_algebra.py`
- `docs/sdk/build123d/examples/python_logo.py`
- `docs/sdk/build123d/examples/roller_coaster.py`
- `docs/sdk/build123d/examples/roller_coaster_algebra.py`
- `docs/sdk/build123d/examples/shamrock.py`
- `docs/sdk/build123d/examples/stud_wall.py`
- `docs/sdk/build123d/examples/tea_cup.py`
- `docs/sdk/build123d/examples/tea_cup_algebra.py`
- `docs/sdk/build123d/examples/toy_truck.py`
- `docs/sdk/build123d/examples/ttt_sm_hanger.py`
- `docs/sdk/build123d/examples/vase.py`
- `docs/sdk/build123d/examples/vase_algebra.py`

### Benchy

Image file: `docs/sdk/build123d/assets/examples/example_benchy_01.png`.

The Benchy examples shows how to import a STL model as a `Solid` object with the class `Mesher` and
modify it by replacing chimney with a BREP version.

- Benchy STL model: `low_poly_benchy.stl` (`../examples/low_poly_benchy.stl`)

     *Attribution:*
     The low-poly-benchy used in this example is by `reddaugherty`, see
     https://www.printables.com/model/151134-low-poly-benchy.

### Gallery

Image file: `docs/sdk/build123d/assets/examples/example_benchy_02.png`.

Image file: `docs/sdk/build123d/assets/examples/example_benchy_03.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/benchy.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Bicycle Tire

Image file: `docs/sdk/build123d/assets/examples/bicycle_tire.png`.

This example demonstrates how to model a realistic bicycle tire with a
patterned tread using build123d. The key concept showcased here is the
use of wrap_faces to project 2D planar geometry onto a curved 3D
surface.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/bicycle_tire.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Bracelet

Image file: `docs/sdk/build123d/assets/examples/bracelet.png`.

Doubly-curved bracelet with an embossed label

This model is a good “stress test” for OCCT because most of the final boundary
surfaces are *freeform* (not analytic planes/cylinders/spheres). The geometry
is assembled from:

- a swept center section (using a curved solid end-face as the sweep profile)
- two freeform “tip caps” built as Gordon surfaces (network of curves)
- an optional embossed text label projected onto a curved solid
- alignment holes for splitting/printing/assembly

Key techniques demonstrated:

- using location_at/position_at/tangent (%) to extract local frames & tangents
- projecting curves onto a non-planar surface to create “true” 3D guide curves
- Gordon surfaces to build high-quality doubly-curved skins
- projecting faces (text) onto a complex solid and thickening them

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/bracelet.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Former build123d Logo

Image file: `docs/sdk/build123d/assets/examples/example_build123d_logo_01.png`.

This example creates the former build123d logo (new logo was created in the end of 2023).

Using text and lines to create the first build123d logo.
The builder mode example also generates the SVG file `logo.svg`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/build123d_logo.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/build123d_logo_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Cast Bearing Unit

Image file: `docs/sdk/build123d/assets/examples/cast_bearing_unit.png`.

This example demonstrates the creation of a castable flanged bearing housing
using the `draft` operation to add appropriate draft angles for mold release.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/cast_bearing_unit.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Canadian Flag Blowing in The Wind

Image file: `docs/sdk/build123d/assets/examples/example_canadian_flag_01.png`.

A Canadian Flag blowing in the wind created by projecting planar faces onto a non-planar face (the_wind).

This example also demonstrates building complex lines that snap to existing features.

### More Images

Image file: `docs/sdk/build123d/assets/examples/example_canadian_flag_02.png`.

Image file: `docs/sdk/build123d/assets/examples/example_canadian_flag_03.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/canadian_flag.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/canadian_flag_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Circuit Board With Holes

Image file: `docs/sdk/build123d/assets/examples/example_circuit_board_01.png`.

This example demonstrates placing holes around a part.

- Builder mode uses `Locations` context to place the positions.
- Algebra mode uses `product` and `range` to calculate the positions.

### More Images

Image file: `docs/sdk/build123d/assets/examples/example_circuit_board_02.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/circuit_board.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/circuit_board_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Clock Face

Image file: `docs/sdk/build123d/assets/examples/clock_face.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/clock.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/clock_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

The Python code utilizes the build123d library to create a 3D model of a clock face.
It defines a minute indicator with arcs and lines, applying fillets, and then
integrates it into the clock face sketch. The clock face includes a circular outline,
hour labels, and slots at specified positions. The resulting 3D model represents
a detailed and visually appealing clock design.

`PolarLocations` are used to position features on the clock face.

### Fast Grid Holes

Image file: `docs/sdk/build123d/assets/examples/fast_grid_holes.png`.

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/fast_grid_holes.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example demonstrates an efficient approach to creating a large number of holes
(625 in this case) in a planar part using build123d.

Instead of modeling and subtracting 3D solids for each hole—which is computationally
expensive—this method constructs a 2D Face from an outer perimeter wire and a list of
hole wires. The entire face is then extruded in a single operation to form the final
3D object. This approach significantly reduces modeling time and complexity.

The hexagonal hole pattern is generated using HexLocations, and each location is
populated with a hexagonal wire. These wires are passed directly to the Face constructor
as holes. On a typical Linux laptop, this script completes in approximately 1.02 seconds,
compared to substantially longer runtimes for boolean subtraction of individual holes in 3D.

### Handle

Image file: `docs/sdk/build123d/assets/examples/handle.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/handle.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/handle_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example demonstrates multisection sweep creating a drawer handle.

### Heat Exchanger

Image file: `docs/sdk/build123d/assets/examples/heat_exchanger.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/heat_exchanger.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/heat_exchanger_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example creates a model of a parametric heat exchanger core. The positions
of the tubes are defined with `HexLocations` and further
limited to fit within the circular end caps. The ends of the tubes are filleted
to the end plates to simulate welding.

### Key Cap

Image file: `docs/sdk/build123d/assets/examples/key_cap.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/key_cap.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/key_cap_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example demonstrates the design of a Cherry MX key cap by using
extrude with a taper and extrude until next.

### Maker Coin

Image file: `docs/sdk/build123d/assets/examples/maker_coin.png`.

This example creates the maker coin as defined by Angus on the Maker's Muse
YouTube channel. There are two key features:

#. the use of `DoubleTangentArc` to create a smooth
   transition from the central dish to the outside arc, and

#. embossing the text into the top of the coin not just as a simple
   extrude but from a projection which results in text with even depth.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/maker_coin.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Multi-Sketch Loft

Image file: `docs/sdk/build123d/assets/examples/loft.png`.

This example demonstrates lofting a set of sketches, selecting
the top and bottom by type, and shelling.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/loft.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/loft_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Peg Board Hook

Image file: `docs/sdk/build123d/assets/examples/peg_board_hook.png`.

This script creates a a J-shaped pegboard hook. These hooks are commonly used for
organizing tools in garages, workshops, or other spaces where tools and equipment
need to be stored neatly and accessibly. The hook is created by defining a complex
path and then sweeping it to define the hook. The sides of the hook are flattened
to aid 3D printing.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/pegboard_j_hook.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/pegboard_j_hook_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Platonic Solids

Image file: `docs/sdk/build123d/assets/examples/platonic_solids.png`.

This example creates a custom Part object PlatonicSolid.

Platonic solids are five three-dimensional shapes that are highly symmetrical,
known since antiquity and named after the ancient Greek philosopher Plato.
These solids are unique because their faces are congruent regular polygons,
with the same number of faces meeting at each vertex. The five Platonic solids
are the tetrahedron (4 triangular faces), cube (6 square faces), octahedron
(8 triangular faces), dodecahedron (12 pentagonal faces), and icosahedron
(20 triangular faces). Each solid represents a unique way in which identical
polygons can be arranged in three dimensions to form a convex polyhedron,
embodying ideals of symmetry and balance.

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/platonic_solids.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Playing Cards

Image file: `docs/sdk/build123d/assets/examples/playing_cards.png`.

This example creates a customs Sketch objects: Club, Spade, Heart, Diamond,
and PlayingCard in addition to a two part playing card box which has suit
cutouts in the lid. The four suits are created with Bézier curves that were
imported as code from an SVG file and modified to the code found here.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/playing_cards.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Stud Wall

Image file: `docs/sdk/build123d/assets/examples/stud_wall.png`.

This example demonstrates creating custom `Part` objects and putting them into
assemblies. The custom object is a `Stud` used in the building industry while
the assembly is a `StudWall` created from copies of `Stud` objects for efficiency.
Both the `Stud` and `StudWall` objects use `RigidJoints` to define snap points which
are used to position all of objects.

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/stud_wall.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Tea Cup

Image file: `docs/sdk/build123d/assets/examples/tea_cup.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/tea_cup.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/tea_cup_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example demonstrates the creation a tea cup, which serves as an example of
constructing complex, non-flat geometrical shapes programmatically.

The tea cup model involves several CAD techniques, such as:

* Revolve Operations: There is 1 occurrence of a revolve operation. This is used
  to create the main body of the tea cup by revolving a profile around an axis,
  a common technique for generating symmetrical objects like cups.
* Sweep Operations: There are 2 occurrences of sweep operations. The handle are
  created by sweeping a profile along a path to generate non-planar surfaces.
* Offset/Shell Operations: the bowl of the cup is hollowed out with the offset
  operation leaving the top open.
* Fillet Operations: There is 1 occurrence of a fillet operation which is used to
  round the edges for aesthetic improvement and to mimic real-world objects more
  closely.

### Toy Truck

Image file: `docs/sdk/build123d/assets/examples/toy_truck.png`.

Image file: `docs/sdk/build123d/assets/examples/toy_truck_picture.jpg`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/toy_truck.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example demonstrates how to design a toy truck using BuildPart and
BuildSketch in Builder mode. The model includes a detailed body, cab, grill,
and bumper, showcasing techniques like sketch reuse, symmetry, tapered
extrusions, selective filleting, and the use of joints for part assembly.
Ideal for learning complex part construction and hierarchical modeling in
build123d.

### Vase

Image file: `docs/sdk/build123d/assets/examples/vase.png`.

### Builder Reference Implementation (Builder Mode)

Code reference: `docs/sdk/build123d/examples/vase.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

### Algebra Reference Implementation (Algebra Mode)

Code reference: `docs/sdk/build123d/examples/vase_algebra.py`.

Include options:
- `:language: build123d`
- `:start-after: [Code]`
- `:end-before: [End]`

This example demonstrates the build123d techniques involving the creation of a vase.
Specifically, it showcases the processes of revolving a sketch, shelling
(creating a hollow object by removing material from its interior), and
selecting edges by position range and type for the application of fillets
(rounding off the edges).

* Sketching: Drawing a 2D profile or outline that represents the side view of
  the vase.
* Revolving: Rotating the sketch around an axis to create a 3D object. This
  step transforms the 2D profile into a 3D vase shape.
* Offset/Shelling: Removing material from the interior of the solid vase to
  create a hollow space, making it resemble a real vase more closely.
* Edge Filleting: Selecting specific edges of the vase for filleting, which
  involves rounding those edges. The edges are selected based on their position
  and type.

### {name-of-your-example-with-spaces}

Image file: `docs/sdk/build123d/assets/examples/example_{name-of-your-example}_01.{extension}`.

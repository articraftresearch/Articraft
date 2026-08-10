# CAD Object Centers

Source:

- https://build123d.readthedocs.io/en/latest/_sources/center.rst.txt

Use this page for center calculations for CAD objects, including bounding-box, geometry, and mass centers.
Finding the center of a CAD object is a surprisingly complex operation.  To illustrate
let's consider two examples: a simple isosceles triangle and a curved line (their bounding
boxes are shown with dashed lines):

Image file: `docs/sdk/build123d/assets/center.svg`.

Image file: `docs/sdk/build123d/assets/one_d_center.svg`.

One can see that there is are significant differences between the different types of
centers. To allow the designer to choose the center that makes the most sense for the given
shape there are three possible values for the `CenterOf` Enum:

==============================  ======  == == == ========
`CenterOf`  Symbol  1D 2D 3D Compound
==============================  ======  == == == ========
CenterOf.BOUNDING_BOX           □       ✓  ✓  ✓  ✓
CenterOf.GEOMETRY               △       ✓  ✓
CenterOf.MASS                   ○       ✓  ✓  ✓  ✓
==============================  ======  == == == ========

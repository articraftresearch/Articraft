# Algebraic definition

Source:

- https://build123d.readthedocs.io/en/latest/_sources/algebra_definition.rst.txt

Use this page for the formal object and placement algebra behind build123d algebra mode.
## Algebraic definition

### Objects and arithmetic

**Set definitions:**

`C^3` is the set of all `Part` objects `p` with `p._dim = 3`

`C^2` is the set of all `Sketch` objects `s` with `s._dim = 2`

`C^1` is the set of all `Curve` objects `c` with `c._dim = 1`

**Neutral elements:**

`c^3_0` is the empty `Part` object `p0 = Part()` with `p0._dim = 3` and `p0.wrapped = None`

`c^2_0` is the empty `Sketch` object `s0 = Sketch()` with `s0._dim = 2` and `s0.wrapped = None`

`c^1_0` is the empty `Curve` object `c0 = Curve()` with `c0._dim = 1` and `c0.wrapped = None`

**Sets of predefined basic shapes:**

`B^3 := \lbrace` `Part`, `Box`, `Cylinder`, `Cone`, `Sphere`, `Torus`, `Wedge`, `Hole`, `CounterBoreHole`, `CounterSinkHole` `\rbrace`

`B^2 := \lbrace` `Sketch`, `Rectangle`, `Circle`, `Ellipse`, `Rectangle`, `Polygon`, `RegularPolygon`, `Text`, `Trapezoid`, `SlotArc`, `SlotCenterPoint`, `SlotCenterToCenter`, `SlotOverall` `\rbrace`

`B^1 := \lbrace` `Curve`, `Bezier`, `FilletPolyline`, `PolarLine`, `Polyline`, `Spline`, `Helix`, `CenterArc`, `EllipticalCenterArc`, `ParabolicCenterArc`, `HyperbolicCenterArc`, `RadiusArc`, `SagittaArc`, `TangentArc`, `ThreePointArc`, `JernArc` `\rbrace`

with `B^3 \subset C^3, B^2 \subset C^2` and `B^1 \subset C^1`

**Operations:**

`+: C^n \times C^n \rightarrow C^n` with `(a,b) \mapsto a + b`,  `\;` for `n=1,2,3`

    `\; a + b :=` `a.fuse(b)` for each operation

`-: C^n \rightarrow C^n` with `a \mapsto -a`,  `\;` for `n=1,2,3`

    `\; b + (-a) :=` `b.cut(a)` for each operation (implicit definition)

`\&: C^n \times C^n \rightarrow C^n` with `(a,b) \mapsto a \; \& \; b`,  `\;` for `n=2,3`

    `\; a \; \& \; b :=` `a.intersect(b)` for each operation

    * `\&` is not defined for `n=1` in build123d
    * The following relationship holds: `a \; \& \; b = (a + b) + -(a + (-b)) + -(b + (-a))`

**Abelian groups**

`( C^n, \; c^n_0, \; +, \; -)` `\;`  are abelian groups for `n=1,2,3`.

* The implementation `a - b = a.cut(b)` needs to be read as `a + (-b)` since the group does not have a binary `-` operation. As such, `a - (b - c) = a + -(b + -c)) \ne a - b + c`
* This definition also includes that neither `-` nor `&` are commutative.

### Locations, planes and location arithmetic

**Set definitions:**

`L  := \lbrace` `Location((x, y, z), (a, b, c))` `: x,y,z \in R \land a,b,c \in R \rbrace\;`

    with `a,b,c` being angles in degrees.

`P  := \lbrace` `Plane(o, x, z)` `: o,x,z ∈ R^3 \land \|x\| = \|z\| = 1\rbrace`

    with `o` being the origin and `x`, `z` the x- and z-direction of the plane.

Neutral element: `\; l_0 \in L`: `Location()`

**Operations:**

`*: L \times L \rightarrow L` with `(l_1,l_2) \mapsto l_1 * l_2`

    `\; l_1 * l_2 :=`  `l1 * l2` (multiply two locations)

`*: P \times L \rightarrow P` with `(p,l) \mapsto p * l`

    `\; p * l :=` `Plane(p.location * l)` (move plane `p \in P` to location `l \in L`)

Inverse element: `\; l^{-1} \in L`: `l.inverse()`

**Placing objects onto planes**

`*: P \times C^n  \rightarrow C^n \;`  with `(p,c) \mapsto p * c`,  `\;` for `n=1,2,3`

    Locate an object `c \in C^n` onto plane `p \in P`, i.e. `c.moved(p.location)`

**Placing objects at locations**

`*: L \times C^n  \rightarrow C^n \;`  with `(l,c) \mapsto l * c`,  `\;` for `n=1,2,3`

    Locate an object `c \in C^n` at location `l \in L`, i.e. `c.moved(l)`

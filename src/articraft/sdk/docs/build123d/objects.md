# Objects

Source:

- https://build123d.readthedocs.io/en/latest/_sources/objects.rst.txt

Use this page for object constructors for 1D curves, 2D sketches, and 3D parts, including common primitives and object placement notes.
Objects are Python classes that take parameters as inputs and create 1D, 2D or 3D Shapes.
For example, a `Torus` is defined by a major and minor radii. In
Builder mode, objects are positioned with `Locations` while in Algebra mode, objects
are positioned with the `*` operator and shown in these examples:

```python
with BuildPart() as disk:
    with BuildSketch():
        Circle(a)
        with Locations((b, 0.0)):
            Rectangle(c, c, mode=Mode.SUBTRACT)
        with Locations((0, b)):
            Circle(d, mode=Mode.SUBTRACT)
    extrude(amount=c)
```

```python
sketch = Circle(a) - Pos(b, 0.0) * Rectangle(c, c) - Pos(0.0, b) * Circle(d)
disk = extrude(sketch, c)
```

The following sections describe the 1D, 2D and 3D objects:

### Align

2D/Sketch and 3D/Part objects can be aligned relative to themselves, either centered, or justified
right or left of each Axis. The following diagram shows how this alignment works in 2D:

Image file: `docs/sdk/build123d/assets/align.svg`.

For example:

```python
with BuildSketch():
    Circle(1, align=(Align.MIN, Align.MIN))
```

creates a circle who's minimal X and Y values are on the X and Y axis and is located in the top right corner.
The `Align` enum has values: `MIN`, `CENTER` and `MAX`.

In 3D the `align` parameter also contains a Z align value but otherwise works in the same way.

Note that the `align` will also accept a single `Align` value which will be used on all axes -
as shown here:

```python
with BuildSketch():
    Circle(1, align=Align.MIN)
```

### Mode

With the Builder API the `mode` parameter controls how objects are combined with lines, sketches, or parts
under construction.  The `Mode` enum has values:

* `ADD`: fuse this object to the object under construction
* `SUBTRACT`: cut this object from the object under construction
* `INTERSECT`: intersect this object with the object under construction
* `REPLACE`: replace the object under construction with this object
* `PRIVATE`: don't interact with the object under construction at all

The Algebra API doesn't use the `mode` parameter - users combine objects with operators.

### 1D Objects

The following objects all can be used in BuildLine contexts. Note that
1D objects are not affected by `Locations` in Builder mode.

#### Reference

Class reference: `BaseLineObject`.

Class reference: `Airfoil`.

Class reference: `Bezier`.

Class reference: `BlendCurve`.

Class reference: `BSpline`.

Class reference: `CenterArc`.

Class reference: `ConstrainedArcs`.

Class reference: `ConstrainedLines`.

Class reference: `DoubleTangentArc`.

Class reference: `EllipticalCenterArc`.

Class reference: `EllipticalStartArc`.

Class reference: `ParabolicCenterArc`.

Class reference: `HyperbolicCenterArc`.

Class reference: `FilletPolyline`.

Class reference: `Helix`.

Class reference: `IntersectingLine`.

Class reference: `JernArc`.

Class reference: `Line`.

Class reference: `PolarLine`.

Class reference: `Polyline`.

Class reference: `RadiusArc`.

Class reference: `SagittaArc`.

Class reference: `Spline`.

Class reference: `TangentArc`.

Class reference: `ThreePointArc`.

Class reference: `ArcArcTangentLine`.

Class reference: `ArcArcTangentArc`.

Image file: `docs/sdk/build123d/assets/objects/arcarctangentarc_keep_table.png`.

Class reference: `PointArcTangentLine`.

Class reference: `PointArcTangentArc`.

### 2D Objects

#### Reference

Class reference: `BaseSketchObject`.

Class reference: `drafting.Arrow`.

Class reference: `drafting.ArrowHead`.

Class reference: `Circle`.

Class reference: `drafting.DimensionLine`.

Class reference: `Ellipse`.

Class reference: `drafting.ExtensionLine`.

Class reference: `Polygon`.

Class reference: `Rectangle`.

Class reference: `RectangleRounded`.

Class reference: `RegularPolygon`.

Class reference: `SlotArc`.

Class reference: `SlotCenterPoint`.

Class reference: `SlotCenterToCenter`.

Class reference: `SlotOverall`.

Class reference: `drafting.TechnicalDrawing`.

Class reference: `Text`.

Class reference: `Trapezoid`.

Class reference: `Triangle`.

### 3D Objects

#### Reference

Class reference: `BasePartObject`.

Class reference: `Box`.

Class reference: `Cone`.

Class reference: `ConvexPolyhedron`.

Class reference: `CounterBoreHole`.

Class reference: `CounterSinkHole`.

Class reference: `Cylinder`.

Class reference: `Hole`.

Class reference: `Sphere`.

Class reference: `Torus`.

Class reference: `Wedge`.

### Text

Included upstream source omitted here: `objects/text.rst`.

### Custom Objects

All of the objects presented above were created using one of three base object classes:
`BaseLineObject` , `BaseSketchObject` , and
`BasePartObject` .  Users can use these base object classes to
easily create custom objects that have all the functionality of the core objects.

Image file: `docs/sdk/build123d/assets/card_box.svg`.

Here is an example of a custom sketch object specially created as part of the design of
this playing card storage box (`see the playing_cards.py example` (`../examples/playing_cards.py`)):

Code reference: `docs/sdk/build123d/examples/playing_cards.py`.

Include options:
- `:language: build123d`
- `:start-after: [Club]`
- `:end-before: [Club]`

Here the new custom object class is called `Club` and it's a sub-class of
`BaseSketchObject` .  The `__init__` method contains all
of the parameters used to instantiate the custom object, specially a `height`,
`rotation`, `align`, and `mode` - your objects may contain a sub or super set of
these parameters but should always contain a `mode` parameter such that it
can be combined with a builder's object.

Next is the creation of the object itself, in this case a sketch of the club suit.

The final line calls the `__init__` method of the super class - i.e.
`BaseSketchObject` with its parameters.

That's it, now the `Club` object can be used anywhere a `Circle`
would be used - with either the Algebra or Builder API.

Image file: `docs/sdk/build123d/assets/buildline_example_6.svg`.

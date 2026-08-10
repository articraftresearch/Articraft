# Moving Objects

Source:

- https://build123d.readthedocs.io/en/latest/_sources/moving_objects.rst.txt

Use this page when deciding how to place, rotate, or relocate build123d shapes.
The right approach depends on whether code is in builder mode, algebra mode, or
direct shape manipulation.

## Builder Mode

In builder mode, location must be established before objects are created. Use a
location context around the object constructor.

Common tools:

- `Locations`: explicit locations.
- `GridLocations`: rectangular grid placement.
- `PolarLocations`: radial placement.
- `HexLocations`: hexagonal grid placement.

Example:

```python
with BuildPart():
    with Locations((10, 20, 30)):
        Box(5, 5, 5)
```

Do not create a builder object and then call `.moved(...)` on the temporary
return value. By then, the object has already been added to the builder.

## Algebra Mode

In algebra mode, movement is expressed with placement operators.

```python
moved_box = Pos(10, 20, 30) * Box(5, 5, 5)
```

Combine a plane and a position to place a shape relative to that plane:

```python
placed_box = Plane.XZ * Pos(10, 20, 30) * Box(5, 5, 5)
```

Use `Rotation` or `Rot` for orientation:

```python
rotated_box = Rotation(45, 0, 0) * box
```

## Direct Manipulation

Existing shapes can be moved by changing their location-related properties or by
calling movement methods.

Set an absolute position:

```python
shape.position = (x, y, z)
```

Apply a relative position change:

```python
shape.position += (x, y, z)
shape.position -= (x, y, z)
```

Set an absolute orientation:

```python
shape.orientation = (x_angle, y_angle, z_angle)
```

Apply a relative orientation change:

```python
shape.orientation += (x_angle, y_angle, z_angle)
shape.orientation -= (x_angle, y_angle, z_angle)
```

## Movement Methods

Relative move of the same object:

```python
shape.move(location)
```

Relative move of a copy:

```python
relocated_shape = shape.moved(location)
```

Absolute move of the same object:

```python
shape.locate(location)
```

Absolute move of a copy:

```python
relocated_shape = shape.located(location)
```

## Translation And Rotation Methods

`translate()` and `rotate()` transform a shape. These methods can optionally
transform the underlying geometry, which is slower and can be more fragile than
changing the shape's `Location`.

Translate relative to the current position:

```python
relocated_shape = shape.translate((x, y, z))
```

Rotate around an axis:

```python
rotated_shape = shape.rotate(axis, angle_in_degrees)
```


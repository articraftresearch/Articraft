# Key Concepts: Algebra Mode

Source:

- https://build123d.readthedocs.io/en/latest/_sources/key_concepts_algebra.rst.txt

Use this page when reading or writing build123d algebra-style shape
expressions. Builder mode is usually clearer for longer generated models, but
algebra mode is useful for compact shape composition and placement.

## Core Ideas

Algebra mode works with objects such as `Shape`, `Part`, `Sketch`, and `Curve`.
It has two main concepts:

- Object arithmetic: combine shapes with boolean-style operators.
- Placement arithmetic: place shapes with planes and locations using `*`.

## Object Arithmetic

Create shapes:

```python
b = Box(1, 2, 3)
c = Cylinder(0.2, 5)
```

Fuse shapes:

```python
r = Box(1, 2, 3) + Cylinder(0.2, 5)
```

Cut one shape from another:

```python
r = Box(1, 2, 3) - Cylinder(0.2, 5)
```

Intersect shapes:

```python
r = Box(1, 2, 3) & Cylinder(0.2, 5)
```

The resulting objects can be displayed by viewers that support build123d
compounds.

## Placement Arithmetic

Algebra objects do not usually take location or rotation arguments directly.
Instead, place them with `*`.

Common forms:

```python
plane * shape
location * shape
plane * location * shape
```

`Plane.XY * shape` is equivalent to the default position on the XY plane, so it
can usually be omitted:

```python
Plane.XY * Box(1, 2, 3)
Box(1, 2, 3)
```

Place a box at Y=1:

```python
Plane.XY * Pos(0, 1, 0) * Box(1, 2, 3)
Pos(0, 1, 0) * Box(1, 2, 3)
Pos(Y=1) * Box(1, 2, 3)
```

Place a box on the XZ plane:

```python
Plane.XZ * Box(1, 2, 3)
```

Place relative to a plane:

```python
Plane.XZ * Pos(1, 2, 3) * Box(1, 2, 3)
```

Place and rotate relative to a plane:

```python
Plane.XZ * Pos(1, 2, 3) * Rot(0, 100, 45) * Box(1, 2, 3)
Location((1, 2, 3), (0, 100, 45)) * Box(1, 2, 3)
```

Order matters. This rotates on the plane, then moves relative to that plane:

```python
Plane.XZ * Rot(0, 100, 45) * Pos(0, 1, 2) * Box(1, 2, 3)
```

## Combining Object And Placement Arithmetic

Placement and object arithmetic can be combined:

```python
b = Plane.XZ * Rot(X=30) * Box(1, 2, 3) + Plane.YZ * Pos(X=-1) * Cylinder(0.2, 5)
```

In Python, `*` binds more tightly than `+`, `-`, and `&`, so this kind of
expression often does not need extra parentheses.


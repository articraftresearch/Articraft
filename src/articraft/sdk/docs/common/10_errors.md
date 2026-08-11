# Errors

The public SDK exports `SDKError` and `ValidationError`.

```python
from articraft.sdk import SDKError, ValidationError
```

## `SDKError`

`SDKError` is the base exception for errors defined by the Articraft SDK.
Catch it when code should handle any SDK validation error in one place.

## `ValidationError`

`ValidationError` is a subclass of `SDKError`. The SDK raises it when an
object, part, shape, mesh, articulation, or named test selector does not satisfy
the public contract. A failed authored assertion is recorded in `TestReport`
instead of raising an exception.

Examples include these cases:

- A required name is empty.
- A part or shape name is duplicated where it must be unique.
- A shape is empty, invalid, or has invalid mesh indices.
- A color has the wrong length or a value outside zero through one.
- An articulation refers to an unknown part.
- Motion limits do not match the articulation type.
- The part graph does not have exactly one root.

## When validation runs

The SDK validates data at several points.

`RigidBodyAssembly(...)`, `RigidBody(...)`, `JointFrame(...)`, and `JointDOF(...)`
validate their own values when you create them. `Part.add(...)` validates the
shape name, geometry, and color before it stores the shape.
`model.articulation(...)` validates the articulation and checks that both parts
already exist.

`model.validate()` checks the complete model. It checks every part and mesh
again, so an invalid direct edit to a mutable `MeshGeometry` is caught before
export. It also checks the full articulation tree.

```python
from articraft.sdk import RigidBodyAssembly, ValidationError


model = RigidBodyAssembly("example")
body = model.rigid_body("body")

try:
    model.validate()
except ValidationError as exc:
    print(exc)
```

This example fails because `body` has no named shape yet.

## Built in Python errors

Not every public helper failure is an `SDKError`.

- Python raises `TypeError` when a required argument is missing or a value has
  the wrong Python type for an operation.
- Mesh builders and transforms often raise `ValueError` for an impossible
  dimension, a zero rotation axis, or an invalid profile.
- Manifold boolean helpers raise `ValueError` unless each input is a nonempty
  closed manifold solid and the result has healthy topology. Empty results,
  bad triangles, and unintended separate solid fragments are errors.
- Field based welds and smooth cuts raise `ValueError` when their result has
  unhealthy mesh geometry. The message includes issue counts and affected
  bounds.
- Allowance helpers raise `ValueError` when their required reason is empty.

Catch the narrow error that the operation documents. Do not catch an error only
to continue with geometry that did not build correctly.

## Lookup errors

Lookup helpers raise `ValidationError` when a name cannot be resolved.

```python
part = model.get_rigid_body("body")
shape = part.get_shape("housing")
joint = model.get_joint("body_to_lid")
```

Part names are model scoped. Shape names are part scoped. The same shape name
may be used on two different parts, so a shape lookup always starts from a
specific part.

## Imports

Import public SDK types from `articraft.sdk`.

```python
from articraft.sdk import JointDOF, JointFrame, RigidBodyAssembly
```

Paths such as `docs/sdk/common/20_core_types.md` are documentation paths. They
are not Python modules.

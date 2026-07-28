# Articulated objects and parts

`ArticulatedObject` is the root model. It owns rigid parts and the
articulations that connect them.

```python
from mini_articraft.sdk import ArticulatedObject
```

## Construction

```python
ArticulatedObject(name: str)
```

The name must be a nonempty string. Leading and trailing whitespace is removed.
There is no `units` argument. The model always uses meters, and
`model.meters_per_unit` is always `1.0`.

`model.parts` and `model.articulations` are public lists for inspection. Use
the authoring helpers below instead of appending to them directly.

## The authoring order

Build a model in this order:

1. Create the `ArticulatedObject`.
2. Create every rigid `Part` with `model.part(...)`.
3. Add one or more named shapes to each part.
4. Add articulations between parts.
5. Add design checks and call `model.validate()` before export.

Parent and child parts must already exist when an articulation is added.

## `model.part(...)`

```python
model.part(name: str) -> Part
```

This method creates a `Part`, appends it to `model.parts`, and returns it. Part
names must be unique within the model.

```python
body = model.part("body")
head = model.part("head")
```

An empty part is allowed while the model is being built. Full model validation
requires every part to contain at least one named shape.

## `Part`

```python
Part(name: str)
```

Each part is one rigid body. Put geometry that must move together in the same
part. A decorative trim piece should stay in the same part as its housing when
they never move relative to each other.

Create a separate part only when the geometry needs a separate rigid motion.

### `part.add(...)`

```python
part.add(
    shape: build123d.Shape | MeshGeometry,
    *,
    name: str,
    material: Material | None = None,
    coating: Material | None = None,
    color: Sequence[float] | None = None,
    appearance: Appearance | None = None,
) -> build123d.Shape | MeshGeometry
```

`name` is required and must be unique within this part. The same shape name may
be used on a different part. The method returns the exact geometry object that
was passed in.

The optional color has three RGB values or four RGBA values. Values range from
zero through one. RGB gets an alpha value of one.

```python
from build123d import Box, Cylinder, Pos


body = model.part("body")
shell = Box(0.30, 0.22, 0.28)
trim = Pos(Z=0.15) * Cylinder(0.09, 0.02)

body.add(shell, name="shell", color=(0.70, 0.10, 0.10))
body.add(trim, name="top_trim", color=(0.80, 0.80, 0.82, 0.70))
```

This creates one rigid part with two named shapes and two colors. It does not
create two rigid bodies.

### Materials

`material` says what the shape is made of. One word settles three things: how
heavy the shape is, how it behaves on contact, and how it looks. It is usually
the only one you need.

```python
from mini_articraft.sdk import Material

body.add(shell, name="shell", material=Material.ALUMINUM)
body.add(trim, name="lens", material=Material.GLASS)
```

| Material | Density kg/m^3 | Looks like |
| --- | --- | --- |
| `Material.STEEL` | 7850 | light grey metal |
| `Material.ALUMINUM` | 2700 | pale metal |
| `Material.ABS_PLASTIC` | 1050 | off-white dielectric |
| `Material.GLASS` | 2500 | pale, translucent |
| `Material.HARDWOOD` | 700 | mid brown, diffuse |
| `Material.RUBBER` | 1200 | near-black, matte |

Different shapes on one part may be different materials. A steel shell with a
hardwood handle weighs both, shape by shape.

### Coating a shape in something else

Density belongs to the bulk; friction and looks belong to the surface. `coating`
covers a shape in a different material, so the two can differ:

```python
# heavy like steel, grippy like rubber
bar.add(shape, name="bar", material=Material.STEEL, coating=Material.RUBBER)

# chrome-plated plastic: light, but slides and looks like metal
knob.add(shell, name="knob", material=Material.ABS_PLASTIC, coating=Material.STEEL)
```

| | Comes from |
| --- | --- |
| mass | `material` -- the bulk |
| friction, restitution | `coating` if given, else `material` |
| appearance, textures | `coating` if given, else `material` |

### Changing how something looks

`color` recolors without changing any physics. The surface keeps responding to
light the same way, so red rubber stays matte and a blue steel panel stays
metallic.

```python
foot.add(pad, name="foot", material=Material.RUBBER, color=(0.8, 0.1, 0.1))
```

An `Appearance` replaces the look entirely and claims nothing physical -- paint,
a screen, an indicator lamp. Use `coating` instead when the surface is really a
different material, or the friction will be wrong.

```python
from mini_articraft.sdk import Appearance

panel.add(face, name="face", material=Material.STEEL,
          appearance=Appearance(base_color=(0.1, 0.1, 0.12, 1.0), emissive=(0.0, 0.6, 0.4)))
```

```python
Appearance(
    base_color: (r, g, b, a) in [0, 1],   # alpha is opacity
    metallic: float = 0.0,                 # 0 dielectric, 1 raw metal
    roughness: float = 0.6,                # 0 mirror, 1 fully diffuse
    emissive: (r, g, b) | None = None,     # optional unlit glow
)
```

Presets: `Appearance.metal(color, roughness=...)`, `Appearance.plastic(color)`,
`Appearance.rubber(color)`, `Appearance.matte(color)`, `Appearance.glass(color)`.
`color` and `appearance` cannot both be set -- an appearance already carries its
color. Exporters never infer a material from a shape name.

### Build123d placement

A build123d shape keeps its current build123d location. Apply `Pos`, `Rot`, or
another build123d `Location` before adding it.

```python
from build123d import Box, Pos, Rot


handle = Pos(X=0.12, Z=0.04) * Rot(Y=20.0) * Box(0.08, 0.02, 0.02)
body.add(handle, name="handle")
```

The `Pos` values are treated as meters by Mini Articraft. Build123d `Rot` uses
degrees. There is no second per shape transform in `Part`.

### Mesh geometry

`MeshGeometry` uses the same part local frame.

```python
from mini_articraft.sdk import BoxGeometry


badge = BoxGeometry((0.05, 0.002, 0.02)).translate(0.0, -0.111, 0.03)
body.add(badge, name="badge", color=(0.85, 0.65, 0.12))
```

The part stores the mesh object itself. If you edit that mesh later, the part
sees the edit. `model.validate()` validates the edited vertices and faces
again.

### `part.get_shape(...)`

```python
part.get_shape(name: str) -> build123d.Shape | MeshGeometry
```

This method returns the named geometry object. It raises `ValidationError` when
the name is empty or unknown.

```python
housing = model.get_part("body").get_shape("shell")
```

Use the part and shape name together when a test or inspection command must
target one feature.

## Local and world coordinates

Every shape on a part uses that part's local coordinates.

The root part frame is the world frame at rest. A child part frame comes from
its parent articulation. At a zero motion value, the child frame is the
articulation `Origin` relative to the parent.

Do not place child geometry twice. Author the child around its own local frame,
then use the articulation origin to place that frame on the parent.

The transform details are in [articulations](35_joints.md).

## `model.get_part(...)`

```python
model.get_part(part: str | Part) -> Part
```

Pass a part name or a `Part`. The method resolves the name against this model
and returns the stored part. It raises `ValidationError` for an unknown part.

## `model.get_articulation(...)`

```python
model.get_articulation(name: str | Articulation) -> Articulation
```

Pass an articulation name or an `Articulation`. The method returns the stored
articulation with that name. It raises `ValidationError` when no match exists.

## `model.validate()`

```python
model.validate() -> None
```

Validation does not return a repaired model. It returns `None` on success and
raises `ValidationError` on failure.

Validation checks all of these rules:

- The model has at least one part.
- Every entry in `model.parts` is a `Part`.
- Every part has a nonempty unique name.
- Every part has at least one named nonempty shape.
- Every build123d shape is nonempty and valid.
- Every `MeshGeometry` has valid finite vertices and triangle indices.
- Every shape color has three or four values in the allowed range, and every
  shape material is a valid `Material`.
- Part names are unique.
- Every entry in `model.articulations` is an `Articulation`.
- Articulation names are unique.
- Every articulation satisfies its type and limit rules.
- Every parent and child name refers to a part in this model.
- A part has at most one parent articulation.
- The model has exactly one root part.
- Every part is reachable from that root, so cycles and detached branches are
  invalid.

A model with one part and no articulation is valid. That part is the root.

Validation checks the authored connection tree. Compile time geometry checks
also look for physically isolated parts and unintended overlaps. See [testing
geometry and assemblies](40_testing.md) for those checks.

## Naming rules

Object, part, shape, and articulation names must be strings. The SDK removes
leading and trailing whitespace and rejects an empty result.

Uniqueness has these scopes:

- Part names are unique within one model.
- Articulation names are unique within one model.
- Shape names are unique within one part.

Use short stable names because test failures, compile signals, JSON records,
and USD metadata report them.

## Export layout

Each part exports as one USD rigid body. Each named shape exports as a child
mesh with its own name and color.

```text
/World/<object>/parts/<part>/shapes/<shape>
```

An articulation targets the parent and child part bodies, not their individual
shape meshes. The USD stage uses meters and Z up.

## Complete mixed geometry example

```python
from build123d import Box

from mini_articraft.sdk import ArticulatedObject, BoxGeometry


model = ArticulatedObject("mixed_body")
body = model.part("body")

body.add(Box(0.30, 0.20, 0.08), name="housing", color=(0.25, 0.30, 0.36))

feet = BoxGeometry((0.24, 0.14, 0.02)).translate(0.0, 0.0, -0.05)
body.add(feet, name="feet", color=(0.05, 0.05, 0.06))

model.validate()
object_model = model
```

See `docs/sdk/examples/mixed_articulated_assembly.py` for a model with a moving
mesh part.

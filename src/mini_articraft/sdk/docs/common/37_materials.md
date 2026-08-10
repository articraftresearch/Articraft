# Materials and mass

`Material` says what a shape is made of. Everything physical follows from it:
mass, contact behavior, and appearance.

**When the physics lane is on, every part must be weighable or the compile
fails.** A part is weighable when its shapes say what they are made of.

Units are SI: mass in kilograms, density in kg/m^3, lengths in meters, inertia in kg*m^2.

## Naming a material

```python
from mini_articraft.sdk import RigidBodyAssembly, Material

model = RigidBodyAssembly("kettle")
base = model.rigid_body("base")
base.add(shell, name="shell", material=Material.STEEL)
```

| Material | Density | Static / dynamic friction | Restitution |
| --- | --- | --- | --- |
| `Material.STEEL` | 7850 | 0.42 / 0.36 | 0.55 |
| `Material.ALUMINUM` | 2700 | 0.45 / 0.38 | 0.40 |
| `Material.ABS_PLASTIC` | 1050 | 0.40 / 0.32 | 0.45 |
| `Material.GLASS` | 2500 | 0.40 / 0.35 | 0.60 |
| `Material.HARDWOOD` | 700 | 0.50 / 0.40 | 0.35 |
| `Material.RUBBER` | 1200 | 0.95 / 0.85 | 0.75 |

## Deriving and inventing

`Material.but(...)` derives a variant, keeping everything you do not change.
Name it to reuse it across shapes and parts; it keeps its origin's texture, so
brushed steel still looks like steel.

```python
BRUSHED = Material.STEEL.but(roughness=0.75)
GRIPPY = Material.RUBBER.but(name="tacky", friction=(1.1, 0.95))
```

Build one only when the library has nothing close. `density` is required;
`friction` is optional, and omitting it authors no contact behavior rather than
inventing a coefficient.

```python
CERAMIC = Material(name="ceramic", density=2400.0, friction=(0.45, 0.40))
```

Prefer the library: its numbers were checked, and the manifest records which
materials came from it.

## One part, several materials

Material lives on the shape, so a part can be made of more than one thing. Each
shape is weighed by its own material and the part's mass is the total:

```python
box = model.rigid_body("toolbox")
box.add(shell, name="shell", material=Material.STEEL)      # 1.4 kg of steel
box.add(grip, name="grip", material=Material.HARDWOOD)     # 0.2 kg of wood
box.add(pad, name="foot", material=Material.RUBBER)        # grips the table
```

The center of mass and inertia are composed across the shapes, so a heavy steel
base with a light plastic top sits low, exactly as the real object would.

## What gets measured

Volume, center of mass, and the full inertia tensor come from the part's meshes. Shapes
of the same material are unioned before they are weighed, so geometry that deliberately
overlaps (an embedded handle end, for example) does not contribute its shared volume
twice. Hollow geometry behaves correctly on its own: a shell built with
`boolean_difference` measures the volume of the wall, not of a solid block.

Every shape in the part must be a closed solid. A shape that is not closed cannot be
measured, so the compile fails naming the part rather than quietly leaving that shape's
weight out of the total. Inverted winding is repaired automatically.

If you override `center_of_mass` but let the inertia be measured, the measured tensor is
shifted to your center with the parallel-axis theorem, so the exported pair stays
physically consistent. Overriding `diagonal_inertia` takes your tensor as given.

## Overrides

`MassProperties` on the part exists for what measurement cannot reach. Every
field is optional and anything left out is measured.

```python
# a substance the library does not cover
model.rigid_body("stone", mass_properties=MassProperties(density=2600.0))

# geometry that stands in for something whose real weight you know
model.rigid_body("motor", mass_properties=MassProperties(mass=0.85, center_of_mass=(0.0, 0.0, 0.04)))
```

An explicit `mass` or `density` applies to the whole part and ignores the shape
materials for weighing -- they still decide contact behavior and appearance.

## Physics mode

When the physics lane is enabled (`mini-articraft generate --physics ...`), **every part
must be weighable** and the compile fails otherwise, naming the parts it could not weigh.
This is deliberate: a silently assumed density is an invisible wrong answer, so the
compiler asks you to state what each shape is made of instead of guessing.

With physics off, parts that cannot be weighed simply export without mass.

## Modelling hollow things

Mass is measured from the geometry you built, so a part modelled as a solid block is
weighed as a solid block. Most real objects are not solid, and dense materials punish
the mistake hard: a 0.35 x 0.30 m steel panel weighs **0.8 kg at 1 mm** and **8 kg at
1 cm**. Model the wall, not the envelope:

```python
shell = boolean_difference(outer, inner)  # measured volume is the wall
```

Real thicknesses, when you have no better reference:

| Thing | Typical wall |
| --- | --- |
| Sheet-metal panel, appliance body, toolbox lid | 0.5-1.5 mm |
| Cast metal housing, cookware | 2-4 mm |
| Injection-moulded plastic shell | 1.5-3 mm |
| Wooden board, plank, panel | 10-20 mm |
| Glass bottle, jar, window | 2-5 mm |

Sanity-check the result: a cordless drill is about 1.5 kg, a kettle 1 kg, a laptop 2 kg,
a cast-iron pan 2 kg, a toolbox 5 kg. If a hand-held object comes out at 20 kg, the
geometry is solid where it should be a shell. Genuinely solid parts (a steel bolt, a
hardwood block) are fine as-is -- this is about parts that only look solid because the
cavity was never cut.

## Export

Weighable parts export standard `UsdPhysics.MassAPI` attributes on the part prim:
`physics:mass`, `physics:centerOfMass`, `physics:diagonalInertia`, and
`physics:principalAxes`.

Each shape's surface material -- its `coating` if it has one, otherwise its own material --
binds a `UsdPhysics.MaterialAPI` to its collider when it declares friction, carrying
`physics:staticFriction`, `physics:dynamicFriction`, and `physics:restitution`. Friction
is a surface property, so it binds per shape: a steel frame on rubber feet grips through
the feet. See `docs/sdk/common/50_usdz_export.md` for the rest of the USD layout.

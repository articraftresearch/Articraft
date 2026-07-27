# Mass properties

`MassProperties` declares what a part is made of; `MaterialDensity` names a
material from the built-in density library.

**When the physics lane is on, every part must declare mass properties or the
compile fails.** Pass `mass_properties=MassProperties(material=...)` to every `model.part()`
call as you create it.

Each part is one rigid body, so it carries the physical values a simulator needs: how
heavy it is, where that weight sits, and how it resists rotation. You say what the part
is **made of**; the SDK measures the rest from the geometry the part already contains.

Units are SI: mass in kilograms, density in kg/m^3, lengths in meters, inertia in kg*m^2.

## Naming a material

```python
from mini_articraft.sdk import ArticulatedObject, MassProperties, MaterialDensity

model = ArticulatedObject("kettle")
base = model.part("base", mass_properties=MassProperties(material=MaterialDensity.STEEL))
```

Built-in materials and their densities in kg/m^3:

| Material | Density |
| --- | --- |
| `MaterialDensity.STEEL` | 7850 |
| `MaterialDensity.ALUMINUM` | 2700 |
| `MaterialDensity.ABS_PLASTIC` | 1050 |
| `MaterialDensity.GLASS` | 2500 |
| `MaterialDensity.HARDWOOD` | 700 |
| `MaterialDensity.RUBBER` | 1200 |

## Precedence

Give exactly **one** of `material`, `density`, or `mass`. Supplying more than one, or
none at all, is a validation error.

1. `mass=` sets the mass directly in kilograms and ignores volume. Use it when you know
   what the real object weighs, or when the geometry is a stand-in for something whose
   volume would mislead.
2. `density=` multiplies your value by the part's measured volume. Use it for a material
   that is not in the library.
3. `material=` looks up the density above and does the same.

`center_of_mass`, `diagonal_inertia`, and `principal_axes` are always measured from the
geometry unless you set them explicitly, in which case your values are used verbatim.

```python
# measured from geometry, plastic density
model.part("lid", mass_properties=MassProperties(material=MaterialDensity.ABS_PLASTIC))

# a density the library does not cover
model.part("stone", mass_properties=MassProperties(density=2600.0))

# a known weight, with the mass concentrated where you say
model.part("motor", mass_properties=MassProperties(mass=0.85, center_of_mass=(0.0, 0.0, 0.04)))
```

## What gets measured

Volume, center of mass, and the full inertia tensor come from the part's meshes. A part
built from several shapes is unioned first, so shapes that overlap (an embedded handle
end, for example) do not contribute their shared volume twice. Hollow geometry behaves
correctly on its own: a shell built with `boolean_difference` measures the volume of the
wall, not of a solid block.

Every shape in the part must be a closed solid. A shape that is not closed cannot be
measured, so the compile fails naming the part rather than quietly leaving that shape's
weight out of the total. Inverted winding is repaired automatically.

If you override `center_of_mass` but let the inertia be measured, the measured tensor is
shifted to your center with the parallel-axis theorem, so the exported pair stays
physically consistent. Overriding `diagonal_inertia` takes your tensor as given.

## Physics mode

When the physics lane is enabled (`mini-articraft generate --physics ...`), **every part must declare
mass properties** and the compile fails otherwise, listing the parts that are missing
them. This is deliberate: a silently assumed density is an invisible wrong answer, so the
compiler asks you to state what each part is made of instead of guessing.

With physics off, parts without mass properties simply export without them.

## Solid or hollow changes the answer by 10x

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

Parts with mass properties export standard `UsdPhysics.MassAPI` attributes on the part
prim: `physics:mass`, `physics:density`, `physics:centerOfMass`,
`physics:diagonalInertia`, and `physics:principalAxes`. See
`docs/sdk/common/50_usdz_export.md` for the rest of the USD layout.

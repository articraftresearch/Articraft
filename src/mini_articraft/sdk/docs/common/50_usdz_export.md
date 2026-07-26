# USDZ export

The compile worker exports one USDZ package after it runs the authored tests and compiler owned
checks. It saves the package when the object can be exported, even if a check fails. In an agent
run, use the `compile` tool. Do not call `export_object` from generated `main.py`, because that
would bypass the compiler owned checks and publication rules.

Code using mini-articraft as a regular Python SDK can export directly:

```python
from mini_articraft.sdk.export import export_object

result = export_object(object_model, "output")
print(result.usdz)
```

The explicit `mini_articraft.sdk.export` import keeps the OpenUSD dependency out of a plain
`mini_articraft.sdk` import until export is requested.

The exporter validates the object and tessellates build123d shapes with a default tolerance of
`0.001` meter.

## Output files

A successful export writes:

```text
result/
  model.json
  usdz/
    0000.usdz
```

Later successful exports keep the earlier packages and use `0001.usdz`, `0002.usdz`, and so on.
The exporter finds the largest numeric USDZ stem and adds one. It does not fill gaps. Files with a
nonnumeric stem do not affect the number.

`model.json` is the manifest for the latest successful export. Each success replaces this file and
updates its `files.usdz` value to the new numbered package.

## USD hierarchy

The stage default prim is `/World`. The exporter writes this hierarchy when all names are already
valid USD identifiers:

```text
/World
  /<object>
    /parts
      /<part>
        /shapes
          /<shape>
    /joints
      /<articulation>
```

The exact path for a named shape is:

```text
/World/<object>/parts/<part>/shapes/<shape>
```

`/World/<object>` is an `Xform` with `UsdPhysics.ArticulationRootAPI`. The `parts` and `shapes`
containers are `Scope` prims. Each part is an `Xform` with `UsdPhysics.RigidBodyAPI`. Each named
shape is one child `UsdGeom.Mesh`.

Part prims are siblings under `parts`. Each part `Xform` has its world transform at the rest pose.
The mesh points remain in part local coordinates. A build123d `Pos`, `Rot`, or `Location` is
already part of the tessellated shape and does not create another USD transform.

The exporter converts invalid USD identifier characters. If two original names map to the same
identifier, later names receive suffixes such as `_2`. The manifest and custom USD attributes keep
the original names.

## Geometry and materials

Both build123d shapes and `MeshGeometry` values are exported as triangle meshes. Each mesh has:

- The mesh has triangle face counts and indices.
- The mesh has authored point data and an extent.
- The mesh uses `none` as its subdivision setting.
- The mesh has the custom `mini_articraft:name` attribute.

The exporter uses `mesh_tolerance` when it tessellates a build123d shape. A `MeshGeometry` uses its
current vertices and triangle faces.

When a named shape has a material, the mesh gets a bound `UsdPreviewSurface` plus
`primvars:displayColor` and `primvars:displayOpacity` fallbacks. An RGB color is
shorthand for a matte material with opacity `1.0`; RGBA keeps its authored alpha.
A shape without a material has no display color or material authored by mini-articraft.

The exporter does not create inertial values or separate collision meshes.

## Units and stage metadata

The USD stage uses:

```text
metersPerUnit = 1.0
upAxis = "Z"
```

All mesh points, part translations, articulation origins, and prismatic limits therefore use
meters.

The object prim also has these custom attributes:

```text
mini_articraft:name
mini_articraft:units = "meters"
```

## Articulations and joints

Joint prims are children of `/World/<object>/joints`. Their body relationships target the parent
and child part `Xform` prims, not the child mesh prims.

The SDK articulation types map to USD schemas as follows:

| SDK type | USD schema | Standard limits |
| --- | --- | --- |
| `FIXED` | `UsdPhysics.FixedJoint` | none |
| `REVOLUTE` | `UsdPhysics.RevoluteJoint` | lower and upper in degrees |
| `CONTINUOUS` | `UsdPhysics.RevoluteJoint` | none |
| `PRISMATIC` | `UsdPhysics.PrismaticJoint` | lower and upper in meters |

The SDK stores revolute limits in radians. The exporter converts them to degrees for the standard
USD revolute limit attributes. The custom `mini_articraft:limits:lower` and
`mini_articraft:limits:upper` attributes keep the original SDK values.

USD joints use their local X axis. The exporter rotates that X axis to the normalized SDK
articulation axis. The parent local frame includes `Origin.xyz` and `Origin.rpy`. The child local
position is zero.

Every joint also records custom mini-articraft attributes for its original name, type, parent,
child, axis, and origin. When motion limits exist, it records effort and velocity. These values are
metadata. The exporter does not create a USD drive.

## Manifest

The manifest is plain JSON with this shape:

```json
{
  "name": "stand_mixer",
  "units": "meters",
  "meters_per_unit": 1.0,
  "up_axis": "Z",
  "parts": [
    {
      "name": "body",
      "shapes": [
        {
          "name": "shell",
          "geometry_type": "Box",
          "color": [0.7, 0.1, 0.1, 1.0]
        }
      ]
    }
  ],
  "articulations": [],
  "files": {
    "usdz": "usdz/0000.usdz"
  }
}
```

Each articulation entry includes its name, type, parent, child, origin, axis, and motion limits.
`geometry_type` is the Python class name of the authored build123d shape or mesh value.

## Validation and safe publication

Before packaging, the exporter runs these OpenUSD stage checks:

- OpenUSD checks the stage metadata.
- OpenUSD checks for composition errors.
- OpenUSD checks the rigid bodies.
- OpenUSD checks the physics joints.
- OpenUSD checks the articulation structure.

After packaging, it opens the USDZ and runs the OpenUSD package validator. The exporter publishes
the final numbered path only after these checks pass.

The package and manifest use temporary files while they are being written. If export or validation
fails, the exporter removes the new USDZ and does not replace the manifest. Earlier successful
numbered packages remain in place.

## Compile failure behavior

The compile worker runs the authored and baseline checks before it calls the exporter. A blocking
check failure still creates a new numbered USDZ and updates the manifest when the object can be
exported. This keeps intermediate models available for inspection. The compile result still has
status `error`, so the agent must fix the checks before it can finish the run.

A fatal error before export does not create a USDZ or consume a USDZ number. This includes errors
while loading `main.py`, building the object, or running the required test function. An invalid
model or an OpenUSD export error also prevents the exporter from publishing a package.

Compile attempts update only the attempt count in `record.json`. The agent writes run status
`success` and the relative USDZ result only after the current workspace has a fresh successful
compile and the model has returned a visible final response. A failed compile result may contain
an intermediate USDZ path, but it does not publish that path in the run record.

If an earlier compile in the same run succeeded, a later failed compile does not delete the older
package. The freshness check prevents the agent from publishing that older package for changed
workspace inputs.

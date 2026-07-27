# Visual evidence and artifacts

Use visual evidence after the model builds. The renderer works without a
window or graphics processor. It writes PNG files that the agent can inspect
with `view_image`.

Import these public names from `mini_articraft.sdk`.

```python
from mini_articraft.sdk import (
    ImagePoint,
    LineOverlay,
    MeridionalSectionView,
    ModelView,
    MotionStripView,
    PointOverlay,
    ProjectedPoint,
    Reticle,
    SectionView,
    TestArtifact,
    annotate_image,
    probe_view,
    project_model_points,
    render_view,
)
```

## Reference reticles

`ImagePoint` stores one normalized coordinate and `Reticle` gives it a label and
display color. `annotate_image` draws those reticles on a reference image.

Image-driven runs place the prepared input at `reference.png`, `reference.jpg`, or
`reference.webp` in the workspace. Use normalized coordinates (`u` right, `v`
down) to mark evidence without depending on the image resolution.

```python
annotate_image(
    "reference.png",
    (
        Reticle(ImagePoint(0.18, 0.52), "handle root"),
        Reticle(ImagePoint(0.82, 0.34), "tip"),
    ),
    "qa/previews/reference-markers.png",
)
```

Open the annotated output with `view_image` to confirm each reticle is on the
intended feature. Reticles record visible evidence; they do not infer depth from
a single image.

## Model projection and probing

`project_model_points` maps known world points into the exact normalized
coordinates used by a `ModelView` and returns `ProjectedPoint` values.
`probe_view` performs the inverse visual query: it raycasts screen-space
reticles and returns a `ViewProbe` containing the nearest `SurfaceHit` with its
part, shape, world position, and surface normal.

```python
view = ModelView.three_quarter()
tip = project_model_points(object_model, view, ((0.22, 0.0, 0.14),))[0]
probes = probe_view(
    object_model,
    view,
    (Reticle(ImagePoint(0.72, 0.41), "reference feature"),),
)
print(tip.u, tip.v, probes[0].hit)
```

Use the same `ModelView`, selections, and pose for rendering and queries. A
probe returns `None` when the reticle misses the visible model. Projection does
not claim that an off-screen or occluded point is visible;
`ProjectedPoint.in_frame` only reports whether it lands inside the image bounds.

## Preview before compile

Use a local `previews.py` script to inspect the first complete model before the
final compile. The script can import `object_model` from `main.py` and call the
public renderer directly.

```python
from main import object_model
from mini_articraft.sdk import ModelView, SectionView, render_view


def main() -> None:
    render_view(
        object_model,
        ModelView.three_quarter(
            selected_parts=("housing", "service_door"),
            show_joints=True,
        ),
        "qa/previews/assembly.png",
    )
    render_view(
        object_model,
        SectionView(
            plane_origin=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 1.0, 0.0),
        ),
        "qa/previews/section.png",
    )


if __name__ == "__main__":
    main()
```

Run the script through `exec_command`.

```sh
"$MINI_ARTICRAFT_PYTHON" previews.py
```

`MINI_ARTICRAFT_PYTHON` points to the same Python interpreter that runs
mini-articraft, so the public SDK is available without a private import or an
SDK edit. Open each useful output with `view_image`. Running the script does not
count as visual inspection.

Working previews can stay under `qa/previews/` without being registered. Use
`attach_artifact(...)` in `run_tests()` when a preview is useful as final
evidence.

```python
ctx.attach_artifact(
    "qa/previews/section.png",
    name="housing section",
    caption="The section shows the shell, bore, and internal support.",
)
ctx.attach_artifact(
    "qa/clearance_samples.json",
    name="clearance samples",
    caption="Clearance at each tested joint pose.",
)
```

The path must be relative to the run workspace. Supported files are PNG, JPEG,
WebP, JSON, CSV, Markdown, and plain text.

## Compile behavior

The compiler does not render or copy images. Registered files stay in the
workspace. The `compile` tool returns text signals with safe workspace paths for
registered images and does not attach image content. Open each useful path with
`view_image`.

## General model views

`ModelView` renders visible triangles with a depth buffer.

```python
ModelView(
    direction=(1.0, -1.0, 0.7),
    up=(0.0, 0.0, 1.0),
    projection="orthographic",
    width=640,
    height=480,
    color_by="part",
    show_bounds=False,
    show_joints=False,
    selected_parts=(),
    selected_shapes=(),
    points=(),
    lines=(),
)
```

Use `ModelView.front()`, `side()`, `top()`, or `three_quarter()` for a common
camera. `projection` can be `orthographic` or `perspective`. `color_by` can be
`part`, `shape`, or `material`.

Use `selected_parts` or `selected_shapes` to isolate important geometry.
`PointOverlay` marks a world position. `LineOverlay` draws a world segment.
These overlays can show tracked points, a measured clearance, a contact
position, a collision normal, or a custom axis.

```python
tip_path = ctx.track_point("arm", (0.0, 0.0, 0.4), poses)
view = ModelView.side(
    points=tuple(PointOverlay(point, label=f"sample {index}") for index, point in enumerate(tip_path)),
    lines=tuple(
        LineOverlay(tip_path[index], tip_path[index + 1])
        for index in range(len(tip_path) - 1)
    ),
)
```

## Arbitrary sections

`SectionView` intersects every selected triangle with one plane.

```python
SectionView(
    plane_origin=(0.0, 0.0, 0.0),
    plane_normal=(0.0, 1.0, 0.0),
    horizontal=(1.0, 0.0, 0.0),
    vertical=(0.0, 0.0, 1.0),
)
```

The horizontal and vertical vectors must lie in the section plane. Leave them
out when their exact direction is not important.

`MeridionalSectionView` is a focused section for a form built around an axis.
It is still based on a general plane intersection.

```python
MeridionalSectionView(
    axis=(1.0, 0.0, 0.0),
    radial_direction=(0.0, 0.0, 1.0),
    origin=(0.0, 0.0, 0.0),
)
```

The axis and radial direction must be perpendicular.

## Motion strips

`MotionStripView` places several poses in one image.

```python
MotionStripView(
    articulation="lid_hinge",
    positions=(0.0, 0.6, 1.2),
    view=ModelView.side(width=320, height=280, show_joints=True),
)
```

Leave out `positions` to sample the authored motion limits. Continuous joints
use a half turn.

## Direct rendering

Local preview code can call `render_view(...)` without creating a test report.
`render_view` is the public rendering function.

```python
render_view(
    object_model,
    ModelView.three_quarter(),
    "qa/three_quarter.png",
    pose={"lid_hinge": 0.8},
)
```

`render_view(...)` accepts a `ModelView`, `SectionView`,
`MeridionalSectionView`, or `MotionStripView`. The output must be a PNG.

# Visual evidence and artifacts

Render images after the model can run. The renderer does not need a window or graphics
processor.

It writes PNG files that you can inspect with `view_image`.

Import these public names from `articraft.sdk`:

```python
from articraft.sdk import (
    ImagePoint,
    LineOverlay,
    MeridionalSectionView,
    ModelView,
    MotionStripView,
    PointOverlay,
    Reticle,
    SectionView,
    TestArtifact,
    annotate_image,
    render_view,
)
```

## Mark a reference image

`ImagePoint` stores one normalized image coordinate. `Reticle` gives the point a label and
display color.

Use `annotate_image` to draw reticles on a reference image. The workspace stores the
prepared input as `reference.png`, `reference.jpg`, or `reference.webp`.

The `u` coordinate increases to the right. The `v` coordinate increases down the image.

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

Open the output with `view_image`. Confirm that each reticle marks the intended feature.

A reticle records visible evidence. It cannot find depth from one image.

## Preview before compilation

Create `previews.py` after you have the first complete model. Import `object_model` from
`main.py` and call the public renderer.

```python
from main import object_model
from articraft.sdk import ModelView, SectionView, render_view


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

Run the script with the Articraft Python interpreter:

```sh
"$ARTICRAFT_PYTHON" previews.py
```

`ARTICRAFT_PYTHON` points to the interpreter that runs Articraft. You do not need private
imports or SDK changes.

Open each useful output with `view_image`. Calling the renderer is not visual inspection.

Working previews can stay under `qa/previews/`. Attach a preview when it is useful final
evidence:

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

The path must be relative to the run workspace. `TestArtifact` supports these file types:

- PNG, JPEG, and WebP images.
- JSON and CSV data.
- Markdown and plain text.

## Understand compilation behavior

The compiler does not render or copy images. Registered files stay in the workspace.

The `compile` tool returns safe workspace paths for registered images. It does not attach image
content.

Open each useful path with `view_image`.

## Render a model view

`ModelView` renders visible triangles with a depth buffer:

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

Use `ModelView.front()`, `side()`, `top()`, or `three_quarter()` for a common camera.

`projection` accepts `orthographic` or `perspective`. `color_by` accepts `part`, `shape`, or
`material`.

Use `selected_parts` or `selected_shapes` to isolate geometry. Use `PointOverlay` for a world
position and `LineOverlay` for a world segment.

```python
tip_path = ctx.track_point("arm", (0.0, 0.0, 0.4), poses)
view = ModelView.side(
    points=tuple(
        PointOverlay(point, label=f"sample {index}")
        for index, point in enumerate(tip_path)
    ),
    lines=tuple(
        LineOverlay(tip_path[index], tip_path[index + 1])
        for index in range(len(tip_path) - 1)
    ),
)
```

## Render a section

`SectionView` intersects selected triangles with one plane:

```python
SectionView(
    plane_origin=(0.0, 0.0, 0.0),
    plane_normal=(0.0, 1.0, 0.0),
    horizontal=(1.0, 0.0, 0.0),
    vertical=(0.0, 0.0, 1.0),
)
```

The horizontal and vertical vectors must be in the section plane. Omit them when exact
directions are not necessary.

Use `MeridionalSectionView` for a form around one axis:

```python
MeridionalSectionView(
    axis=(1.0, 0.0, 0.0),
    radial_direction=(0.0, 0.0, 1.0),
    origin=(0.0, 0.0, 0.0),
)
```

The axis and radial direction must be perpendicular.

## Render a motion strip

`MotionStripView` puts several poses in one image:

```python
MotionStripView(
    articulation="lid_hinge",
    positions=(0.0, 0.6, 1.2),
    view=ModelView.side(width=320, height=280, show_joints=True),
)
```

Set `dof` when the joint has several degrees of freedom. It accepts an axis name or the
qualified DOF identifier:

```python
MotionStripView(
    articulation="removable_lid",
    dof="removable_lid.transZ",
    positions=(0.0, 0.04, 0.10),
    view=ModelView.side(width=320, height=280),
)
```

Omit `positions` to sample the authored motion limits. A continuous joint uses a half turn.

## Render directly

Call `render_view` from local preview code. You do not need a test report.

```python
render_view(
    object_model,
    ModelView.three_quarter(),
    "qa/three_quarter.png",
    pose={"lid_hinge": 0.8},
)
```

`render_view(...)` accepts `ModelView`, `SectionView`, `MeridionalSectionView`, or
`MotionStripView`. The output file must be PNG.

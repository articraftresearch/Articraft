from __future__ import annotations

import trimesh

from mini_articraft.sdk import (
    BoxGeometry,
    MeshGeometry,
    MeshHealthIssue,
    analyze_mesh_health,
)
from mini_articraft.sdk.mesh import boolean_difference


def test_analyze_mesh_health_accepts_a_clean_solid() -> None:
    report = analyze_mesh_health(BoxGeometry((1.0, 1.0, 1.0)))

    assert report.healthy
    assert report.watertight
    assert report.winding_consistent
    assert report.component_count == 1
    assert report.signed_volume > 0.0


def test_analyze_mesh_health_finds_an_open_boundary() -> None:
    box = BoxGeometry((1.0, 1.0, 1.0)).to_trimesh()
    open_box = MeshGeometry.from_trimesh(
        trimesh.Trimesh(
            vertices=box.vertices,
            faces=box.faces[:-1],
            process=False,
        )
    )

    report = analyze_mesh_health(open_box)

    assert not report.healthy
    assert not report.watertight
    assert MeshHealthIssue.BOUNDARY_EDGES in report.issues


def test_analyze_mesh_health_finds_bad_faces_and_unused_vertices() -> None:
    geometry = MeshGeometry(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 2.0),
        ],
        faces=[
            (0, 1, 2),
            (3, 1, 2),
            (0, 3, 1),
        ],
    )

    report = analyze_mesh_health(geometry)

    assert MeshHealthIssue.DUPLICATE_VERTICES in report.issues
    assert MeshHealthIssue.DUPLICATE_FACES in report.issues
    assert MeshHealthIssue.DEGENERATE_FACES in report.issues
    assert MeshHealthIssue.UNUSED_VERTICES in report.issues


def test_analyze_mesh_health_finds_a_severe_sliver() -> None:
    geometry = MeshGeometry(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1e-8, 0.0),
        ],
        faces=[(0, 1, 2)],
    )

    report = analyze_mesh_health(geometry)

    assert MeshHealthIssue.SLIVER_FACES in report.issues


def test_analyze_mesh_health_counts_disconnected_components_without_trimesh_split() -> None:
    first = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second.apply_translation((2.0, 0.0, 0.0))
    geometry = MeshGeometry.from_trimesh(trimesh.util.concatenate([first, second]))

    report = analyze_mesh_health(geometry)

    assert report.component_count == 2
    finding = next(
        finding
        for finding in report.findings
        if finding.issue is MeshHealthIssue.MULTIPLE_COMPONENTS
    )
    assert finding.count == 2
    assert finding.bounds is not None


def test_analyze_mesh_health_accepts_a_closed_inner_cavity() -> None:
    shell = boolean_difference(
        BoxGeometry((1.0, 1.0, 1.0)),
        BoxGeometry((0.8, 0.8, 0.8)),
    )

    report = analyze_mesh_health(shell)

    assert report.healthy
    assert report.component_count == 2
    assert report.signed_volume > 0.0


def test_analyze_mesh_health_rejects_a_detached_inward_solid() -> None:
    outer = BoxGeometry((1.0, 1.0, 1.0))
    detached = BoxGeometry((0.2, 0.2, 0.2)).translate(2.0, 0.0, 0.0)
    detached.faces = [(first, third, second) for first, second, third in detached.faces]
    outer.merge(detached)

    report = analyze_mesh_health(outer)

    assert not report.healthy
    finding = next(
        finding
        for finding in report.findings
        if finding.issue is MeshHealthIssue.INWARD_ORIENTATION
    )
    assert finding.count == 1
    assert finding.bounds is not None


def test_analyze_mesh_health_finds_inconsistent_winding() -> None:
    box = BoxGeometry((1.0, 1.0, 1.0)).to_trimesh()
    faces = box.faces.copy()
    faces[0] = faces[0][::-1]
    geometry = MeshGeometry.from_trimesh(
        trimesh.Trimesh(vertices=box.vertices, faces=faces, process=False)
    )

    report = analyze_mesh_health(geometry)

    assert not report.winding_consistent
    assert MeshHealthIssue.INCONSISTENT_WINDING in report.issues

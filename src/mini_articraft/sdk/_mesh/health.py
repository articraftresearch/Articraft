from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import igl
import numpy as np
import trimesh

from mini_articraft.sdk._mesh.core import MeshGeometry, geometry_to_trimesh
from mini_articraft.sdk.errors import ValidationError

Vec3: TypeAlias = tuple[float, float, float]
Bounds: TypeAlias = tuple[Vec3, Vec3]

DEFAULT_SLIVER_QUALITY = 1e-4


class MeshHealthIssue(StrEnum):
    """A specific defect found in a triangle mesh."""

    DEGENERATE_FACES = "degenerate_faces"
    SLIVER_FACES = "sliver_faces"
    DUPLICATE_FACES = "duplicate_faces"
    DUPLICATE_VERTICES = "duplicate_vertices"
    UNUSED_VERTICES = "unused_vertices"
    BOUNDARY_EDGES = "boundary_edges"
    NONMANIFOLD_EDGES = "nonmanifold_edges"
    MULTIPLE_COMPONENTS = "multiple_components"
    INCONSISTENT_WINDING = "inconsistent_winding"
    INWARD_ORIENTATION = "inward_orientation"


@dataclass(frozen=True)
class MeshHealthFinding:
    """One counted mesh defect with an optional affected region."""

    issue: MeshHealthIssue
    count: int
    bounds: Bounds | None = None
    details: str = ""


@dataclass(frozen=True)
class MeshHealthReport:
    """Topology and triangle quality measurements for one mesh."""

    vertex_count: int
    triangle_count: int
    component_count: int
    watertight: bool
    winding_consistent: bool
    signed_volume: float
    findings: tuple[MeshHealthFinding, ...]

    @property
    def healthy(self) -> bool:
        return not self.findings

    @property
    def issues(self) -> tuple[MeshHealthIssue, ...]:
        return tuple(finding.issue for finding in self.findings)


def analyze_mesh_health(
    geometry: object,
    *,
    tolerance: float = 0.001,
    sliver_quality: float = DEFAULT_SLIVER_QUALITY,
) -> MeshHealthReport:
    """Measure broad topology and triangle quality problems.

    ``tolerance`` controls build123d tessellation. Topology uses a much smaller
    scale-relative tolerance so vertices separated only by numeric noise share
    the same canonical position.
    """

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValidationError("tolerance must be a positive finite number")
    sliver_quality = float(sliver_quality)
    if not math.isfinite(sliver_quality) or not 0.0 < sliver_quality < 1.0:
        raise ValidationError("sliver_quality must be between 0 and 1")

    mesh = _as_raw_mesh(geometry, tolerance)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValidationError("geometry produced an empty mesh")

    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    merge_tolerance = max(scale * 1e-9, 1e-12)
    canonical_vertices, vertex_ids = _canonical_vertices(vertices, merge_tolerance)
    canonical_faces = vertex_ids[faces]

    used_vertices = np.unique(faces)
    unused_vertices = np.setdiff1d(np.arange(len(vertices)), used_vertices)
    repeated_corner = (
        (canonical_faces[:, 0] == canonical_faces[:, 1])
        | (canonical_faces[:, 0] == canonical_faces[:, 2])
        | (canonical_faces[:, 1] == canonical_faces[:, 2])
    )
    triangles = canonical_vertices[canonical_faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    area_epsilon = max(scale * scale * 1e-14, np.finfo(np.float64).tiny)
    degenerate = repeated_corner | (doubled_area <= 2.0 * area_epsilon)

    edge_lengths_squared = (
        np.sum((triangles[:, 1] - triangles[:, 0]) ** 2, axis=1)
        + np.sum((triangles[:, 2] - triangles[:, 1]) ** 2, axis=1)
        + np.sum((triangles[:, 0] - triangles[:, 2]) ** 2, axis=1)
    )
    quality = np.divide(
        2.0 * math.sqrt(3.0) * doubled_area,
        edge_lengths_squared,
        out=np.zeros_like(doubled_area),
        where=edge_lengths_squared > 0.0,
    )
    slivers = (~degenerate) & (quality < sliver_quality)

    valid_face_indices = np.flatnonzero(~degenerate)
    valid_faces = canonical_faces[valid_face_indices]
    duplicate_faces = _duplicate_face_mask(valid_faces)
    unique_faces = valid_faces[~duplicate_faces]
    topology = _topology(unique_faces)
    component_ids = _face_components(len(unique_faces), topology.edge_face_indices)
    component_count = int(component_ids.max()) + 1 if len(component_ids) else 0
    component_volumes = np.asarray(
        [
            _signed_volume(canonical_vertices, unique_faces[component_ids == component])
            for component in range(component_count)
        ]
    )
    volume_epsilon = max(scale**3 * 1e-12, np.finfo(np.float64).tiny)
    positive_components = np.flatnonzero(component_volumes > volume_epsilon)
    inward_components = _inward_components_outside_positive_solids(
        canonical_vertices,
        unique_faces,
        component_ids,
        component_volumes,
        positive_components,
        volume_epsilon=volume_epsilon,
    )

    boundary_edges = topology.edges[topology.edge_counts == 1]
    nonmanifold_edges = topology.edges[topology.edge_counts > 2]
    inconsistent_edges = topology.edges[
        (topology.edge_counts == 2) & (topology.edge_direction_sums != 0)
    ]
    watertight = (
        len(boundary_edges) == 0
        and len(nonmanifold_edges) == 0
        and not np.any(degenerate)
        and not np.any(duplicate_faces)
    )
    winding_consistent = len(inconsistent_edges) == 0
    signed_volume = _signed_volume(canonical_vertices, unique_faces)

    findings: list[MeshHealthFinding] = []
    _add_index_finding(
        findings,
        MeshHealthIssue.DEGENERATE_FACES,
        canonical_vertices,
        canonical_faces[np.flatnonzero(degenerate)],
        details="faces have repeated positions or near-zero area",
    )
    _add_index_finding(
        findings,
        MeshHealthIssue.SLIVER_FACES,
        canonical_vertices,
        canonical_faces[np.flatnonzero(slivers)],
        details=f"triangle quality is below {sliver_quality:g}",
    )
    _add_index_finding(
        findings,
        MeshHealthIssue.DUPLICATE_FACES,
        canonical_vertices,
        valid_faces[np.flatnonzero(duplicate_faces)],
        details="faces use the same three positions",
    )
    duplicate_vertex_count = len(vertices) - len(canonical_vertices)
    if duplicate_vertex_count:
        findings.append(
            MeshHealthFinding(
                issue=MeshHealthIssue.DUPLICATE_VERTICES,
                count=duplicate_vertex_count,
                bounds=_bounds(vertices),
                details="vertices share the same position within numeric tolerance",
            )
        )
    if len(unused_vertices):
        findings.append(
            MeshHealthFinding(
                issue=MeshHealthIssue.UNUSED_VERTICES,
                count=len(unused_vertices),
                bounds=_bounds(vertices[unused_vertices]),
                details="vertices are not referenced by any face",
            )
        )
    _add_index_finding(
        findings,
        MeshHealthIssue.BOUNDARY_EDGES,
        canonical_vertices,
        boundary_edges,
        details="edges belong to only one face",
    )
    _add_index_finding(
        findings,
        MeshHealthIssue.NONMANIFOLD_EDGES,
        canonical_vertices,
        nonmanifold_edges,
        details="edges belong to more than two faces",
    )
    if len(positive_components) > 1:
        extra_faces = _faces_outside_largest_positive_component(
            unique_faces,
            component_ids,
            component_volumes,
        )
        findings.append(
            MeshHealthFinding(
                issue=MeshHealthIssue.MULTIPLE_COMPONENTS,
                count=len(positive_components),
                bounds=_bounds(canonical_vertices[np.unique(extra_faces)])
                if len(extra_faces)
                else None,
                details="mesh contains separate positive-volume solids",
            )
        )
    _add_index_finding(
        findings,
        MeshHealthIssue.INCONSISTENT_WINDING,
        canonical_vertices,
        inconsistent_edges,
        details="adjacent faces traverse a shared edge in the same direction",
    )
    if watertight and winding_consistent and len(inward_components):
        inward_faces = unique_faces[np.isin(component_ids, inward_components)]
        findings.append(
            MeshHealthFinding(
                issue=MeshHealthIssue.INWARD_ORIENTATION,
                count=len(inward_components),
                bounds=_bounds(canonical_vertices[np.unique(inward_faces)]),
                details=(
                    "closed components have non-positive volume outside any positive-volume solid"
                ),
            )
        )

    return MeshHealthReport(
        vertex_count=len(vertices),
        triangle_count=len(faces),
        component_count=component_count,
        watertight=watertight,
        winding_consistent=winding_consistent,
        signed_volume=signed_volume,
        findings=tuple(findings),
    )


def _require_healthy_mesh(
    geometry: MeshGeometry,
    *,
    operation: str,
    allowed_issues: tuple[MeshHealthIssue, ...] = (),
) -> MeshGeometry:
    report = analyze_mesh_health(geometry)
    findings = [finding for finding in report.findings if finding.issue not in allowed_issues]
    if not findings:
        return geometry
    details = []
    for finding in findings:
        text = f"{finding.issue.value}={finding.count}"
        if finding.bounds is not None:
            text += f" near {finding.bounds!r}"
        details.append(text)
    raise ValueError(
        f"{operation} produced unhealthy mesh geometry: {'; '.join(details)}. "
        "Adjust the source geometry or operation so it creates clean topology."
    )


@dataclass(frozen=True)
class _Topology:
    edges: np.ndarray
    edge_counts: np.ndarray
    edge_direction_sums: np.ndarray
    edge_face_indices: np.ndarray


def _as_raw_mesh(geometry: object, tolerance: float) -> trimesh.Trimesh:
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    if isinstance(geometry, MeshGeometry):
        return geometry.to_trimesh(process=False)
    return geometry_to_trimesh(geometry, tolerance)


def _canonical_vertices(
    vertices: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin = vertices.min(axis=0)
    quantized = np.rint((vertices - origin) / tolerance).astype(np.int64)
    _, unique_indices, inverse = np.unique(
        quantized,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    return vertices[unique_indices], inverse


def _duplicate_face_mask(faces: np.ndarray) -> np.ndarray:
    if not len(faces):
        return np.zeros(0, dtype=bool)
    canonical = np.sort(faces, axis=1)
    _, first_indices = np.unique(canonical, axis=0, return_index=True)
    duplicate = np.ones(len(faces), dtype=bool)
    duplicate[first_indices] = False
    return duplicate


def _topology(faces: np.ndarray) -> _Topology:
    if not len(faces):
        empty_edges = np.empty((0, 2), dtype=np.int64)
        return _Topology(
            edges=empty_edges,
            edge_counts=np.empty(0, dtype=np.int64),
            edge_direction_sums=np.empty(0, dtype=np.int64),
            edge_face_indices=np.empty((0, 2), dtype=np.int64),
        )
    directed = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    face_indices = np.tile(np.arange(len(faces)), 3)
    undirected = np.sort(directed, axis=1)
    edges, inverse, counts = np.unique(
        undirected,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    direction = np.where(directed[:, 0] == undirected[:, 0], 1, -1)
    direction_sums = np.bincount(inverse, weights=direction, minlength=len(edges)).astype(np.int64)
    edge_face_indices = np.column_stack([inverse, face_indices])
    return _Topology(
        edges=edges,
        edge_counts=counts,
        edge_direction_sums=direction_sums,
        edge_face_indices=edge_face_indices,
    )


def _face_components(face_count: int, edge_face_indices: np.ndarray) -> np.ndarray:
    if face_count == 0:
        return np.empty(0, dtype=np.int64)
    parents = np.arange(face_count)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    order = np.argsort(edge_face_indices[:, 0], kind="stable")
    ordered = edge_face_indices[order]
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end, 0] == ordered[start, 0]:
            end += 1
        first_face = int(ordered[start, 1])
        for row in ordered[start + 1 : end]:
            union(first_face, int(row[1]))
        start = end
    roots = np.asarray([find(index) for index in range(face_count)])
    _, component_ids = np.unique(roots, return_inverse=True)
    return component_ids


def _faces_outside_largest_positive_component(
    faces: np.ndarray,
    component_ids: np.ndarray,
    component_volumes: np.ndarray,
) -> np.ndarray:
    positive = np.flatnonzero(component_volumes > 0.0)
    if not len(positive):
        return np.empty((0, 3), dtype=np.int64)
    largest = int(positive[np.argmax(component_volumes[positive])])
    extra = np.isin(component_ids, positive) & (component_ids != largest)
    return faces[extra]


def _inward_components_outside_positive_solids(
    vertices: np.ndarray,
    faces: np.ndarray,
    component_ids: np.ndarray,
    component_volumes: np.ndarray,
    positive_components: np.ndarray,
    *,
    volume_epsilon: float,
) -> np.ndarray:
    inward = np.flatnonzero(component_volumes <= volume_epsilon)
    if not len(inward) or not len(positive_components):
        return inward

    invalid: list[int] = []
    for component in inward:
        if component_volumes[component] >= -volume_epsilon:
            invalid.append(int(component))
            continue
        component_faces = faces[component_ids == component]
        component_vertices = vertices[np.unique(component_faces)]
        if not any(
            _points_are_inside_component(
                component_vertices,
                vertices,
                faces[component_ids == positive],
            )
            for positive in positive_components
        ):
            invalid.append(int(component))
    return np.asarray(invalid, dtype=np.int64)


def _points_are_inside_component(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> bool:
    signed_distances, _face_indices, _closest_points, _normals = igl.signed_distance(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )
    return bool(np.all(np.asarray(signed_distances) < 0.0))


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if not len(faces):
        return 0.0
    triangles = vertices[faces]
    return float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )


def _add_index_finding(
    findings: list[MeshHealthFinding],
    issue: MeshHealthIssue,
    vertices: np.ndarray,
    indices: np.ndarray,
    *,
    details: str,
) -> None:
    if not len(indices):
        return
    findings.append(
        MeshHealthFinding(
            issue=issue,
            count=len(indices),
            bounds=_bounds(vertices[np.unique(indices)]),
            details=details,
        )
    )


def _bounds(vertices: np.ndarray) -> Bounds:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return (
        (float(minimum[0]), float(minimum[1]), float(minimum[2])),
        (float(maximum[0]), float(maximum[1]), float(maximum[2])),
    )


__all__ = [
    "DEFAULT_SLIVER_QUALITY",
    "MeshHealthFinding",
    "MeshHealthIssue",
    "MeshHealthReport",
    "analyze_mesh_health",
]

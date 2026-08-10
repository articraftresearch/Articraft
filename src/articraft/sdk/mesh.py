"""Advanced mesh authoring and repair toolkit.

Geometry classes live at the package root (``articraft.sdk``) with the
rest of the object-authoring API. This module is the canonical home for mesh
operations and recipe helpers: booleans, welds, snapping, profile and wire
sampling, sweep helpers, section lofts, shell partitioning, and refinement.
"""

from __future__ import annotations

from articraft.sdk._mesh.boolean import (
    boolean_difference,
    boolean_intersection,
    boolean_union,
    cut_opening_on_face,
)
from articraft.sdk._mesh.core import build123d_to_mesh
from articraft.sdk._mesh.profiles import (
    WirePath,
    resample_side_sections,
    rounded_rect_profile,
    sample_arc_3d,
    sample_catmull_rom_spline_2d,
    sample_catmull_rom_spline_3d,
    sample_cubic_bezier_spline_2d,
    sample_cubic_bezier_spline_3d,
    split_superellipse_side_loft,
    superellipse_profile,
    superellipse_side_loft,
)
from articraft.sdk._mesh.refine import refine_mesh, smooth_mesh, subdivide_mesh
from articraft.sdk._mesh.section_loft import (
    LoftSection,
    SectionLoftSpec,
    repair_loft,
    section_loft,
)
from articraft.sdk._mesh.shell_partition import (
    ShellPartitionRegion,
    ShellPartitionSpec,
    partition_shell,
)
from articraft.sdk._mesh.sweeps import (
    SweepSection,
    sweep_profile_along_spline,
    tube_from_spline_points,
    tube_network_from_paths,
    wire_from_points,
)
from articraft.sdk._mesh.weld import SnapRefused, smooth_difference, snap_to, weld

__all__ = [
    "LoftSection",
    "SectionLoftSpec",
    "ShellPartitionRegion",
    "ShellPartitionSpec",
    "SnapRefused",
    "SweepSection",
    "WirePath",
    "boolean_difference",
    "boolean_intersection",
    "boolean_union",
    "build123d_to_mesh",
    "cut_opening_on_face",
    "partition_shell",
    "refine_mesh",
    "repair_loft",
    "resample_side_sections",
    "rounded_rect_profile",
    "sample_arc_3d",
    "sample_catmull_rom_spline_2d",
    "sample_catmull_rom_spline_3d",
    "sample_cubic_bezier_spline_2d",
    "sample_cubic_bezier_spline_3d",
    "section_loft",
    "smooth_difference",
    "smooth_mesh",
    "snap_to",
    "split_superellipse_side_loft",
    "subdivide_mesh",
    "superellipse_profile",
    "superellipse_side_loft",
    "sweep_profile_along_spline",
    "tube_from_spline_points",
    "tube_network_from_paths",
    "weld",
    "wire_from_points",
]

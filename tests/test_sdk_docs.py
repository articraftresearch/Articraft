from __future__ import annotations

import ast
import re
import runpy
import tomllib
from pathlib import Path

import articraft.sdk as sdk
import articraft.sdk.mesh as sdk_mesh
from articraft import package_dir
from articraft.errors import MiniArticraftError
from articraft.sdk import RigidBodyAssembly, TestReport
from articraft.sdk.errors import SDKError, ValidationError

_INSTRUCTION_VERBS = {
    "add",
    "apply",
    "attach",
    "call",
    "catch",
    "change",
    "check",
    "choose",
    "compare",
    "convert",
    "create",
    "do",
    "enable",
    "give",
    "import",
    "keep",
    "leave",
    "omit",
    "open",
    "pass",
    "prefer",
    "read",
    "remove",
    "run",
    "select",
    "set",
    "start",
    "use",
}


def first_party_prose_paths() -> list[Path]:
    repo_root = package_dir.parents[1]
    sdk_docs = package_dir / "sdk" / "docs"
    return [
        repo_root / "docs" / "sdk.md",
        *sorted(sdk_docs.joinpath("common").glob("*.md")),
        *sorted(sdk_docs.joinpath("mesh").glob("*.md")),
    ]


def markdown_prose_blocks(path: Path) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    paragraph: list[str] = []
    paragraph_line = 0
    in_code = False

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append((paragraph_line, " ".join(paragraph)))
            paragraph = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("```"):
            flush()
            in_code = not in_code
            continue
        if (
            in_code
            or not line.strip()
            or line.startswith("#")
            or line.startswith("|")
            or line.lstrip().startswith("<")
        ):
            flush()
            continue

        list_item = re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)
        if list_item:
            flush()
            paragraph_line = line_number
        elif not paragraph:
            paragraph_line = line_number
        paragraph.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line.strip()))

    flush()
    return blocks


def detailed_reference_paths(sdk_docs: Path) -> list[str]:
    return sorted(
        path.relative_to(sdk_docs).as_posix()
        for folder in ("common", "mesh")
        for path in sdk_docs.joinpath(folder).glob("*.md")
        if path.name != "00_quickstart.md"
    )


def test_quickstart_is_short_and_routes_to_targeted_references() -> None:
    sdk_docs = package_dir / "sdk" / "docs"
    quickstart = sdk_docs.joinpath("common", "00_quickstart.md").read_text()

    assert len(quickstart) < 5000
    for reference in [
        "docs/sdk/common/30_assembly.md",
        "docs/sdk/common/35_joints.md",
        "docs/sdk/common/40_testing.md",
        "docs/sdk/mesh/00_mesh_geometry.md",
        "docs/sdk/build123d/",
    ]:
        assert reference in quickstart
    assert quickstart.count("docs/sdk/common/") >= 9
    assert quickstart.count("docs/sdk/mesh/") >= 6
    assert "read every" not in quickstart.lower()


def test_readme_and_public_doc_links_resolve() -> None:
    repo_root = package_dir.parents[1]
    paths = [repo_root / "README.md", *sorted(repo_root.joinpath("docs").glob("*.md"))]
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")

    for source in paths:
        for target in pattern.findall(source.read_text(encoding="utf-8")):
            path_text = target.split("#", 1)[0]
            if not path_text or "://" in path_text or path_text.startswith("mailto:"):
                continue
            assert source.parent.joinpath(path_text).exists(), (source.name, target)


def test_first_party_sdk_prose_uses_short_sentences_and_plain_punctuation() -> None:
    failures: list[str] = []

    for path in first_party_prose_paths():
        for line_number, block in markdown_prose_blocks(path):
            if match := re.search(r"[;\u2013\u2014]", block):
                failures.append(f"{path}:{line_number}: disallowed punctuation {match.group()!r}")

            text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", block)
            text = re.sub(r"`[^`]+`", "TECH", text)
            for sentence in re.split(r"(?<=[.!?])\s+|:\s*$", text):
                words = re.findall(r"[A-Za-z0-9]+", sentence)
                if not words:
                    continue
                limit = 20 if words[0].lower() in _INSTRUCTION_VERBS else 25
                if len(words) > limit:
                    failures.append(
                        f"{path}:{line_number}: {len(words)} words exceeds {limit}: {sentence}"
                    )

    assert not failures, "\n".join(failures)


def test_quickstart_routes_every_detailed_sdk_reference() -> None:
    sdk_docs = package_dir / "sdk" / "docs"
    quickstart = sdk_docs.joinpath("common", "00_quickstart.md").read_text()

    for reference in detailed_reference_paths(sdk_docs):
        assert sdk_docs.joinpath(reference).is_file(), reference
        assert f"docs/sdk/{reference}" in quickstart, reference


def test_every_public_sdk_symbol_is_documented() -> None:
    sdk_docs = package_dir / "sdk" / "docs"
    reference_text = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("common", "mesh")
        for path in sorted(sdk_docs.joinpath(folder).glob("*.md"))
    )

    missing = sorted(
        name for name in {*sdk.__all__, *sdk_mesh.__all__} if f"`{name}`" not in reference_text
    )
    assert not missing


def test_root_and_mesh_export_disjoint_surfaces() -> None:
    """One canonical import path per category: geometry classes and the object
    API at the root, mesh operations and recipes under ``articraft.sdk.mesh``."""
    assert set(sdk.__all__).isdisjoint(sdk_mesh.__all__)


def test_sdk_is_a_leaf_package_with_compatible_errors() -> None:
    sdk_root = package_dir / "sdk"
    outside_imports: list[tuple[str, str]] = []

    for path in sdk_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            outside_imports.extend(
                (path.name, module)
                for module in modules
                if module.startswith("articraft.") and not module.startswith("articraft.sdk")
            )

    assert not outside_imports
    assert sdk.SDKError is SDKError
    assert sdk.ValidationError is ValidationError
    assert issubclass(ValidationError, SDKError)
    assert issubclass(SDKError, MiniArticraftError)


def test_key_apis_are_documented_by_their_owner_pages() -> None:
    sdk_docs = package_dir / "sdk" / "docs"
    expected = {
        "common/30_assembly.md": (
            "`RigidBodyAssembly`",
            "`Material`",
            "`RigidBody`",
            "`body.add(...)`",
        ),
        "common/35_joints.md": (
            "`JointFrame`",
            "`JointDOF`",
            "`model.joint(...)`",
            "`model.articulation(...)`",
        ),
        "common/40_testing.md": (
            "`TestContext`",
            "`TestReport`",
            "`allow_overlap",
        ),
        "mesh/00_mesh_geometry.md": (
            "`MeshGeometry`",
            "`build123d_to_mesh`",
            "`ExtrudeWithHolesGeometry`",
        ),
        "mesh/10_profiles.md": (
            "`rounded_rect_profile`",
            "`superellipse_profile`",
            "`sample_catmull_rom_spline_3d`",
        ),
        "mesh/20_wires_and_sweeps.md": (
            "`WirePath`",
            "`PipeGeometry`",
            "`tube_network_from_paths`",
        ),
        "mesh/30_section_lofts.md": (
            "`SectionLoftSpec`",
            "`section_loft`",
            "`repair_loft`",
        ),
        "mesh/40_booleans_and_shells.md": (
            "`boolean_difference`",
            "`cut_opening_on_face`",
            "`partition_shell`",
        ),
        "mesh/50_refinement_and_smoothing.md": (
            "`refine_mesh`",
            "`subdivide_mesh`",
            "`smooth_mesh`",
        ),
    }

    for relative_path, names in expected.items():
        text = sdk_docs.joinpath(relative_path).read_text(encoding="utf-8")
        for name in names:
            assert name in text, f"{name} missing from {relative_path}"


def test_build123d_reference_tree_and_examples_are_available() -> None:
    root = package_dir / "sdk" / "docs" / "build123d"

    assert (root / "VENDORED.md").is_file()
    assert (root / "introduction.md").is_file()
    assert (root / "examples" / "benchy.py").is_file()
    assert (root / "examples" / "low_poly_benchy.stl").is_file()
    assert (root / "assets" / "ttt" / "ttt-23-t-24-curved_support.py").is_file()
    assert (root / "assets" / "ttt" / "ttt-23-t-24-curved_support.png").is_file()
    assert (root / "media" / "tea_cup.png").is_file()
    assert (root / "snippets" / "selector_example.py").is_file()
    assert not (root / "index.md").exists()


def test_every_vendored_svg_has_a_model_readable_preview() -> None:
    root = package_dir / "sdk" / "docs" / "build123d"
    svg_paths = sorted(root.rglob("*.svg"))

    assert svg_paths
    for path in svg_paths:
        assert Path(f"{path}.webp").is_file(), path


def test_all_backticked_sdk_doc_paths_resolve() -> None:
    sdk_docs_root = package_dir / "sdk" / "docs"
    text = "\n".join(path.read_text(encoding="utf-8") for path in sdk_docs_root.rglob("*.md"))
    pattern = re.compile(r"`(docs/sdk/[^`]+)`")

    for doc_path in sorted(set(pattern.findall(text))):
        if "{" in doc_path or "}" in doc_path or doc_path.endswith("/"):
            continue
        local_path = sdk_docs_root / Path(doc_path).relative_to("docs/sdk")
        assert local_path.exists(), doc_path


def test_relative_markdown_doc_links_resolve() -> None:
    sdk_docs = package_dir / "sdk" / "docs"
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")

    for relative_path in detailed_reference_paths(sdk_docs):
        source = sdk_docs / relative_path
        for target in pattern.findall(source.read_text(encoding="utf-8")):
            path_text = target.split("#", 1)[0]
            if not path_text.endswith(".md"):
                continue
            assert not Path(path_text).is_absolute(), (relative_path, target)
            assert source.parent.joinpath(path_text).is_file(), (relative_path, target)


def test_sdk_docs_and_examples_are_package_data() -> None:
    repo_root = package_dir.parents[1]
    pyproject = tomllib.loads(repo_root.joinpath("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["articraft"]

    for pattern in [
        "sdk/docs/common/*.md",
        "sdk/docs/mesh/*.md",
        "sdk/docs/examples/*.py",
        "sdk/docs/build123d/*.md",
        "sdk/docs/build123d/assets/*",
        "sdk/docs/build123d/assets/*/*",
        "sdk/docs/build123d/examples/*",
        "sdk/docs/build123d/media/*",
        "sdk/docs/build123d/snippets/*",
    ]:
        assert pattern in package_data


def test_all_first_party_sdk_examples_execute() -> None:
    repo_root = package_dir.parents[1]
    paths = [
        *(package_dir / "sdk" / "docs" / "examples").glob("*.py"),
        *repo_root.glob("examples/*/main.py"),
    ]

    for path in sorted(paths):
        values = runpy.run_path(str(path))
        model = values["object_model"]
        report = values["run_tests"]()
        assert isinstance(model, RigidBodyAssembly), path.name
        model.validate()
        assert isinstance(report, TestReport), path.name
        assert report.passed, (path.name, report.failures)


def test_prompt_and_docs_state_the_new_authoring_contract() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            package_dir / "prompts" / "system.md",
            package_dir / "prompts" / "task.md",
            package_dir / "sdk" / "docs" / "common" / "00_quickstart.md",
            package_dir / "sdk" / "docs" / "common" / "20_core_types.md",
            package_dir / "sdk" / "docs" / "common" / "35_joints.md",
        ]
    )

    for required in [
        "meters",
        "radians",
        "build123d",
        "MeshGeometry",
        "name=",
        "JointFrame",
        "JointDOF",
        "realistic geometry",
        "intentional motion",
        "removable part",
        "unintended overlaps",
    ]:
        assert required.lower() in text.lower()


def test_prompts_encourage_research_and_mesh_without_a_usage_quota() -> None:
    system = (package_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    task = (package_dir / "prompts" / "task.md").read_text(encoding="utf-8")
    text = f"{system}\n{task}"
    normalized = " ".join(text.split()).lower()

    for required in [
        "complementary authoring choices",
        "Research plausible approaches before you",
        "Do not stop at the first workable API",
        "geometry strategy for each major visible form",
        "A successful compile does not replace visual inspection",
        "Mesh usage is not a goal by itself",
        "Prefer build123d when exact boundaries",
        "field based weld rebuilds every input surface",
        "allow_mesh_issues",
    ]:
        assert required.lower() in normalized

    for forbidden in [
        "from build123d import *",
        "must use mesh",
        "use at least one mesh",
        "read exactly",
        "jet engine",
        "stand mixer",
        "desk lamp",
        "realistic articulated 3d object",
        "model the primary mechanism",
        "lid_hinge",
    ]:
        assert forbidden.lower() not in normalized


def test_prompts_do_not_force_articulation_or_invent_retention() -> None:
    system = (package_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    task = (package_dir / "prompts" / "task.md").read_text(encoding="utf-8")
    quickstart = (package_dir / "sdk" / "docs" / "common" / "00_quickstart.md").read_text(
        encoding="utf-8"
    )
    text = " ".join(f"{system}\n{task}\n{quickstart}".split()).lower()

    for required in [
        "3d object or assembly",
        "model only the motion the design requires",
        "a removable part may be a separate seated body",
        "do not invent a hinge",
        "a loose teapot or saucepan lid is a separate seated part",
        "name the main motion only when independent motion is part of the design",
        "blocking threshold only for an explicit requirement or a clear defect",
    ]:
        assert required in text

    for forbidden in [
        "turn the user's request into a realistic articulated 3d object",
        "model the primary mechanism",
        "state what you built and name the main motion",
        "lid_hinge",
    ]:
        assert forbidden not in text


def test_prompt_requires_agent_driven_visual_review() -> None:
    system = (package_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    visual_docs = (package_dir / "sdk" / "docs" / "common" / "45_visual_evidence.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(system.split())
    normalized_docs = " ".join(visual_docs.split())

    for required in [
        "Write `previews.py`",
        '"$ARTICRAFT_PYTHON" previews.py',
        "before the first compile",
        "Calling the renderer is not visual inspection",
        "silhouette, proportions, part transitions, repeated features",
        "Every validation brief must include at least one overall model view",
        "Register the final useful preview files",
        "The compiler does not render or copy images",
        "Run `compile` after an actual file change",
    ]:
        assert required.lower() in normalized.lower()

    for required in [
        "from main import object_model",
        "qa/previews/assembly.png",
        "Working previews can stay",
        "does not render or copy images",
    ]:
        assert required.lower() in normalized_docs.lower()

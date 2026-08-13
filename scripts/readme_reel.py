"""Render the README reel: runs side by side, rotating and moving their joints.

Frames come from the same renderer the live viewer uses, driven headlessly:
USDZ through USDLoader, authored PBR appearances, image based lighting, ACES
tone mapping. Each panel spins one full turn. Joints replay the recorded
joints sweep their limits, solved through the SDK so closed loops stay shut.

    uv run python scripts/readme_reel.py runs/<run-id>:"a toolbox" runs/<other>

The label after the colon is optional; the run directory name is the default.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

import typer
from PIL import Image, ImageChops

from articraft.sdk import JointAxis
from articraft.sdk.assembly import _frame_matrix
from articraft.viewer import load_viewer_run

BACKGROUND = "#f7f8fa"
PAGE = Path(__file__).with_name("reel.html")
CHROMES = (
    Path.home()
    / "Library/Caches/ms-playwright/chromium_headless_shell-1228"
    / "chrome-headless-shell-mac-arm64/chrome-headless-shell",
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
)
app = typer.Typer(help=__doc__, no_args_is_help=True)


def _poses(run: Path, version: dict, frames: int) -> list[dict[str, list[float]]]:
    """One body placement per frame: {body name: 16 numbers, column major}.

    The SDK solves the kinematics, including any closed loop, and the page just
    applies the matrices. Nothing about joints or frames crosses into
    JavaScript, so there is only ever one implementation of the motion.
    """
    import contextlib
    import importlib.util
    import io
    import sys

    import numpy as np

    workspace = run / "workspace"
    spec = importlib.util.spec_from_file_location(f"reel_{run.name}", workspace / "main.py")
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"cannot import {run}/workspace/main.py")
    module = importlib.util.module_from_spec(spec)
    # A run may split its code across files, so its own directory has to be
    # importable while main.py loads. The module also has to be registered:
    # dataclasses defined in it look themselves up through sys.modules later,
    # and find None if it was only ever executed.
    sys.path.insert(0, str(workspace))
    sys.modules[spec.name] = module
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(workspace))
    resolved = module.object_model.resolve()

    tree = [item.joint for item in resolved.joints if not item.exclude_from_articulation]
    closers = [item.joint for item in resolved.joints if item.exclude_from_articulation]

    # A ring decides most of its own joints. Sweeping all of them drives a four
    # bar against itself, so each ring keeps one driver and the rest are solved.
    hangs = {joint.body1: joint for joint in tree}

    def to_root(body: object) -> list:
        walk, seen = [], set()
        while body in hangs and id(body) not in seen:
            seen.add(id(body))
            walk.append(hangs[body])
            body = hangs[body].body0
        return walk

    rings: list[list] = []
    for closer in closers:
        near, far = to_root(closer.body0), to_root(closer.body1)
        far_names = {joint.name for joint in far}
        near_names = {joint.name for joint in near}
        rings.append([j for j in near if j.name not in far_names]
                     + [j for j in far if j.name not in near_names])

    # Rings that share a joint are one mechanism -- a gripper's two dogbones
    # both hang off the same slider -- so they get one driver between them.
    # Merging has to be transitive: an excavator's bucket ring and its stick
    # ring both touch the stick hinge, and left as separate groups they pick
    # two drivers that fight over it, which shrinks the whole loop to whatever
    # amplitude that conflict still allows.
    merged: list[list] = []
    for ring in rings:
        names = {j.name for j in ring}
        touching = [g for g in merged if names & {j.name for j in g}]
        combined = list(ring)
        for group in touching:
            combined.extend(j for j in group if j.name not in {c.name for c in combined})
            merged.remove(group)
        merged.append(combined)

    def reach_of(dof) -> float:
        """How far this joint travels from rest, out toward its further stop.

        Rest is the authored pose and the travel that matters usually runs one
        way from it: a lid opens from shut, a gear retracts from down. Capped so
        a joint with no real stops does not wind through a full turn.
        """
        if dof.limits is None:
            return 0.0
        lower, upper = dof.limits
        far = upper if abs(upper) >= abs(lower) else lower
        limit = 1.2 if cast(JointAxis, dof.axis).is_rotational else 0.4
        return max(-limit, min(limit, far * 0.75))

    def moves(candidate) -> bool:
        """Does driving this joint actually pose the mechanism?

        Which joint of a ring is the input is not something the graph says: a
        gripper is driven by its slider, not by a finger, and asking the finger
        leaves the loop unsatisfiable. So try it and see.
        """
        dof = candidate.dofs[0]
        if dof.limits is None:
            return True
        # Test at the value the sweep will actually use: a driver that solves
        # halfway and fails at full travel silently falls back to rest, which
        # is how a whole panel ends up standing still.
        try:
            state = resolved.forward_kinematics({candidate.dof_id(dof): reach_of(dof)})
        except Exception:
            return False
        return any(abs(v) > 1e-6 for v in state.dof_positions.values())

    followers: set[str] = set()
    for group in merged:
        usable = [j for j in group if len(j.dofs) == 1 and moves(j)]
        keep = usable[0] if usable else None
        followers.update(j.name for j in group if keep is None or j.name != keep.name)

    def turns_about_vertical(joint) -> bool:
        """Does this joint spin the object the way the reel already does?

        Every panel rotates a full turn about world up. A joint on that same
        axis -- an excavator's slew ring -- adds to the camera instead of
        showing anything new, and the panel appears to judder while the others
        look smooth.
        """
        dof = joint.dofs[0]
        axis = cast(JointAxis, dof.axis)
        if not axis.is_rotational:
            return False
        direction = np.zeros(3)
        direction[axis.component] = 1.0
        world = _frame_matrix(joint.frame0)[:3, :3] @ direction
        return abs(float(world[2])) > 0.99

    candidates = [
        joint for joint in tree if len(joint.dofs) == 1 and joint.name not in followers
    ]
    # Dropping the turntable only helps while something else still moves. A tool
    # that lies flat -- a pair of pliers -- has its one hinge on that axis too,
    # and skipping it would leave the panel standing still.
    upright = [joint for joint in candidates if not turns_about_vertical(joint)]
    driven = [(joint, joint.dofs[0]) for joint in (upright or candidates)]

    def wanted(blend: float, turn: float, scale: float) -> dict[str, float]:
        asked: dict[str, float] = {}
        for joint, dof in driven:
            if dof.limits is None:
                asked[joint.dof_id(dof)] = turn
                continue
            lower, upper = dof.limits
            asked[joint.dof_id(dof)] = max(
                lower, min(upper, reach_of(dof) * blend * scale)
            )
        return asked

    def blends(count: int) -> list[tuple[float, float]]:
        # Ease in and out so the loop has no visible seam.
        return [
            (
                0.5 - 0.5 * math.cos(2.0 * math.pi * index / count),
                2.0 * math.pi * index / count,
            )
            for index in range(count)
        ]

    # One amplitude for the whole loop, the largest every frame can hold. A
    # linkage that cannot reach the deepest part of a sweep used to be caught
    # per frame and backed off there, which put a jump in the middle of the
    # motion; scaling the loop instead keeps it smooth and merely shallower.
    scale = 1.0
    for candidate in (1.0, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1):
        try:
            for blend, turn in blends(frames):
                resolved.forward_kinematics(wanted(blend, turn, candidate))
        except Exception:
            continue
        scale = candidate
        break
    else:
        scale = 0.0

    placements: list[dict[str, list[float]]] = []
    for blend, turn in blends(frames):
        try:
            state = resolved.forward_kinematics(wanted(blend, turn, scale))
        except Exception:
            state = resolved.forward_kinematics({})
        placements.append(
            {
                name: [float(v) for v in np.asarray(matrix, dtype=float).T.flatten()]
                for name, matrix in state.body_poses.items()
            }
        )
    return placements


def _handler(bootstrap: bytes, models: dict[str, Path], captured: dict[tuple[int, int], bytes]):
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        finished_event = finished

        def log_message(self, format: str, *args: object) -> None:  # keep the console quiet
            del format, args

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(PAGE.read_bytes(), "text/html")
            elif self.path == "/bootstrap.json":
                self._send(bootstrap, "application/json")
            elif self.path.startswith("/models/"):
                name = self.path.removeprefix("/models/").removesuffix(".usdz")
                self._send(models[name].read_bytes(), "model/vnd.usdz+zip")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/done":
                finished.set()
            else:
                _, _, panel, frame = self.path.split("/")
                captured[(int(panel), int(frame))] = body
            self._send(b"", "text/plain")

    return Handler, finished


def _above_noise(value: int) -> int:
    """Anything more than a hair off the background counts as content."""

    return 255 if value > 6 else 0


def _trim(shots: list[Image.Image], size: int) -> list[Image.Image]:
    """Crop a panel to what it actually draws, so thin objects keep no dead air.

    The crop is the union across every frame, so the object still holds still
    inside its panel while it turns.
    """
    canvas = Image.new("RGB", shots[0].size, BACKGROUND)
    box = None
    for shot in shots:
        mask = ImageChops.difference(shot, canvas).convert("L")
        found = mask.point(_above_noise)
        bounds = found.getbbox()
        if bounds is None:
            continue
        box = (
            bounds
            if box is None
            else (
                min(box[0], bounds[0]),
                min(box[1], bounds[1]),
                max(box[2], bounds[2]),
                max(box[3], bounds[3]),
            )
        )
    if box is None:
        box = (0, 0, shots[0].width, shots[0].height)
    pad = round(shots[0].height * 0.04)
    box = (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(shots[0].width, box[2] + pad),
        min(shots[0].height, box[3] + pad),
    )
    height = box[3] - box[1]
    width = round((box[2] - box[0]) * size / height)
    # A very wide object (pliers, a wrench) would dominate a grid row; letterbox
    # it instead so every panel stays within a tame aspect.
    cap = round(size * 1.6)
    if width <= cap:
        return [shot.crop(box).resize((width, size), Image.Resampling.LANCZOS) for shot in shots]
    scale = cap / width
    inner = max(1, round(size * scale))
    panels = []
    for shot in shots:
        image = shot.crop(box).resize((cap, inner), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (cap, size), BACKGROUND)
        panel.paste(image, (0, (size - inner) // 2))
        panels.append(panel)
    return panels


def _chrome() -> str:
    for path in CHROMES:
        if path.exists():
            return str(path)
    found = shutil.which("chromium") or shutil.which("google-chrome")
    if not found:
        raise typer.BadParameter("no Chrome or Chromium found to render with")
    return found


@app.command()
def main(
    runs: Annotated[list[str], typer.Argument(help="Run directories, as path or path:label.")],
    output: Annotated[Path, typer.Option(help="GIF to write.")] = Path("assets/readme/reel.gif"),
    frames: Annotated[int, typer.Option(help="Frames per rotation.")] = 30,
    size: Annotated[int, typer.Option(help="Panel size in pixels.")] = 320,
    fps: Annotated[float, typer.Option(help="Playback rate.")] = 20.0,
    elevation: Annotated[float, typer.Option(help="Camera height, as a fraction.")] = 0.34,
    zoom: Annotated[float, typer.Option(help="Fit margin; 1.0 just fits, more adds air.")] = 1.0,
    gap: Annotated[int, typer.Option(help="Pixels of air between panels.")] = 28,
    rows: Annotated[int, typer.Option(help="Grid rows to split the panels into.")] = 1,
    supersample: Annotated[int, typer.Option(help="Render scale before downsampling.")] = 3,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for the browser.")] = 600.0,
) -> None:
    """Render one side-by-side rotating loop from the given runs."""
    panels, models = [], {}
    for index, entry in enumerate(runs):
        run, _, _ = entry.partition(":")
        version = load_viewer_run(Path(run)).versions[0]
        identifier = f"panel{index}"
        models[identifier] = load_viewer_run(Path(run)).files[str(version["id"])]
        panels.append(
            {
                "id": identifier,
                "model": version["model"],
                "poses": _poses(Path(run), version, frames),
            }
        )

    bootstrap = json.dumps(
        {
            "panels": panels,
            "frames": frames,
            "size": size,
            "elevation": elevation,
            "zoom": zoom,
            "supersample": supersample,
            "background": BACKGROUND,
        }
    ).encode()
    captured: dict[tuple[int, int], bytes] = {}
    handler, finished = _handler(bootstrap, models, captured)

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/"
        browser = subprocess.Popen(
            [
                _chrome(),
                "--headless=new",
                "--disable-gpu",
                "--enable-unsafe-swiftshader",
                "--use-angle=swiftshader",
                "--hide-scrollbars",
                f"--window-size={size},{size}",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            if not finished.wait(timeout):
                raise typer.BadParameter(
                    f"browser captured {len(captured)}/{frames * len(panels)} frames "
                    f"before the {timeout:.0f}s timeout"
                )
        finally:
            browser.terminate()

    columns = []
    for index in range(len(panels)):
        shots = []
        for frame in range(frames):
            raw = captured.get((index, frame))
            if raw is None:
                raise typer.BadParameter(f"panel {index} never sent frame {frame}")
            shots.append(Image.open(BytesIO(raw)).convert("RGB"))
        columns.append(_trim(shots, size))

    per_row = -(-len(columns) // rows)  # ceil: panels per grid row
    strips = []
    for frame in range(frames):
        row_images = []
        for start in range(0, len(columns), per_row):
            chunk = [column[frame] for column in columns[start : start + per_row]]
            width = sum(image.width for image in chunk) + gap * (len(chunk) - 1)
            strip = Image.new("RGB", (width, size), BACKGROUND)
            x = 0
            for image in chunk:
                strip.paste(image, (x, 0))
                x += image.width + gap
            row_images.append(strip)
        total = max(image.width for image in row_images)
        grid = Image.new(
            "RGB", (total, size * len(row_images) + gap * (len(row_images) - 1)), BACKGROUND
        )
        y = 0
        for strip in row_images:
            grid.paste(strip, ((total - strip.width) // 2, y))
            y += size + gap
        strips.append(grid)

    output.parent.mkdir(parents=True, exist_ok=True)
    extra = (
        # WebP keeps 24 bit colour. GIF quantises to 256 and bands every
        # gradient, which is worst on exactly the glossy dark surfaces here.
        {"quality": 92, "method": 6} if output.suffix.lower() == ".webp" else {"optimize": True}
    )
    strips[0].save(
        output,
        save_all=True,
        append_images=strips[1:],
        duration=round(1000.0 / fps),
        loop=0,
        **extra,
    )
    typer.echo(
        f"{output} ({output.stat().st_size / 1e6:.1f} MB, {len(strips)} frames, "
        f"{strips[0].width}x{strips[0].height})"
    )


if __name__ == "__main__":
    app()

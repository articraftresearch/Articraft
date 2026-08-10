"""Render the README reel: runs side by side, rotating and moving their joints.

Frames come from the same renderer the live viewer uses, driven headlessly:
USDZ through USDLoader, authored PBR appearances, image based lighting, ACES
tone mapping. Each panel spins one full turn. Joints replay the recorded
simulation trajectory when a run has one, and otherwise sweep their limits.

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
from typing import Annotated

import typer
from PIL import Image, ImageChops

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


def _poses(version: dict, frames: int) -> list[dict[str, float]]:
    """One joint pose per frame: recorded trajectory, else a limit sweep."""
    trajectory = version.get("trajectory")
    if trajectory:
        names, steps = trajectory["joints"], trajectory["frames"]
        return [
            dict(
                zip(
                    names,
                    steps[round(index * (len(steps) - 1) / (frames - 1))]["joints"],
                    strict=True,
                )
            )
            for index in range(frames)
        ]
    joints = [joint for joint in version["model"]["articulations"] if joint["type"] != "fixed"]
    poses = []
    for index in range(frames):
        # Ease in and out, and stop short of the stops: a joint slamming into
        # its limit every loop looks like a bug rather than a demonstration.
        blend = 0.5 - 0.5 * math.cos(2.0 * math.pi * index / frames)
        turn = 2.0 * math.pi * index / frames
        pose = {}
        for joint in joints:
            limits = joint.get("motion_limits") or {}
            lower, upper = limits.get("lower"), limits.get("upper")
            if lower is None or upper is None:
                # A fan blade or a wheel has no stops; give it a full turn.
                if joint["type"] in ("revolute", "continuous"):
                    pose[joint["name"]] = turn
                continue
            pose[joint["name"]] = lower + (upper - lower) * (0.08 + 0.84 * blend)
        poses.append(pose)
    return poses


def _handler(bootstrap: bytes, models: dict[str, Path], captured: dict[tuple[int, int], bytes]):
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        finished_event = finished

        def log_message(self, *args) -> None:  # keep the console quiet
            pass

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


def _trim(shots: list[Image.Image], size: int) -> list[Image.Image]:
    """Crop a panel to what it actually draws, so thin objects keep no dead air.

    The crop is the union across every frame, so the object still holds still
    inside its panel while it turns.
    """
    canvas = Image.new("RGB", shots[0].size, BACKGROUND)
    box = None
    for shot in shots:
        found = ImageChops.difference(shot, canvas).convert("L").point(lambda v: v > 6 and 255)
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
    return [shot.crop(box).resize((width, size), Image.LANCZOS) for shot in shots]


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
                "poses": _poses(version, frames),
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

    strips = []
    for frame in range(frames):
        row = [column[frame] for column in columns]
        width = sum(image.width for image in row) + gap * (len(row) - 1)
        strip = Image.new("RGB", (width, size), BACKGROUND)
        x = 0
        for image in row:
            strip.paste(image, (x, 0))
            x += image.width + gap
        strips.append(strip)

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

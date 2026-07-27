from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import quote

from mini_articraft.sdk import ArticulatedObject, TestArtifact, TestReport
from mini_articraft.sdk.export import ExportAudit


def write_qa_report(
    *,
    workspace: Path,
    result_dir: Path,
    model: ArticulatedObject,
    test_report: TestReport,
    export_audit: ExportAudit,
) -> tuple[TestReport, Path]:
    qa_dir = result_dir / "qa"
    files_dir = qa_dir / "artifacts"
    qa_dir.mkdir(parents=True, exist_ok=True)

    report_artifacts: list[dict[str, str]] = []
    warnings = list(test_report.warnings)
    valid_artifacts: list[TestArtifact] = []
    used_names: set[str] = set()
    workspace_root = workspace.resolve()
    for artifact in test_report.artifacts:
        try:
            source = (workspace / artifact.path).resolve()
            source.relative_to(workspace_root)
            if not source.is_file():
                raise FileNotFoundError(artifact.path)
            if artifact.kind == "image":
                report_path = f"../../workspace/{quote(artifact.path, safe='/')}"
            else:
                files_dir.mkdir(exist_ok=True)
                filename = _unique_filename(source.name, used_names)
                destination = files_dir / filename
                shutil.copyfile(source, destination)
                report_path = destination.relative_to(qa_dir).as_posix()
            report_artifacts.append(asdict(artifact) | {"report_path": report_path})
            valid_artifacts.append(artifact)
        except (FileNotFoundError, OSError, ValueError) as exc:
            warnings.append(
                f"Artifact {artifact.name!r} could not be copied into the QA report: "
                f"{type(exc).__name__}: {exc}"
            )

    report = replace(
        test_report,
        warnings=tuple(dict.fromkeys(warnings)),
        artifacts=tuple(valid_artifacts),
    )
    payload = {
        "model": {
            "name": model.name,
            "parts": len(model.parts),
            "shapes": sum(1 for part in model.parts for _shape in part._iter_shapes()),
            "articulations": len(model.articulations),
        },
        "test_report": asdict(report),
        "export_audit": asdict(export_audit),
        "artifacts": report_artifacts,
    }
    report_json = qa_dir / "report.json"
    report_html = qa_dir / "report.html"
    report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_html.write_text(_render_html(payload), encoding="utf-8")
    return report, report_html


def _render_html(payload: dict[str, object]) -> str:
    model = payload["model"]
    test_report = payload["test_report"]
    audit = payload["export_audit"]
    artifacts = payload["artifacts"]
    assert isinstance(model, dict)
    assert isinstance(test_report, dict)
    assert isinstance(audit, dict)
    assert isinstance(artifacts, list)

    failures = test_report.get("failures", [])
    warnings = test_report.get("warnings", [])
    metrics = test_report.get("metrics", [])
    status = "Passed" if test_report.get("passed") else "Failed"
    artifact_cards = (
        "".join(_artifact_html(item) for item in artifacts if isinstance(item, dict))
        or "<p>No visual or file evidence was attached.</p>"
    )
    metric_rows = (
        "".join(
            "<tr>"
            f"<td>{_escape(item.get('name'))}</td>"
            f"<td>{_escape(item.get('value'))}</td>"
            f"<td>{_escape(item.get('unit'))}</td>"
            f"<td>{_escape(_range_text(item))}</td>"
            "</tr>"
            for item in metrics
            if isinstance(item, dict)
        )
        or '<tr><td colspan="4">No metrics were recorded.</td></tr>'
    )
    failure_items = _list_html(failures, failure=True)
    warning_items = _list_html(warnings)
    audit_rows = "".join(
        f"<tr><td>{_escape(key.replace('_', ' '))}</td><td>{_escape(value)}</td></tr>"
        for key, value in audit.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(model.get("name"))} QA report</title>
<style>
body {{ margin: 0; font: 15px/1.45 system-ui, sans-serif; color: #20242a; background: #f4f5f7; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
h1, h2 {{ margin: 0 0 16px; }}
h2 {{ margin-top: 30px; }}
.summary, .card {{ background: white; border: 1px solid #dfe2e7; border-radius: 10px; padding: 18px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }}
.summary div {{ padding: 8px; }}
.label {{ color: #606771; font-size: 12px; text-transform: uppercase; }}
.value {{ display: block; font-size: 20px; margin-top: 4px; }}
.artifacts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.card img {{ display: block; width: 100%; height: auto; border: 1px solid #e2e4e8; background: #fafafa; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ border: 1px solid #dfe2e7; padding: 9px; text-align: left; vertical-align: top; }}
th {{ background: #eef0f3; }}
code {{ font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<main>
<h1>{_escape(model.get("name"))} QA report</h1>
<section class="summary">
<div><span class="label">Status</span><span class="value">{status}</span></div>
<div><span class="label">Parts</span><span class="value">{_escape(model.get("parts"))}</span></div>
<div><span class="label">Shapes</span><span class="value">{_escape(model.get("shapes"))}</span></div>
<div><span class="label">Articulations</span><span class="value">{_escape(model.get("articulations"))}</span></div>
</section>
<h2>Evidence</h2>
<section class="artifacts">{artifact_cards}</section>
<h2>Checks</h2>
<section class="card"><h3>Failures</h3>{failure_items}<h3>Warnings</h3>{warning_items}</section>
<h2>Metrics</h2>
<table><thead><tr><th>Name</th><th>Value</th><th>Unit</th><th>Expected range</th></tr></thead>
<tbody>{metric_rows}</tbody></table>
<h2>USDZ audit</h2>
<table><tbody>{audit_rows}</tbody></table>
</main>
</body>
</html>
"""


def _artifact_html(item: dict[str, object]) -> str:
    title = _escape(item.get("name"))
    caption = _escape(item.get("caption"))
    report_path = _escape(item.get("report_path"))
    if item.get("kind") == "image":
        body = f'<img src="{report_path}" alt="{title}">'
    else:
        body = f'<p><a href="{report_path}">Open attached file</a></p>'
    return f'<article class="card"><h3>{title}</h3>{body}<p>{caption}</p></article>'


def _list_html(values: object, *, failure: bool = False) -> str:
    if not isinstance(values, list | tuple) or not values:
        return "<p>None.</p>"
    items: list[str] = []
    for value in values:
        if failure and isinstance(value, dict):
            text = f"{value.get('name')}: {value.get('details')}"
        else:
            text = str(value)
        items.append(f"<li>{_escape(text)}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _range_text(metric: dict[str, object]) -> str:
    minimum = metric.get("minimum")
    maximum = metric.get("maximum")
    if minimum is None and maximum is None:
        return ""
    return f"{minimum if minimum is not None else 'none'} to {maximum if maximum is not None else 'none'}"


def _unique_filename(name: str, used: set[str]) -> str:
    path = Path(name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "artifact"
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", path.suffix.lower())
    candidate = f"{stem}{suffix}"
    index = 2
    while candidate in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


__all__ = ["write_qa_report"]

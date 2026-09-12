"""Render a portable, offline research report with no browser dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _finite_values(value: Any) -> Any:
    """Represent unavailable numeric results as JSON null, never invalid NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_values(item) for item in value]
    return value


def render_report(report: dict, output: Path) -> Path:
    """Write a self-contained HTML dashboard and return its destination.

    All report content enters through a JSON data block. Escaping HTML-sensitive
    characters prevents a source label or note from closing that block; the
    browser renders untrusted strings with textContent, never HTML interpolation.
    Nonfinite floating-point values become null and display as unavailable.
    """
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    payload = json.dumps(_finite_values(report), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                               ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(character, escaped)
    marker = "__ECON_REPORT_DATA__"
    if template.count(marker) != 1:
        raise RuntimeError("Report template must contain exactly one data marker")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace(marker, payload), encoding="utf-8")
    return destination

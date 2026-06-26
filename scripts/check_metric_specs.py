#!/usr/bin/env python3
"""
scripts/check_metric_specs.py

CI enforcement: every metric implementation must have a corresponding
entry in METRIC_STANDARDS.md with required fields populated.

Paper §3.4 "Metric Governance": "All metrics shipped in stratum-eval
are registered in METRIC_STANDARDS.md. This script enforces that
registration as a hard CI gate."

Usage:
    python scripts/check_metric_specs.py
    python scripts/check_metric_specs.py --inject-broken   # deliberate failure for CI test

Exit codes:
    0  all metrics registered and valid
    1  one or more violations found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
STANDARDS_PATH = REPO_ROOT / "METRIC_STANDARDS.md"
METRICS_GLOB = "stratum/layers/metrics/**/*.py"

# Every metric file must contain these marker strings.
# The value is the human-readable name used in error messages.
REQUIRED_MARKERS = {
    "METRIC_NAME":    "metric name declaration (# METRIC_NAME: ...)",
    "METRIC_LAYER":   "layer assignment (# METRIC_LAYER: ...)",
    "PAPER_REF":      "paper section reference (# PAPER_REF: ...)",
    "THRESHOLD_CITE": "threshold source citation (# THRESHOLD_CITE: ...)",
}

# Required column headers in METRIC_STANDARDS.md table.
REQUIRED_MD_COLUMNS = {
    "Metric",
    "Layer",
    "Paper §",
    "Threshold Source",
    "Status",
}

# Injected broken metric for --inject-broken CI self-test.
_BROKEN_METRIC_CONTENT = '''
"""Deliberately broken metric — missing required markers. For CI self-test."""
def compute(y_true, y_score):
    return 0.0
'''


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_metric_file(path: Path) -> dict[str, str | None]:
    """Extract marker values from a metric source file."""
    text = path.read_text(encoding="utf-8")
    result: dict[str, str | None] = {}
    for marker in REQUIRED_MARKERS:
        pattern = rf"#\s*{marker}\s*:\s*(.+)"
        m = re.search(pattern, text)
        result[marker] = m.group(1).strip() if m else None
    return result


def parse_standards_table(path: Path) -> set[str]:
    """
    Return the set of metric names registered in METRIC_STANDARDS.md.
    Expects a Markdown table with a 'Metric' column.
    """
    text = path.read_text(encoding="utf-8")
    registered: set[str] = set()
    in_table = False
    header_cols: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if in_table:
                break   # left the table
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]

        if not in_table:
            # First pipe row is the header
            header_cols = cells
            if not REQUIRED_MD_COLUMNS.issubset(set(header_cols)):
                missing = REQUIRED_MD_COLUMNS - set(header_cols)
                raise ValueError(
                    f"METRIC_STANDARDS.md table missing required columns: {missing}"
                )
            in_table = True
            continue

        if set(cells) == {"---", "---", "---", "---", "---"} or all(
            c.startswith("-") for c in cells if c
        ):
            continue   # separator row

        if "Metric" in header_cols:
            idx = header_cols.index("Metric")
            if idx < len(cells) and cells[idx]:
                registered.add(cells[idx])

    return registered


def check_standards_file_headers(path: Path) -> list[str]:
    """Verify METRIC_STANDARDS.md has all required column headers."""
    text = path.read_text(encoding="utf-8")
    errors = []
    for col in REQUIRED_MD_COLUMNS:
        if col not in text:
            errors.append(f"METRIC_STANDARDS.md missing required column header: '{col}'")
    return errors


# ── Main check ───────────────────────────────────────────────────────────────

def run_checks(inject_broken: bool = False) -> list[str]:
    """
    Run all metric spec checks. Returns a list of violation strings.
    Empty list means all checks pass.
    """
    violations: list[str] = []

    # 0. Standards file must exist
    if not STANDARDS_PATH.exists():
        return [f"METRIC_STANDARDS.md not found at {STANDARDS_PATH}"]

    # 1. Standards file structure
    violations.extend(check_standards_file_headers(STANDARDS_PATH))

    # 2. Parse registered metric names from standards table
    try:
        registered_metrics = parse_standards_table(STANDARDS_PATH)
    except ValueError as e:
        return [str(e)]

    # 3. Check every metric implementation file
    metric_files = list(REPO_ROOT.glob(METRICS_GLOB))

    if inject_broken:
        # Write a deliberately broken temp file for CI self-test
        broken_path = REPO_ROOT / "stratum/layers/metrics/_broken_test_metric.py"
        broken_path.write_text(_BROKEN_METRIC_CONTENT, encoding="utf-8")
        metric_files.append(broken_path)

    for fpath in metric_files:
        if fpath.name.startswith("__"):
            continue

        markers = parse_metric_file(fpath)
        rel = fpath.relative_to(REPO_ROOT)

        # 3a. Required markers present
        for marker, description in REQUIRED_MARKERS.items():
            if markers[marker] is None:
                violations.append(
                    f"{rel}: missing {description}"
                )

        # 3b. Metric registered in METRIC_STANDARDS.md
        metric_name = markers.get("METRIC_NAME")
        if metric_name and metric_name not in registered_metrics:
            violations.append(
                f"{rel}: metric '{metric_name}' not found in METRIC_STANDARDS.md. "
                f"Add a row to the standards table before merging."
            )

    # 4. Cleanup injected broken file
    if inject_broken:
        broken_path = REPO_ROOT / "stratum/layers/metrics/_broken_test_metric.py"
        if broken_path.exists():
            broken_path.unlink()

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inject-broken",
        action="store_true",
        help="Inject a deliberately broken metric to verify the checker catches it. "
             "CI self-test only.",
    )
    args = parser.parse_args()

    violations = run_checks(inject_broken=args.inject_broken)

    if not violations:
        print("✓ All metric specs pass METRIC_STANDARDS.md enforcement.")
        sys.exit(0)
    else:
        print(f"✗ {len(violations)} metric spec violation(s) found:\n")
        for v in violations:
            print(f"  • {v}")
        print(
            "\nAll metrics must be registered in METRIC_STANDARDS.md with "
            "Metric, Layer, Paper §, Threshold Source, and Status columns "
            "before the CI gate will pass. See §3.4 of the paper."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

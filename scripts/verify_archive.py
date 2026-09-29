#!/usr/bin/env python3
"""Offline archive check: integrity, report media links, and outcome semantics."""
from html.parser import HTMLParser
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "hybrid_rollout/robodojo/reports"
REPORT_FILES = (
    REPORTS / "gpt6_layout0_20260919/gpt6_test_report.html",
    REPORTS / "pi05_layout0_20260919/pi05_test_report.html",
    REPORTS / "hybrid_gpt_pi_layout0_20260919/hybrid_gpt_pi_test_report.html",
)
RESULTS = {
    "GPT fixed 5": ("gpt6_layout0_20260919/run_20260919_120207/result.json", 700, False),
    "GPT fixed 50": ("gpt6_layout0_h50_20260919/run_20260919_235529/result.json", 700, False),
    "GPT adaptive": ("gpt6_layout0_adaptive50_v3_20260920/run_20260920_083346/result.json", 700, False),
    "hybrid first": ("hybrid_gpt_pi_layout0_20260919/run_20260919_152000/result.json", 502, True),
    "hybrid repeat": ("hybrid_gpt_pi_layout0_20260919/run_20260919_212200/result.json", 700, False),
}


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag not in {"a", "img", "video", "source"}:
            return
        attributes = dict(attrs)
        value = attributes.get("href" if tag == "a" else "src")
        if value:
            self.links.append(value)


def check():
    errors = []
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        errors.append("MANIFEST.sha256 missing")
    else:
        checked = 0
        for line in manifest.read_text().splitlines():
            expected, sep, name = line.partition("  ")
            path = ROOT / name
            if not sep or not path.is_file():
                errors.append(f"manifest entry missing: {name}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"hash mismatch: {name}")
            checked += 1
        print(f"Manifest: {checked} files checked")
    for report in REPORT_FILES:
        if not report.is_file():
            errors.append(f"report missing: {report.relative_to(ROOT)}")
            continue
        parser = LocalLinks()
        parser.feed(report.read_text())
        checked = 0
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                if url.hostname in {"127.0.0.1", "localhost"}:
                    errors.append(f"local machine link: {report.name}: {link}")
                continue
            target = (report.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append(f"broken report link: {report.name}: {link}")
            checked += 1
        print(f"{report.name}: {checked} local links checked")
    for name, (relative, steps, success) in RESULTS.items():
        path = REPORTS / relative
        try:
            data = json.loads(path.read_text())
            if data["step_id"] != steps or data["success"] is not success:
                errors.append(f"unexpected result: {name}")
            print(f"{name}: {data['step_id']} steps; native_success={data['success']}; "
                  f"wall={data['wall_seconds']:.2f}s")
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"invalid result {name}: {error}")
    pi = sorted((REPORTS / "pi05_layout0_20260919/artifacts").glob("*summary.json"))
    if len(pi) != 4:
        errors.append(f"expected four π summaries, found {len(pi)}")
    print(f"π-only: {len(pi)} run summaries")
    for filename, inline in (("robodojo-decision-report-20260929.html", False),
                             ("robodojo-decision-report-20260929-offline.html", True)):
        report = ROOT / "docs" / filename
        if not report.is_file():
            errors.append(f"decision report missing: {filename}")
            continue
        source = report.read_text()
        marker = '<script id="report-data" type="application/json">'
        try:
            start = source.index(marker) + len(marker)
            data = json.loads(source[start:source.index("</script>", start)])
            episodes = data["episodes"]
            if len(episodes) != 9 or sum(len(e["decisions"]) for e in episodes) != 357:
                errors.append(f"decision count mismatch: {filename}")
            for episode in episodes:
                video = episode["video"]
                if inline:
                    raw = base64.b64decode(video["data"].split(",", 1)[1])
                else:
                    raw = (report.parent / video["data"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != video["sha256"]:
                    errors.append(f"decision video mismatch: {filename}: {episode['name']}")
                if episode["decisions"][-1]["end"] != episode["steps"]:
                    errors.append(f"decision step mismatch: {filename}: {episode['name']}")
            print(f"{filename}: {len(episodes)} episodes, "
                  f"{sum(len(e['decisions']) for e in episodes)} decisions, "
                  f"{len(episodes)} video hashes checked")
        except (ValueError, KeyError, IndexError, OSError) as error:
            errors.append(f"invalid decision report {filename}: {error}")
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(check())

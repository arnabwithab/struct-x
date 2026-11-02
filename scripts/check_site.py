"""Link + asset check for site/index.html (stdlib only).

Verifies: required sections present, local href/src targets exist,
<img> files are non-empty PNGs, external links are well-formed https URLs.
Usage: python3 scripts/check_site.py [--dir site]
"""

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_IDS = [
    "question",
    "method",
    "metrics",
    "verdict",
    "cost",
    "survey",
    "limitations",
    "links",
]

REQUIRED_IMGS = [
    "figs/o5_forest.png",
    "figs/matched_rmse.png",
    "figs/pretrained_vs_control.png",
]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

EXTERNAL_RE = re.compile(r"^https://[A-Za-z0-9][^ \t\"'<>]*$")


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.imgs = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.imgs.append(attrs.get("src", ""))

    def error(self, message):
        self.errors.append(message)


def main():
    site_dir = Path(sys.argv[sys.argv.index("--dir") + 1]) if "--dir" in sys.argv else Path("site")
    index = site_dir / "index.html"
    failures = []
    if not index.is_file():
        print(f"FAIL: {index} missing")
        return 1

    parser = LinkCollector()
    try:
        parser.feed(index.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- report parse failure as check failure
        print(f"FAIL: index.html does not parse: {exc}")
        return 1
    if parser.errors:
        failures.append(f"HTML parse errors: {parser.errors}")

    for rid in REQUIRED_IDS:
        if rid not in parser.ids:
            failures.append(f"missing required section id=#{rid}")

    for img in REQUIRED_IMGS:
        if img not in parser.imgs:
            failures.append(f"required figure not referenced: {img}")

    for src in parser.imgs:
        if not src or src.startswith(("http://", "https://", "data:")):
            failures.append(f"img src must be a relative local path: {src!r}")
            continue
        target = (site_dir / src).resolve()
        base = site_dir.resolve()
        if base not in target.parents and target != base:
            failures.append(f"img escapes site dir: {src!r}")
            continue
        if not target.is_file():
            failures.append(f"img file missing: {src}")
        elif target.stat().st_size == 0:
            failures.append(f"img file empty: {src}")
        elif not target.read_bytes()[:8].startswith(PNG_MAGIC):
            failures.append(f"img is not a PNG: {src}")

    for href in parser.links:
        if not href:
            failures.append("empty href found")
        elif href.startswith("#"):
            anchor = href[1:]
            if anchor and anchor not in parser.ids:
                failures.append(f"anchor #{anchor} has no matching id")
        elif href.startswith(("http://", "https://")):
            if not EXTERNAL_RE.match(href):
                failures.append(f"malformed external URL: {href}")
        else:  # relative local link
            target = site_dir / href
            if not target.exists():
                failures.append(f"local link target missing: {href}")

    if failures:
        print(f"FAIL: {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"OK: index.html parses; {len(REQUIRED_IDS)} sections, "
        f"{len(parser.imgs)} imgs present, {len(parser.links)} links checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

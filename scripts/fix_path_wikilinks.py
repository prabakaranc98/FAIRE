"""One-off: resolve already-on-disk path-style `[[a/b/c]]` wikilinks.

The new nodes.py resolver handles them for NEW pages, but ~14 pages already
on disk have raw [[curriculum/...track.../slug]] text. This script walks
every page, extracts the last segment of each path-style wikilink, looks it
up in the concept index, and rewrites either to a real markdown link or to
the italic-placeholder format.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "docs" / "curriculum" / "core"


def _index_concepts() -> dict[str, Path]:
    """{slug: absolute path to the concept page}."""
    out: dict[str, Path] = {}
    for p in CORE.rglob("concepts/*.md"):
        if p.name == "index.md":
            continue
        out[p.stem] = p
    return out


def main():
    pages = _index_concepts()
    if not pages:
        print("no concepts indexed")
        return 1

    pattern = re.compile(r"\[\[([a-zA-Z0-9][a-zA-Z0-9\-/.]*?/[a-zA-Z0-9][a-zA-Z0-9\-/.]*)\]\]")
    fixed_files = 0
    fixed_links = 0
    for page in CORE.rglob("*.md"):
        if page.name == "index.md" and "/arcs/" not in str(page):
            # track-level index.md is owned by the rebuilder; don't touch
            continue
        text = page.read_text(encoding="utf-8", errors="ignore")
        new_text = text
        page_changes = 0
        for m in pattern.finditer(text):
            raw = m.group(1)
            slug = raw.rsplit("/", 1)[-1].removesuffix(".md").removesuffix(".html").lower()
            target = pages.get(slug)
            if target:
                try:
                    rel = target.resolve().relative_to(page.parent.resolve())
                    rel_str = str(rel)
                except (ValueError, RuntimeError):
                    # Fallback to os.path.relpath
                    import os
                    rel_str = os.path.relpath(str(target), str(page.parent))
                replacement = f"[{slug.replace('-', ' ').title()}]({rel_str})"
            else:
                human = slug.replace("-", " ")
                replacement = f"*{human}* <!-- [[{slug}]] -->"
            new_text = new_text.replace(m.group(0), replacement, 1)
            page_changes += 1
        if page_changes:
            page.write_text(new_text, encoding="utf-8")
            fixed_files += 1
            fixed_links += page_changes
            print(f"  {page.relative_to(ROOT)}: {page_changes} link(s) fixed")
    print(f"\nfixed {fixed_links} path-style wikilinks across {fixed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Insert blank lines around display-math `\[ ... \]` blocks so arithmatex renders them.

pymdownx.arithmatex (generic mode) only recognises display-math blocks when the
opening `\[` line is preceded by a blank line and the closing `\]` is followed
by one. Pages where the writer puts `\[` right after a colon ('rewrites this as:\n\[')
render as raw \[ … \] text on the live site — kills reader trust on math-heavy pages.

This script walks every curriculum/system page and normalises blank-line placement
around display-math blocks. Idempotent — running twice produces no extra changes.
"""
from __future__ import annotations
import re
from pathlib import Path


def fix(text: str) -> str:
    lines = text.split('\n')
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Before opening \[ — ensure blank line before
        if stripped == '\\[':
            if out and out[-1].strip() != '':
                out.append('')
            out.append(line)
            continue
        # After closing \] — ensure blank line after
        if stripped == '\\]':
            out.append(line)
            if i + 1 < len(lines) and lines[i + 1].strip() != '':
                out.append('')
            continue
        out.append(line)
    return '\n'.join(out)


def main():
    roots = [Path('docs/curriculum/core'), Path('docs/curriculum/index.md')]
    fixed = 0
    scanned = 0
    for root in roots:
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob('*.md') if p.name != 'CNAME']
        for p in files:
            scanned += 1
            orig = p.read_text(encoding='utf-8', errors='ignore')
            new = fix(orig)
            if new != orig:
                p.write_text(new, encoding='utf-8')
                fixed += 1
                print(f"  fixed: {p}")
    print(f"\n  {fixed}/{scanned} files needed blank-line normalisation around \\[ \\]")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

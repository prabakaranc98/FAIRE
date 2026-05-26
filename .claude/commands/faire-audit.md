Run a structural audit of all Frontier Wiki pages and report issues.

Arguments: $ARGUMENTS (optional: "page-slug" to audit one specific page)

If a specific page is given, read that file and report its quality against the SCHEMA:
- Does it have all required sections? (reader table, core concepts, math, MVB, connected topics)
- Are all math equations in \[...\] / \(...\) format (not $...$ dollar signs)?
- Are all internal links real (not [[wikilinks]])?
- Does it have frontmatter with title, track, has_mvb, updated?
- Does it link to any relevant arcs?

If no argument, run the server audit:
```bash
curl -s http://localhost:8765/audit 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Server not running."
```

Then scan locally for known issues:
```bash
cd /Users/pchandran/Desktop/Experiments/FAIRE
echo "=== Remaining wikilinks (broken) ===" && grep -rn '\[\[' docs/curriculum/ | grep -v "index.md" | head -20
echo "=== Dollar-sign math (won't render) ===" && grep -rn '\$\$' docs/curriculum/ | grep -v "index.md" | head -10  
echo "=== Pages missing MVB ===" && grep -rL "Minimum Valuable Build" docs/curriculum/**/*.md 2>/dev/null | grep -v "index.md" | head -10
```

Summarize the top issues and which pages need the most attention.

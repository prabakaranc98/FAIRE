Rebuild one or more Frontier Wiki pages through the full pipeline (research → plan → write → review → commit).

Arguments: $ARGUMENTS (format: "topic track" or "topic" or "all-flagged")

If "all-flagged": rebuild all pages currently flagged in runs/runs.jsonl
If a topic is given: rebuild that specific page
If no args: show what's currently in the sprint queue

For a specific topic:
```bash
cd agents && uv run python generate.py generate $TOPIC --track $TRACK --depth applied frontier
```

For all-flagged:
```bash
cd agents && uv run python generate.py rebuild --depth applied frontier
```

If the server is running, prefer using POST /trigger to run a full cycle, or POST /generate for a single page.

Show the result including: confidence score, approved/flagged status, and output file path.
Note: pages with confidence ≥ 0.7 are auto-committed to git.

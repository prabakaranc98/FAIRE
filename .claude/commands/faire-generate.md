Generate or improve a specific Frontier Wiki page.

Arguments: $ARGUMENTS (format: "topic track" or just "topic")

Parse the arguments: the first word is the topic slug, the second (if provided) is the track.
If no track is provided, search docs/curriculum/ to find which track the topic is in.

Then run:
```bash
curl -s -X POST "http://localhost:8765/generate?topic=TOPIC&track=TRACK&page_type=core-concept&depth_emphasis=applied,frontier" 2>/dev/null
```

If the server is not running, instead run the CLI directly:
```bash
cd agents && uv run python generate.py generate TOPIC --track TRACK --depth applied frontier
```

Show the result and tell the user where the output page will be written.

Start the Frontier Wiki autonomous agent server.

Arguments: $ARGUMENTS (optional flags: --dry-run, --run-now, --interval 4)

Check if already running:
```bash
curl -s http://localhost:8765/ 2>/dev/null | head -5
```

If not running, start it:
```bash
./start.sh $ARGUMENTS &
sleep 3
curl -s http://localhost:8765/ 2>/dev/null | head -20
```

The server:
- Runs on http://localhost:8765
- Auto-bootstraps the supervisor on startup (updates sprint queue)
- Runs a full cycle every 48h (supervisor → audit → sprint → changelog)
- Restarts automatically on crash (via start.sh)

Key flags:
- --dry-run: simulate, no disk writes or git commits
- --run-now: run one full cycle immediately on startup
- --interval 4: 4h cycle instead of 48h (for testing)

After starting, show the dashboard at GET /.

Check the Frontier Wiki agent server status.

Run the following and show the results clearly:

```bash
curl -s http://localhost:8765/ 2>/dev/null || echo "Server not running. Start with: ./start.sh --dry-run"
```

If the server is running, also show:
```bash
curl -s http://localhost:8765/budget 2>/dev/null
```

Summarize: is the server running, what's the wiki coverage, how many pages approved, what's the budget status, what's next in the sprint queue.

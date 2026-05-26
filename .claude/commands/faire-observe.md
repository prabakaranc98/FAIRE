Show the Frontier Wiki observer control dashboard — the system's real-time quality signals.

Run:
```bash
curl -s http://localhost:8765/observer 2>/dev/null || cat docs/system/observer.md 2>/dev/null || echo "No observer data yet. Start server and run a cycle."
```

Then also fetch metrics JSON for key numbers:
```bash
curl -s http://localhost:8765/metrics 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f'Observed: {d.get(\"observed_at\", \"?\")}')
    print(f'Coverage: {d.get(\"coverage_pct\", 0):.1%} ({d.get(\"generated_pages\", 0)}/{d.get(\"total_pages\", 0)} pages)')
    print(f'Approved: {d.get(\"approved_pages\", 0)} | Avg conf: {d.get(\"avg_confidence\", 0):.2f}')
    b = d.get('budget', {})
    r = b.get('remaining_usd')
    print(f'Budget: {\"$\" + str(round(r, 2)) if r is not None else \"unlimited\"} ({b.get(\"mode\", \"?\")} mode)')
    e = d.get('error_signals', {})
    print(f'Error signals: coverage_deficit={e.get(\"coverage_deficit\", 0):.1%}, quality_deficit={e.get(\"quality_deficit\", 0):.2f}')
except: pass
" 2>/dev/null
```

Interpret the control signals: what needs the most attention right now?

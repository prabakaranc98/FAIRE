#!/usr/bin/env bash
# Frontier Wiki — autonomous agent server process supervisor
#
# Starts the server and restarts it automatically on crash.
# Think of this like a systemd unit or a plant controller:
#   - Runs indefinitely
#   - Restarts on failure
#   - Logs to agents/logs/server.log
#   - Ctrl+C stops cleanly
#
# Usage:
#   ./start.sh                  # 48h cycle, local only
#   ./start.sh --run-now        # run one full cycle immediately, then schedule
#   ./start.sh --dry-run        # simulate — no disk writes, no git commits
#   ./start.sh --interval 4     # 4h cycle (for testing)
#   ./start.sh --run-now --dry-run

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$SCRIPT_DIR/agents"
LOG_DIR="$AGENTS_DIR/logs"

mkdir -p "$LOG_DIR"

ARGS="$@"
RESTART_COUNT=0
STARTED_AT="$(date -u +"%Y-%m-%d %H:%M UTC")"

# Clean shutdown on Ctrl+C or SIGTERM
trap 'echo ""; echo "[$(date -u +"%Y-%m-%d %H:%M UTC")] Frontier Wiki server stopped (${RESTART_COUNT} restarts since ${STARTED_AT})."; exit 0' INT TERM

echo "════════════════════════════════════════════"
echo "  Frontier Wiki — Autonomous Agent Server"
echo "════════════════════════════════════════════"
echo "  Started:  $STARTED_AT"
echo "  Logs:     $LOG_DIR/server.log"
echo "  Stop:     Ctrl+C"
echo "  Args:     ${ARGS:-<none>}"
echo "════════════════════════════════════════════"
echo ""

while true; do
    CYCLE_START="$(date -u +"%Y-%m-%d %H:%M UTC")"
    echo "[$CYCLE_START] Starting server (restart #${RESTART_COUNT})..."

    # Rotate log if over 10MB
    LOG_FILE="$LOG_DIR/server.log"
    if [ -f "$LOG_FILE" ] && [ "$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)" -gt 10485760 ]; then
        mv "$LOG_FILE" "$LOG_DIR/server.$(date -u +%Y%m%d%H%M%S).log"
        echo "[$CYCLE_START] Log rotated."
    fi

    cd "$AGENTS_DIR"
    # Run the server; capture exit code
    uv run python server.py $ARGS 2>&1 | tee -a "$LOG_FILE" || true
    EXIT_CODE=${PIPESTATUS[0]}

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[$(date -u +"%Y-%m-%d %H:%M UTC")] Server exited (code=$EXIT_CODE, total restarts=$RESTART_COUNT). Restarting in 10s..."
    echo "  (Press Ctrl+C to stop permanently)"
    sleep 10
done

#!/usr/bin/env python
"""Frontier Wiki — agent CLI entry point.

Run from the agents/ directory:
  uv run python generate.py --help
  uv run python generate.py generate --topic diffusion-models --track 02-generative-modeling
  uv run python generate.py generate --track 06-reinforcement-learning --all-stubs
  uv run python generate.py generate --topic rlhf --track 06-reinforcement-learning --mvb-only
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from frontier_agents.cli import cli

if __name__ == "__main__":
    cli()

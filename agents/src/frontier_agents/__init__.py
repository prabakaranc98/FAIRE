"""Frontier Wiki — editorial agent system.

Four agent types:
- Editorial agent   (claude-opus-4-7)   — researches + writes topic pages
- Pedagogical agent (claude-opus-4-7)   — curates reading lists, arc sequences, SotA
- MVB recipe agent  (claude-opus-4-7)   — generates Minimum Valuable Build sections (on-demand)
- Reviewer agent    (claude-haiku-4-5)  — schema compliance + source policy checks

Entry point: agents/generate.py
"""

__version__ = "0.1.0"

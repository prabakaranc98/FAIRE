"""Shared fixtures for all agent tests."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

# Load .env — try agents/.env first, then root FAIRE/.env
_AGENTS_ENV = Path(__file__).parent.parent / ".env"        # agents/.env
_ROOT_ENV   = Path(__file__).parent.parent.parent / ".env" # FAIRE/.env (root)
for _env_path in (_AGENTS_ENV, _ROOT_ENV):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
        break

from frontier_agents.state import WikiPageState


@pytest.fixture
def base_state() -> WikiPageState:
    return {
        "topic": "diffusion-models",
        "track": "02-generative-modeling",
        "depth": "all",
        "mode": "full",
        "revision_count": 0,
        "persona": {
            "track": "02-generative-modeling",
            "domain": "Generative modeling in deep learning",
            "expertise": "diffusion models, score matching, flow matching",
            "seminal_authors": ["Ho", "Song", "Lipman"],
            "key_venues": ["NeurIPS", "ICML", "ICLR"],
            "mvb_focus": "minimal diffusion training loop",
            "search_seeds": ["diffusion models arxiv"],
            "production_examples": ["Stable Diffusion", "DALL-E 3"],
        },
        "existing_stub": "",
        "research_results": [],
        "draft": "",
        "approved": False,
        "committed": False,
    }


@pytest.fixture
def stub_state(base_state) -> WikiPageState:
    """State with existing stub content."""
    return {
        **base_state,
        "existing_stub": "---\ntitle: Diffusion Models\ntrack: 02-generative-modeling\nupdated: 2026-05-25\n---\n# Diffusion Models\n> 🚧 Agent-generated content pending.\n",
    }


@pytest.fixture
def draft_state(base_state) -> WikiPageState:
    """State with a draft ready for review."""
    return {
        **base_state,
        "draft": """---
title: Diffusion Models
track: 02-generative-modeling
has_mvb: true
updated: 2026-05-25
---

# Diffusion Models
> **TL;DR:** Score-based generative models that learn to reverse a noise process.

## For your reader type
| I am... | Start here | Goal |
|---|---|---|
| Applied practitioner | Key algorithms | Build a diffusion model |

## What it is
Diffusion models are generative models.

## Why it matters at the frontier
Used in DALL-E 3, Stable Diffusion, and many frontier systems.

## Core concepts
- Score function
- Forward process

## Mathematical foundations
The ELBO objective: $$L = \\mathbb{E}[...]$$

## Key algorithms
DDPM, DDIM, Score Matching.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [DDPM](https://arxiv.org/abs/2006.11239) | 2020 | Ho et al. | Baseline |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Score Matching](https://arxiv.org/abs/1907.05600) | 2019 | Foundational |

## Current SotA
FLUX.1 and SD3 represent current state of the art.

## What's happening now
**Research:** Active work on flow matching.
**Engineering & Systems:** Optimized inference pipelines.
**Open problems:** Fast sampling, controllability.

## In production
- **Stability AI:** Stable Diffusion XL at scale.

## Minimum Valuable Build
**What you're building:** CIFAR-10 class-conditional generator.
**Stack:** diffusers, PyTorch
**The recipe:**
1. Load dataset
2. Train

## Code & implementations
- [diffusers](https://github.com/huggingface/diffusers)

## Connected topics
- [[flow-matching]]

## Further reading
- [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)
""",
    }


@pytest.fixture
def tmp_docs_dir(tmp_path):
    """Temporary docs directory for write/commit tests."""
    docs = tmp_path / "docs" / "curriculum" / "02-generative-modeling"
    docs.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def tmp_arcs_json(tmp_path):
    """Temporary arcs.json for update tests."""
    data = {
        "arcs": [
            {
                "id": "arc-02-generative",
                "nodes": [
                    {"id": "diffusion-models", "concept": "Diffusion Models", "status": "stub"}
                ],
            }
        ]
    }
    f = tmp_path / "arcs.json"
    f.write_text(json.dumps(data, indent=2))
    return f


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM that returns a fixed response."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mock LLM response content for testing."
    llm.invoke.return_value = response
    return llm

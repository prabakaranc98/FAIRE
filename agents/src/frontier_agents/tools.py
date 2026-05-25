"""Tools used by the editorial agent system.

All tools are pure functions that take typed inputs and return typed outputs.
They are called from nodes.py, not directly from the LangGraph graph.

Source policy (enforced at search time):
  - Default: arxiv.org, *.edu, huggingface.co, official library docs
  - "In production" section only: official engineering blogs allowed
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


APPROVED_DOMAINS = [
    "arxiv.org",
    "huggingface.co",
    ".edu",
    "pytorch.org",
    "jax.readthedocs.io",
    "openai.com/research",
    "anthropic.com",
    "deepmind.google",
]

APPROVED_ENGINEERING_BLOGS = [
    "engineering.linkedin.com",
    "ai.meta.com/research",
    "developer.nvidia.com/blog",
    "research.google",
    "openai.com/research",
    "blog.google",
    "aws.amazon.com/blogs/machine-learning",
    "techblog.netflix.com",
    "databricks.com/blog",
]


def exa_search(
    query: str,
    num_results: int = 8,
    section: str = "default",  # "default" | "in_production"
) -> list[dict]:
    """Search Exa API for relevant sources, filtered by domain policy.

    Args:
        query: search query
        num_results: number of results to fetch
        section: "default" for arXiv/edu/HF, "in_production" also allows engineering blogs

    Returns:
        List of {url, title, text, domain, published_date}
    """
    try:
        from exa_py import Exa
    except ImportError:
        raise ImportError("exa-py not installed. Run: uv sync")

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not set in environment. See .env.example")

    from exa_py.api import ContentsOptions

    exa = Exa(api_key=api_key)

    include_domains = list(APPROVED_DOMAINS)
    if section == "in_production":
        include_domains.extend(APPROVED_ENGINEERING_BLOGS)

    results = exa.search(
        query=query,
        num_results=num_results,
        include_domains=include_domains,
        contents=ContentsOptions(text={"max_characters": 2000}),
    )

    return [
        {
            "url": r.url,
            "title": r.title,
            "text": getattr(r, "text", "") or "",
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
        }
        for r in results.results
    ]


def hf_search_models(query: str, limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for models relevant to a topic.

    Returns: List of {model_id, downloads, likes, url, description}
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: uv sync")

    resp = httpx.get(
        "https://huggingface.co/api/models",
        params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
        timeout=10,
    )
    resp.raise_for_status()

    models = []
    for m in resp.json():
        models.append({
            "model_id": m.get("modelId", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "url": f"https://huggingface.co/{m.get('modelId', '')}",
            "description": (m.get("cardData") or {}).get("model-index", [{}]),
        })
    return models


def hf_search_datasets(query: str, limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for datasets relevant to a topic."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: uv sync")

    resp = httpx.get(
        "https://huggingface.co/api/datasets",
        params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
        timeout=10,
    )
    resp.raise_for_status()

    datasets = []
    for d in resp.json():
        datasets.append({
            "dataset_id": d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "url": f"https://huggingface.co/datasets/{d.get('id', '')}",
        })
    return datasets


def read_stub(path: str) -> str:
    """Read an existing wiki page stub or return empty string if not found."""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def write_file(path: str, content: str) -> None:
    """Write content to a wiki page, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def update_arc_json(arc_id: str, node_id: str, data: dict, arcs_json_path: str) -> None:
    """Patch docs/arcs.json with updated node data (e.g., status, doc_path)."""
    p = Path(arcs_json_path)
    arcs_data = json.loads(p.read_text(encoding="utf-8"))

    for arc in arcs_data["arcs"]:
        if arc["id"] == arc_id:
            for node in arc["nodes"]:
                if node["id"] == node_id:
                    node.update(data)
                    break
            break

    p.write_text(json.dumps(arcs_data, indent=2, ensure_ascii=False), encoding="utf-8")


def git_commit(path: str, message: str, docs_dir: str = "../docs") -> bool:
    """Stage and commit a wiki page. Returns True on success."""
    auto_commit = os.getenv("GIT_AUTO_COMMIT", "true").lower() == "true"
    if not auto_commit:
        return False

    try:
        repo_root = Path(docs_dir).parent.resolve()
        subprocess.run(["git", "add", path], cwd=repo_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def load_persona(track: str, personas_dir: str) -> dict:
    """Load per-track editor persona from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml not installed. Run: uv sync")

    persona_path = Path(personas_dir) / f"{track}.yaml"
    if persona_path.exists():
        return yaml.safe_load(persona_path.read_text(encoding="utf-8"))

    # Fallback: generic persona
    return {
        "track": track,
        "domain": track.replace("-", " ").title(),
        "expertise": "machine learning, deep learning",
        "depth_focus": "mathematical foundations + research-level understanding",
        "seminal_authors": [],
        "key_venues": ["NeurIPS", "ICML", "ICLR"],
        "search_seeds": [f"{track} deep learning arxiv"],
    }


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc

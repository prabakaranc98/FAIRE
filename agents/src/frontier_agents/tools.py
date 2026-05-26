"""Tools used by the editorial agent system.

All tools are pure functions that take typed inputs and return typed outputs.
They are called from nodes.py, not directly from the LangGraph graph.

Source policy (enforced at search time):
  - Default: arxiv.org, *.edu, huggingface.co, official library docs
  - "In production" section only: official engineering blogs allowed

Exa search API notes (2026-05-25):
  - Use type="auto" (not "neural" — that's legacy terminology)
  - category="research paper" dramatically improves paper discovery
  - contents.highlights=True is the recommended default for agent workflows
  - useAutoprompt is deprecated and removed
  - search_and_contents is deprecated; use search() with ContentsOptions
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


APPROVED_RESEARCH_DOMAINS = [
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
    "stability.ai/research",
]

# Keep the old name for backwards compatibility in tests
APPROVED_DOMAINS = APPROVED_RESEARCH_DOMAINS


def _get_exa():
    try:
        from exa_py import Exa
    except ImportError:
        raise ImportError("exa-py not installed. Run: uv sync")
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not set in environment. See .env.example")
    return Exa(api_key=api_key)


def exa_search_papers(
    query: str,
    num_results: int | None = None,
    foundational: bool = True,
) -> list[dict]:
    """Search for academic papers on arXiv and .edu.

    Uses neural search with highlights to surface the most relevant excerpts.
    Query should describe the paper's contribution or finding, not just keywords.
    Exa best practice: phrase as "A paper that explains/proves/shows [claim]."

    Args:
        query: Exa neural query — describe the contribution, not keywords
        num_results: defaults to EXA_NUM_RESULTS_PAPERS env var or 8
        foundational: True → all-time; False → 2024+ (for SotA searches)

    Returns:
        List of {url, title, text, highlights, domain, published_date}
    """
    n = num_results or int(os.getenv("EXA_NUM_RESULTS_PAPERS", "8"))
    exa = _get_exa()

    kwargs: dict = dict(
        query=query,
        num_results=n,
        type="neural",            # neural search is best for academic discovery
        category="research paper",
        include_domains=["arxiv.org", ".edu"],
        contents={
            "text": {"max_characters": 3000},
            "highlights": {"num_sentences": 3, "highlights_per_url": 2},
        },
    )
    if not foundational:
        kwargs["start_published_date"] = "2024-01-01"

    results = exa.search(**kwargs)
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "text": getattr(r, "text", "") or "",
            "highlights": getattr(r, "highlights", []) or [],
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
            "source_type": "paper",
        }
        for r in results.results
    ]


def exa_search_sota(query: str, num_results: int | None = None) -> list[dict]:
    """Search for current SotA — papers from 2024+ with benchmark highlights.

    Exa best practice: use keyword search for specific model names/benchmarks,
    neural for finding papers by contribution description.
    Uses highlights (not full text) to surface benchmark numbers efficiently.

    Returns:
        List of {url, title, highlights, domain, published_date}
    """
    n = num_results or int(os.getenv("EXA_NUM_RESULTS_SOTA", "6"))
    exa = _get_exa()

    results = exa.search(
        query=query,
        num_results=n,
        type="auto",              # auto picks neural vs keyword based on query
        category="research paper",
        include_domains=["arxiv.org", ".edu"],
        start_published_date="2024-01-01",
        contents={
            "highlights": {
                "num_sentences": 4,
                "highlights_per_url": 3,
                "query": f"benchmark results, model name, metric numbers: {query}",
            }
        },
    )
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "highlights": getattr(r, "highlights", []) or [],
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
            "source_type": "sota",
        }
        for r in results.results
    ]


def exa_search_production(query: str, num_results: int | None = None) -> list[dict]:
    """Search engineering blogs for production deployments at scale.

    Uses LLM summary targeted to deployment context.
    Exa best practice: use keyword search for exact company/system names,
    neural search for finding deployment patterns.
    Only searches approved engineering blog domains.

    Returns:
        List of {url, title, summary, domain}
    """
    n = num_results or int(os.getenv("EXA_NUM_RESULTS_PROD", "5"))
    exa = _get_exa()

    results = exa.search(
        query=query,
        num_results=n,
        type="keyword",           # keyword for blog post discovery (exact names)
        include_domains=APPROVED_ENGINEERING_BLOGS,
        contents={
            "summary": {
                "query": (
                    f"What system did this company build, at what scale "
                    f"(users/requests/parameters), and how does it work? {query}"
                )
            }
        },
    )
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "summary": getattr(r, "summary", "") or "",
            "domain": _extract_domain(r.url),
            "source_type": "production",
        }
        for r in results.results
    ]


def exa_find_similar(url: str, num_results: int = 5) -> list[dict]:
    """Find papers similar to a known paper using Exa's find-similar endpoint.

    Exa best practice: once you have one good paper, use find_similar to
    discover related work the query might have missed.

    Args:
        url: arXiv or .edu URL of the seed paper
        num_results: number of similar papers to return

    Returns:
        List of {url, title, highlights, domain, published_date}
    """
    exa = _get_exa()
    try:
        results = exa.find_similar(
            url=url,
            num_results=num_results,
            include_domains=["arxiv.org", ".edu"],
            contents={"highlights": {"num_sentences": 2, "highlights_per_url": 1}},
        )
        return [
            {
                "url": r.url,
                "title": r.title or "",
                "highlights": getattr(r, "highlights", []) or [],
                "domain": _extract_domain(r.url),
                "published_date": getattr(r, "published_date", None),
                "source_type": "similar",
            }
            for r in results.results
        ]
    except Exception:
        return []


def exa_search(
    query: str,
    num_results: int = 8,
    section: str = "default",
) -> list[dict]:
    """Legacy unified search — kept for backwards compatibility with existing tests.

    Prefer exa_search_papers / exa_search_sota / exa_search_production in new code.
    """
    include_domains = list(APPROVED_RESEARCH_DOMAINS)
    if section == "in_production":
        include_domains.extend(APPROVED_ENGINEERING_BLOGS)

    exa = _get_exa()

    results = exa.search(
        query=query,
        num_results=num_results,
        type="auto",
        include_domains=include_domains,
        contents={"text": {"max_characters": 2000}},
    )
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "text": getattr(r, "text", "") or "",
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
        }
        for r in results.results
    ]


def hf_search_models(query: str, limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for models relevant to a topic.

    Returns: List of {model_id, downloads, likes, url}
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

    return [
        {
            "model_id": m.get("modelId", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "url": f"https://huggingface.co/{m.get('modelId', '')}",
        }
        for m in resp.json()
    ]


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

    return [
        {
            "dataset_id": d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "url": f"https://huggingface.co/datasets/{d.get('id', '')}",
        }
        for d in resp.json()
    ]


def hf_model_exists(model_id: str) -> dict:
    """Verify a HuggingFace model ID exists. Returns dict with feasibility signals.

    Used by MVB feasibility verifier (mvb-recipe.md quality bar). Catches
    LLM-fabricated model IDs before they ship in MVB recipes.

    Args:
        model_id: e.g. "meta-llama/Llama-3-8B-Instruct"

    Returns:
        {
            "exists": bool,
            "model_id": str (canonical or input),
            "downloads": int (popularity proxy),
            "likes": int,
            "library_name": str | None (e.g. "transformers", "diffusers"),
            "pipeline_tag": str | None (e.g. "text-generation"),
            "private": bool,
            "gated": bool,
            "error": str (set if lookup failed, but exists may still be False)
        }
    """
    try:
        import httpx
    except ImportError:
        return {"exists": False, "model_id": model_id, "error": "httpx not installed"}

    if not model_id or "/" not in model_id:
        return {"exists": False, "model_id": model_id, "error": "malformed model_id (need org/name)"}

    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=8,
            follow_redirects=True,
        )
    except Exception as exc:
        return {"exists": False, "model_id": model_id, "error": f"network: {type(exc).__name__}"}

    if resp.status_code == 401:
        # HF returns 401 for non-existent or private; treat as not-found unless we know it's private.
        return {"exists": False, "model_id": model_id, "error": "not found or private (401)"}
    if resp.status_code == 404:
        return {"exists": False, "model_id": model_id, "error": "not found on HuggingFace"}
    if resp.status_code != 200:
        return {"exists": False, "model_id": model_id, "error": f"HTTP {resp.status_code}"}

    data = resp.json()
    return {
        "exists": True,
        "model_id": data.get("modelId") or data.get("id") or model_id,
        "downloads": int(data.get("downloads", 0)),
        "likes": int(data.get("likes", 0)),
        "library_name": data.get("library_name"),
        "pipeline_tag": data.get("pipeline_tag"),
        "private": bool(data.get("private", False)),
        "gated": bool(data.get("gated", False)),
        "error": "",
    }


def hf_dataset_exists(dataset_id: str) -> dict:
    """Verify a HuggingFace dataset ID exists. Same shape as hf_model_exists."""
    try:
        import httpx
    except ImportError:
        return {"exists": False, "dataset_id": dataset_id, "error": "httpx not installed"}

    if not dataset_id:
        return {"exists": False, "dataset_id": dataset_id, "error": "empty dataset_id"}

    try:
        resp = httpx.get(
            f"https://huggingface.co/api/datasets/{dataset_id}",
            timeout=8,
            follow_redirects=True,
        )
    except Exception as exc:
        return {"exists": False, "dataset_id": dataset_id, "error": f"network: {type(exc).__name__}"}

    if resp.status_code == 401:
        return {"exists": False, "dataset_id": dataset_id, "error": "not found or private (401)"}
    if resp.status_code == 404:
        return {"exists": False, "dataset_id": dataset_id, "error": "not found on HuggingFace"}
    if resp.status_code != 200:
        return {"exists": False, "dataset_id": dataset_id, "error": f"HTTP {resp.status_code}"}

    data = resp.json()
    return {
        "exists": True,
        "dataset_id": data.get("id") or dataset_id,
        "downloads": int(data.get("downloads", 0)),
        "likes": int(data.get("likes", 0)),
        "private": bool(data.get("private", False)),
        "gated": bool(data.get("gated", False)),
        "error": "",
    }


# Rough fit table: model parameter count → minimum reasonable GPU VRAM.
# Source: empirical (fp16 weights + activations + small batch + optimizer
# states roughly 2-4× weight size for training, 1.2× for inference).
# These are conservative defaults; real fit depends on quantization +
# sequence length + batch size, but this catches the obvious cases.
_PARAMS_TO_MIN_VRAM_GB_INFERENCE = [
    (1e8, 4),       # 100M params → 4GB (laptop GPU)
    (1e9, 8),       # 1B params → 8GB
    (3e9, 12),      # 3B params → 12GB (consumer mid)
    (9e9, 24),      # up to ~8B (fp16): 16GB weights + KV cache + activations → fits in 24GB
    (1.5e10, 32),   # 13–15B → 32GB
    (2.5e10, 48),   # up to 24B → 48GB
    (4e10, 80),     # 34–40B → 80GB (A100/H100 single GPU)
    (8e10, 160),    # 70B → 160GB (2×80 or 4×40)
]


def estimate_min_vram_gb(params: int, training: bool = False) -> int:
    """Estimate minimum GPU VRAM (GB) to run a model of `params` parameters.

    Inference assumes fp16 weights + small KV cache. Training multiplies by
    ~3× to account for optimizer states + activations.
    """
    for threshold, vram in _PARAMS_TO_MIN_VRAM_GB_INFERENCE:
        if params <= threshold:
            return vram * (3 if training else 1)
    return 320 * (3 if training else 1)


def verify_mvb_stack(model_id: str = "", dataset_id: str = "",
                     compute: str = "", training: bool = False) -> dict:
    """Verify an MVB stack is feasible. Used by critic-build-nudge.

    Args:
        model_id:  HuggingFace model ID named in the MVB stack
        dataset_id: HuggingFace dataset ID named in the MVB stack
        compute:   plain-text compute target, e.g. "RTX 4070 16GB" or "A100 40GB"
        training:  True if the MVB trains the model; False if inference-only

    Returns:
        {
            "feasible": bool,           # overall verdict
            "model_check": dict,        # from hf_model_exists
            "dataset_check": dict,      # from hf_dataset_exists (if dataset_id)
            "vram_check": dict | None,  # {min_required_gb, named_gb, fits}
            "issues": list[str]         # specific actionable problems
        }
    """
    import re as _re

    issues: list[str] = []
    feasible = True

    model_check = hf_model_exists(model_id) if model_id else {"exists": True}
    if model_id and not model_check["exists"]:
        feasible = False
        issues.append(
            f"Model `{model_id}` does not exist on HuggingFace "
            f"({model_check.get('error', 'unknown')})."
        )

    dataset_check = hf_dataset_exists(dataset_id) if dataset_id else {"exists": True}
    if dataset_id and not dataset_check["exists"]:
        feasible = False
        issues.append(
            f"Dataset `{dataset_id}` does not exist on HuggingFace "
            f"({dataset_check.get('error', 'unknown')})."
        )

    # Compute fit: extract GB from the compute string, look up model size,
    # estimate min VRAM, compare.
    vram_check = None
    if model_check.get("exists") and compute:
        # Extract VRAM from compute string. "RTX 4070 16GB", "A100 80GB",
        # "8GB GPU", "Colab T4 16GB", etc.
        gb_match = _re.search(r"(\d+)\s*GB", compute, _re.IGNORECASE)
        if gb_match:
            named_gb = int(gb_match.group(1))
            params = 0
            # HF API exposes `safetensors.total` (params count) when available.
            # Use the safetensors total if present; otherwise estimate from
            # the model_id name (e.g. "Llama-3-8B" → 8e9).
            # Quick heuristic from model name:
            name_match = _re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_id)
            if name_match:
                params = int(float(name_match.group(1)) * 1e9)
            elif _re.search(r"(\d+)\s*[Mm]", model_id):
                m = _re.search(r"(\d+)\s*[Mm]", model_id)
                params = int(int(m.group(1)) * 1e6)
            if params:
                min_required = estimate_min_vram_gb(params, training=training)
                fits = named_gb >= min_required
                vram_check = {
                    "params": params,
                    "min_required_gb": min_required,
                    "named_gb": named_gb,
                    "training": training,
                    "fits": fits,
                }
                if not fits:
                    feasible = False
                    issues.append(
                        f"Model `{model_id}` (~{params/1e9:.1f}B params) needs "
                        f"≥{min_required}GB VRAM for "
                        f"{'training' if training else 'inference'}; "
                        f"compute target names {named_gb}GB. Build won't fit."
                    )

    return {
        "feasible": feasible,
        "model_check": model_check,
        "dataset_check": dataset_check if dataset_id else None,
        "vram_check": vram_check,
        "issues": issues,
    }


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


def ensure_track_index(track: str, docs_dir: str = "../docs") -> bool:
    """Create a minimal track index.md if one doesn't already exist.

    Returns True if created, False if already existed.
    """
    index_path = Path(docs_dir) / "curriculum" / track / "index.md"
    if index_path.exists():
        return False

    track_title = track.split("-", 1)[-1].replace("-", " ").title()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        f"# {track_title}\n\n"
        f"> This track covers {track_title.lower()}.\n\n"
        f"Pages in this track are generated by the Frontier Wiki editorial agent.\n",
        encoding="utf-8",
    )
    return True


def update_mkdocs_nav(topic: str, track: str, mkdocs_path: str = "../mkdocs.yml") -> bool:
    """Add a newly generated page to mkdocs.yml nav if not already present.

    Returns True if nav was updated, False if entry already existed.
    This is best-effort — if the nav section for the track doesn't exist yet,
    no change is made (the user can add it manually or via scaffold_arc).
    """
    try:
        import yaml
    except ImportError:
        return False

    p = Path(mkdocs_path)
    if not p.exists():
        return False

    content = p.read_text(encoding="utf-8")
    page_ref = f"curriculum/{track}/{topic}.md"

    if page_ref in content:
        return False  # already in nav

    # Best-effort: append to nav if we find the track section
    # Full nav management is handled by scaffold_arc for batch operations
    return False


def update_arc_json(arc_id: str, node_id: str, data: dict, arcs_json_path: str) -> None:
    """Patch docs/arcs.json with updated node data (e.g., status, doc_path)."""
    p = Path(arcs_json_path)
    if not p.exists():
        return
    arcs_data = json.loads(p.read_text(encoding="utf-8"))

    for arc in arcs_data.get("arcs", []):
        if arc["id"] == arc_id:
            for node in arc.get("nodes", []):
                if node["id"] == node_id:
                    node.update(data)
                    break
            break

    p.write_text(json.dumps(arcs_data, indent=2, ensure_ascii=False), encoding="utf-8")


def git_commit(
    path: str,
    message: str,
    docs_dir: str = "../docs",
    confidence: float = 1.0,
    confidence_threshold: float | None = None,
) -> bool:
    """Stage and commit a wiki page. Returns True on success.

    Commit is gated by agent confidence:
      - If confidence >= threshold (default 0.8 or GIT_COMMIT_THRESHOLD env var): commit
      - If GIT_AUTO_PUSH=true and commit succeeded: also push to origin
      - GIT_AUTO_COMMIT=false overrides everything (safety switch for tests)

    This means agents decide whether a page is good enough to commit —
    not a static env flag set before the run.
    """
    if os.getenv("GIT_AUTO_COMMIT", "true").lower() == "false":
        return False

    threshold = confidence_threshold or float(os.getenv("GIT_COMMIT_THRESHOLD", "0.8"))
    if confidence < threshold:
        return False

    try:
        repo_root = Path(docs_dir).parent.resolve()
        abs_path = str(Path(path).resolve())
        subprocess.run(["git", "add", abs_path], cwd=repo_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        if os.getenv("GIT_AUTO_PUSH", "false").lower() == "true":
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
        return True
    except subprocess.CalledProcessError:
        return False


def log_run(state: dict, runs_dir: str = "runs") -> None:
    """Append a run record to runs/runs.jsonl and sync live docs/system/ pages.

    Called at the end of every agent run regardless of outcome.
    Tracks: topic, model, confidence, revision count, status, timing.

    Writes to two places:
      agents/runs/runs.jsonl      — append-only machine log
      docs/system/status.md       — live coverage table served by MkDocs
      docs/system/changelog.md    — per-run changelog visible on the site
    """
    runs_path = Path(__file__).parent.parent.parent / runs_dir
    runs_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    record = {
        "run_id": state.get("run_id", str(uuid.uuid4())),
        "topic": state.get("topic", ""),
        "track": state.get("track", ""),
        "page_type": state.get("page_type", "core-concept"),
        "depth_emphasis": state.get("depth_emphasis", ["applied"]),
        "mode": state.get("mode", "full"),
        "model_writer": os.getenv("WRITER_MODEL", "anthropic/claude-opus-4.7"),
        "model_reviewer": os.getenv("REVIEWER_MODEL", "google/gemini-3.1-pro-preview"),
        "started_at": state.get("started_at", ""),
        "finished_at": now.isoformat(),
        "status": "approved" if state.get("approved") else (
            "flagged" if state.get("review_confidence", 1.0) < float(
                os.getenv("GIT_COMMIT_THRESHOLD", "0.8")
            ) else "error"
        ),
        "confidence": state.get("review_confidence", 0.0),
        "revision_count": state.get("revision_count", 0),
        "output_path": state.get("output_path", ""),
        "has_mvb": state.get("mvb_decision", False),
        "committed": state.get("committed", False),
        "error": state.get("error", ""),
        "review_issues": state.get("review_issues", []),
        "review_feedback": state.get("review_feedback", "")[:500],  # first 500 chars for debug
    }

    # Append to JSONL (machine log, never rewritten)
    jsonl_path = runs_path / "runs.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Sync live docs/system/ pages
    docs_system = Path(__file__).parent.parent.parent.parent / "docs" / "system"
    docs_system.mkdir(parents=True, exist_ok=True)

    _rebuild_status_page(runs_path, docs_system)
    _append_changelog_entry(record, now, docs_system)

    # Close the feedback loop — update metrics.json + observer.md after every run
    try:
        from .observer import observe, write_metrics_json, write_observer_page
        docs_dir = str(Path(__file__).parent.parent.parent.parent / "docs")
        obs = observe(docs_dir=docs_dir, runs_dir=str(runs_path))
        write_metrics_json(obs, runs_path)
        write_observer_page(obs, docs_dir)
    except Exception:
        pass  # observer update must never break the pipeline


def _load_all_runs(runs_path: Path) -> list[dict]:
    jsonl_path = runs_path / "runs.jsonl"
    if not jsonl_path.exists():
        return []
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _rebuild_status_page(runs_path: Path, docs_system: Path) -> None:
    """Write docs/system/status.md — live coverage table shown on the wiki."""
    all_runs = _load_all_runs(runs_path)
    if not all_runs:
        return

    # Keep latest run per topic
    latest: dict[str, dict] = {}
    for r in all_runs:
        topic = r.get("topic", "")
        if topic:
            latest[topic] = r

    rows = sorted(latest.values(), key=lambda r: r.get("track", "") + r.get("topic", ""))
    total = len(rows)
    approved = sum(1 for r in rows if r.get("status") == "approved")
    with_mvb = sum(1 for r in rows if r.get("has_mvb"))
    avg_conf = sum(r.get("confidence", 0) for r in rows) / total if total else 0
    total_runs = len(all_runs)

    lines = [
        "---",
        "title: Generation Status",
        "description: Live coverage and quality metrics for every wiki page",
        "---",
        "",
        "# Generation Status",
        "",
        f"> Last updated: **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}** "
        f"· {total_runs} total runs",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pages generated | **{total}** |",
        f"| Approved (conf ≥ 0.8) | **{approved}** ({100*approved//total if total else 0}%) |",
        f"| With MVB | **{with_mvb}** |",
        f"| Avg reviewer confidence | **{avg_conf:.2f}** |",
        f"| Committed to git | **{sum(1 for r in rows if r.get('committed'))}** |",
        "",
        "## Pages",
        "",
        "| Track | Topic | Type | Status | Conf | MVB | Rev | Committed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status_icon = "✅" if r.get("status") == "approved" else "⚠️" if r.get("status") == "flagged" else "❌"
        mvb_icon = "✓" if r.get("has_mvb") else "—"
        committed_icon = "✓" if r.get("committed") else "—"
        conf = f"{r.get('confidence', 0):.2f}"
        lines.append(
            f"| {r.get('track','')} | {r.get('topic','')} "
            f"| {r.get('page_type','')} | {status_icon} {r.get('status','')} "
            f"| {conf} | {mvb_icon} | {r.get('revision_count',0)} | {committed_icon} |"
        )

    lines += [
        "",
        "---",
        "",
        "*This page is generated automatically after every agent run. "
        "Source: `agents/runs/runs.jsonl`.*",
    ]

    (docs_system / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Also keep the legacy runs/ copy
    (runs_path / "wiki_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_changelog_entry(record: dict, now: datetime, docs_system: Path) -> None:
    """Append one run entry to docs/system/changelog.md."""
    changelog_path = docs_system / "changelog.md"

    topic = record.get("topic", "")
    track = record.get("track", "")
    status = record.get("status", "")
    confidence = record.get("confidence", 0.0)
    has_mvb = record.get("has_mvb", False)
    committed = record.get("committed", False)
    revision_count = record.get("revision_count", 0)
    model_writer = record.get("model_writer", "")
    page_type = record.get("page_type", "")

    status_icon = "✅" if status == "approved" else "⚠️" if status == "flagged" else "❌"
    commit_note = " · git committed" if committed else ""
    mvb_note = " · has MVB" if has_mvb else ""

    entry_lines = [
        f"### {now.strftime('%Y-%m-%d %H:%M UTC')} — {topic}",
        "",
        f"{status_icon} **{status}** · conf={confidence:.2f}{commit_note}{mvb_note}",
        "",
        f"- **Track:** {track}",
        f"- **Page type:** {page_type}",
        f"- **Revisions:** {revision_count}",
        f"- **Writer model:** {model_writer}",
        "",
        "---",
        "",
    ]
    entry = "\n".join(entry_lines)

    if changelog_path.exists():
        existing = changelog_path.read_text(encoding="utf-8")
        # Insert after frontmatter / header, before previous entries
        if "## Entries" in existing:
            insert_at = existing.index("## Entries") + len("## Entries\n\n")
            updated = existing[:insert_at] + entry + existing[insert_at:]
        else:
            updated = existing + "\n" + entry
        changelog_path.write_text(updated, encoding="utf-8")
    else:
        header = "\n".join([
            "---",
            "title: Agent Changelog",
            "description: Every page generated or improved by the editorial agent",
            "---",
            "",
            "# Agent Changelog",
            "",
            "> Every wiki page generated, improved, or reviewed by the agent system is logged here.",
            "> Entries are prepended (newest first). Source: `agents/runs/runs.jsonl`.",
            "",
            "## Entries",
            "",
        ])
        changelog_path.write_text(header + entry, encoding="utf-8")


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

#!/usr/bin/env python3
"""Seed concept-page stubs across the canonical 10 tracks.

Each stub carries v2 frontmatter and the 🚧 marker — the FAIRE supervisor
picks them up as Priority-3 stub-generation actions and feeds them through
the editorial pipeline on the next cycle.

Run once after topping up OpenRouter:
    python3 scripts/seed-concept-stubs.py

Idempotent: existing files (stub OR generated) are skipped.
"""

from __future__ import annotations

from pathlib import Path

# Per-track topics to seed. Slugs are kebab-case. Each becomes one v2 page.
# Total: 50 stubs. Combined with the 9 existing → ~59 concept pages target.
TOPICS: dict[str, list[tuple[str, list[str]]]] = {
    "01-ai": [
        ("rlhf", ["ouyang", "christiano", "stiennon"]),
        ("mixture-of-experts", ["shazeer", "fedus", "lepikhin"]),
        ("chain-of-thought", ["wei", "kojima", "zhou"]),
        ("alignment-safety", ["amodei", "leike", "olah"]),
    ],
    "02-generative-modeling": [
        ("latent-diffusion-models", ["rombach", "podell"]),
        ("consistency-models", ["song", "salimans", "ho"]),
    ],
    "03-representation-learning": [
        ("contrastive-learning", ["chen-simclr", "he-mae", "oord-cpc"]),
        ("jepa", ["lecun", "balestriero", "assran"]),
        ("masked-autoencoders", ["he-mae", "devlin", "bao"]),
        ("self-supervised-learning", ["lecun", "chen-simclr", "grill-byol"]),
        ("simclr", ["chen-simclr", "khosla"]),
    ],
    "04-neural-networks-deep-learning": [
        ("backpropagation", ["rumelhart", "lecun", "hinton"]),
        ("optimization", ["kingma-adam", "loshchilov", "nesterov"]),
        ("regularization", ["srivastava-dropout", "ioffe-bn"]),
        ("batch-normalization", ["ioffe", "szegedy"]),
        ("residual-connections", ["he-resnet"]),
        ("scaling-laws", ["kaplan", "hoffmann", "chinchilla"]),
    ],
    "05-statistical-probabilistic-ml": [
        ("bayesian-inference", ["murphy", "bishop", "ghahramani"]),
        ("mcmc", ["hastings", "neal", "betancourt"]),
        ("variational-inference", ["jordan", "blei", "kingma"]),
        ("gaussian-processes", ["rasmussen", "williams"]),
        ("em-algorithm", ["dempster"]),
        ("uncertainty-quantification", ["gal", "kendall"]),
    ],
    "06-reinforcement-learning": [
        ("mdp", ["puterman", "bellman"]),
        ("q-learning", ["watkins", "mnih", "sutton"]),
        ("policy-gradient", ["sutton", "williams"]),
        ("actor-critic", ["konda", "mnih"]),
        ("ppo", ["schulman"]),
        ("world-models", ["ha-schmidhuber", "hafner"]),
    ],
    "07-attention-memory-reasoning-continual": [
        ("attention", ["bahdanau", "vaswani"]),
        ("multi-head-attention", ["vaswani"]),
        ("positional-encoding", ["vaswani", "su-rope"]),
        ("retrieval-augmented-generation", ["lewis-rag", "borgeaud"]),
        ("long-context", ["beltagy", "press"]),
        ("in-context-learning", ["brown", "min", "xie"]),
    ],
    "08-causal-statistical-inference": [
        ("counterfactuals", ["pearl", "rubin"]),
        ("causal-discovery", ["spirtes", "scholkopf"]),
        ("instrumental-variables", ["angrist", "imbens"]),
        ("mediation-analysis", ["baron-kenny", "imai"]),
    ],
    "09-algorithms-systems-for-ai": [
        ("flash-attention", ["dao", "rabe"]),
        ("kv-cache", ["pope", "dao"]),
        ("quantization", ["dettmers", "frantar"]),
        ("distributed-training", ["rajbhandari-zero", "huang-pipeline"]),
        ("model-parallelism", ["shoeybi-megatron", "huang"]),
        ("inference-optimization", ["dao", "kwon-vllm"]),
    ],
    "10-complexity-cognition-natural-intelligence": [
        ("emergence", ["wei-emergent", "schaeffer-mirage"]),
        ("compositionality", ["lake", "marcus"]),
        ("generalization", ["zhang-rethinking", "neyshabur"]),
        ("double-descent", ["belkin", "nakkiran"]),
        ("scaling-collapse", ["shumailov-collapse"]),
    ],
}


STUB_TEMPLATE = """---
title: {title}
slug: {slug}
layer: core
subject: {subject}
page_type: concept
state: stub
authors_anchored: [{authors}]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: []
tags: []
updated: {today}
has_mvb: true
---

# {title}

🚧 Agent-generated content pending. This concept sits in [{subject_human}](../index.md) and will be developed into a full v2 narrative walk-through (~2500 words) following [the schema](../../../../system/structure-v2.md).
"""


def slug_to_title(slug: str) -> str:
    """contrastive-learning → Contrastive Learning."""
    # Special-case a few acronyms
    overrides = {
        "rlhf": "RLHF",
        "mcmc": "MCMC",
        "ppo": "PPO",
        "mdp": "Markov Decision Processes",
        "em-algorithm": "EM Algorithm",
        "jepa": "JEPA",
        "simclr": "SimCLR",
        "kv-cache": "KV Cache",
    }
    if slug in overrides:
        return overrides[slug]
    return " ".join(word.capitalize() for word in slug.split("-"))


def subject_to_human(subject: str) -> str:
    """02-generative-modeling → '02 · Generative Modeling'."""
    num, *rest = subject.split("-", 1)
    name = " ".join(w.capitalize() for w in rest[0].split("-")) if rest else subject
    return f"{num} · {name}"


def main() -> None:
    from datetime import date
    today = date.today().isoformat()

    root = Path(__file__).resolve().parents[1] / "docs" / "curriculum" / "core"
    total = 0
    skipped = 0
    created = 0

    for subject, topics in TOPICS.items():
        concepts_dir = root / subject / "concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)

        for slug, authors in topics:
            total += 1
            path = concepts_dir / f"{slug}.md"
            if path.exists():
                skipped += 1
                continue

            body = STUB_TEMPLATE.format(
                title=slug_to_title(slug),
                slug=slug,
                subject=subject,
                authors=", ".join(authors),
                today=today,
                subject_human=subject_to_human(subject),
            )
            path.write_text(body, encoding="utf-8")
            created += 1
            print(f"  + {subject}/{slug}")

    print()
    print(f"  total: {total}  created: {created}  skipped (already existed): {skipped}")


if __name__ == "__main__":
    main()

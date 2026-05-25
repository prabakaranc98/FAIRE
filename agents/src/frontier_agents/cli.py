"""CLI for the Frontier Wiki agent system.

Usage:
  # Generate a full topic page
  uv run python generate.py --topic diffusion-models --track 02-generative-modeling

  # Generate with specific depth
  uv run python generate.py --topic transformer --track 07-attention-memory-reasoning --depth all

  # Generate only the MVB section for an existing page
  uv run python generate.py --topic rlhf --track 06-reinforcement-learning --mvb-only

  # Generate all stub topics for a track
  uv run python generate.py --track 02-generative-modeling --all-stubs

  # Dry run (generate but don't write to disk)
  uv run python generate.py --topic flow-matching --track 02-generative-modeling --dry-run
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv(Path(__file__).parent.parent.parent / ".env")

console = Console()

TRACKS = [
    "01-ai",
    "02-generative-modeling",
    "03-representation-learning",
    "04-neural-networks-dl",
    "05-statistical-probabilistic-ml",
    "06-reinforcement-learning",
    "07-attention-memory-reasoning",
    "08-causal-statistical-inference",
    "09-algorithms-systems-ai",
    "10-complexity-cognition",
    "11-robotics-embodied-ai",
    "12-physics-scientific-ai",
    "13-graph-relational-ai",
    "14-biology-life-sciences",
    "15-ml-theory-foundations",
]


@click.group()
def cli():
    """Frontier Wiki — editorial agent system."""
    pass


@cli.command()
@click.option("--topic", required=False, help="Topic slug, e.g. 'diffusion-models'")
@click.option("--track", required=True, type=click.Choice(TRACKS), help="Track slug")
@click.option(
    "--depth",
    default="all",
    type=click.Choice(["all", "applied", "foundations", "research"]),
    help="Depth level to emphasize",
)
@click.option("--mvb-only", is_flag=True, help="Generate only the Minimum Valuable Build section")
@click.option("--all-stubs", is_flag=True, help="Generate pages for all stubs in the track")
@click.option("--dry-run", is_flag=True, help="Generate but don't write to disk or commit")
@click.option("--no-commit", is_flag=True, help="Generate and write but don't git commit")
def generate(topic, track, depth, mvb_only, all_stubs, dry_run, no_commit):
    """Generate or update wiki pages using the editorial agent pipeline."""
    if dry_run:
        os.environ["GIT_AUTO_COMMIT"] = "false"
        console.print("[yellow]Dry run mode — output will not be written to disk[/yellow]")
    if no_commit:
        os.environ["GIT_AUTO_COMMIT"] = "false"

    if all_stubs:
        _generate_all_stubs(track, depth, dry_run)
        return

    if not topic:
        console.print("[red]Error: --topic is required unless --all-stubs is set[/red]")
        raise SystemExit(1)

    mode = "mvb-only" if mvb_only else "full"
    _run_pipeline(topic=topic, track=track, depth=depth, mode=mode, dry_run=dry_run)


def _run_pipeline(topic: str, track: str, depth: str, mode: str, dry_run: bool):
    from .graph import compile_mvb_graph, compile_wiki_graph
    from .state import WikiPageState

    console.print(Panel(
        f"[bold]Frontier Wiki Agent[/bold]\n"
        f"Topic: [cyan]{topic}[/cyan] | Track: [cyan]{track}[/cyan] | "
        f"Depth: [cyan]{depth}[/cyan] | Mode: [cyan]{mode}[/cyan]",
        border_style="indigo",
    ))

    initial_state: WikiPageState = {
        "topic": topic,
        "track": track,
        "depth": depth,
        "mode": mode,
    }

    graph = compile_mvb_graph() if mode == "mvb-only" else compile_wiki_graph()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running editorial pipeline...", total=None)

        result = graph.invoke(initial_state)

        progress.update(task, description="Done.")

    if result.get("approved"):
        if dry_run:
            console.print("\n[bold green]✓ Draft approved (dry run — not written)[/bold green]")
            console.print(Panel(result.get("draft", "")[:3000] + "\n...", title="Preview"))
        else:
            console.print(f"\n[bold green]✓ Page written:[/bold green] {result.get('output_path')}")
            if result.get("committed"):
                console.print("[green]✓ Git commit created[/green]")
            console.print(
                f"[dim]Reviewer confidence: {result.get('review_confidence', 0):.2f}[/dim]"
            )
    else:
        console.print(f"\n[yellow]⚠ Page requires human review[/yellow]")
        console.print(f"[dim]Confidence: {result.get('review_confidence', 0):.2f}[/dim]")
        console.print(f"[dim]Feedback:\n{result.get('review_feedback', '')}[/dim]")


def _generate_all_stubs(track: str, depth: str, dry_run: bool):
    """Find all stubs in a track and generate agent content for them."""
    docs_dir = Path(os.getenv("WIKI_DOCS_DIR", "../docs"))
    track_dir = docs_dir / "curriculum" / track

    if not track_dir.exists():
        console.print(f"[red]Track directory not found: {track_dir}[/red]")
        return

    stubs = [
        p for p in track_dir.glob("*.md")
        if p.name != "index.md" and "🚧 Agent-generated content pending" in p.read_text()
    ]

    if not stubs:
        console.print(f"[yellow]No stubs found in {track_dir}[/yellow]")
        return

    console.print(f"[cyan]Found {len(stubs)} stubs in {track}[/cyan]")
    for stub in stubs:
        topic = stub.stem
        console.print(f"\n[bold]→ Generating: {topic}[/bold]")
        _run_pipeline(topic=topic, track=track, depth=depth, mode="full", dry_run=dry_run)

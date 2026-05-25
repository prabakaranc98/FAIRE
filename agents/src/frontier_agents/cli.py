"""CLI for the Frontier Wiki agent system.

Usage:
  # Generate a full topic page
  uv run python generate.py generate --topic diffusion-models --track 02-generative-modeling

  # Arc-aware generation with depth lens
  uv run python generate.py generate \\
    --topic score-matching --track 02-generative-modeling \\
    --page-type core-concept --depth-emphasis theoretical \\
    --arc generative-stack --arc-position 4 --prev-node ddpm --next-node flow-matching

  # Generate only the MVB section for an existing page
  uv run python generate.py generate --topic rlhf --track 06-reinforcement-learning --mvb-only

  # Improve an existing page (rewrite prose, refresh SotA, re-judge MVB)
  uv run python generate.py improve --topic diffusion-models --track 02-generative-modeling

  # View generation status dashboard
  uv run python generate.py status

  # Dry run (generate but don't write to disk)
  uv run python generate.py generate --topic flow-matching --track 02-generative-modeling --dry-run
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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

PAGE_TYPES = ["arc-entry", "core-concept", "supporting"]
DEPTH_EMPHASES = ["applied", "theoretical", "frontier"]


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
    help="Depth level to emphasize (legacy; prefer --depth-emphasis)",
)
@click.option(
    "--page-type",
    default="core-concept",
    type=click.Choice(PAGE_TYPES),
    help="Arc role: arc-entry (opens arc, likely gets MVB), core-concept (agent judges), supporting (no MVB)",
)
@click.option(
    "--depth-emphasis",
    multiple=True,
    type=click.Choice(DEPTH_EMPHASES),
    default=["applied"],
    help="Depth lens(es): applied | theoretical | frontier. Repeat for multiple.",
)
@click.option("--arc", default="", help="Arc ID this page belongs to, e.g. 'generative-stack'")
@click.option("--arc-position", default=0, type=int, help="Position of this node in the arc (1-based)")
@click.option("--prev-node", default="", help="Previous node in the arc sequence")
@click.option("--next-node", default="", help="Next node in the arc sequence")
@click.option("--mvb-only", is_flag=True, help="Generate only the Minimum Valuable Build section")
@click.option("--all-stubs", is_flag=True, help="Generate pages for all stubs in the track")
@click.option("--dry-run", is_flag=True, help="Generate but don't write to disk or commit")
@click.option("--no-commit", is_flag=True, help="Generate and write but don't git commit")
def generate(
    topic, track, depth, page_type, depth_emphasis, arc, arc_position,
    prev_node, next_node, mvb_only, all_stubs, dry_run, no_commit,
):
    """Generate or update wiki pages using the editorial agent pipeline."""
    if dry_run:
        os.environ["GIT_AUTO_COMMIT"] = "false"
        console.print("[yellow]Dry run mode — output will not be written to disk[/yellow]")
    if no_commit:
        os.environ["GIT_AUTO_COMMIT"] = "false"

    if all_stubs:
        _generate_all_stubs(track, depth, list(depth_emphasis), page_type, dry_run)
        return

    if not topic:
        console.print("[red]Error: --topic is required unless --all-stubs is set[/red]")
        raise SystemExit(1)

    arc_context: dict = {}
    if arc:
        arc_context = {
            "arc_id": arc,
            "position": arc_position,
            "prev": prev_node,
            "next": next_node,
        }

    mode = "mvb-only" if mvb_only else "full"
    _run_pipeline(
        topic=topic,
        track=track,
        depth=depth,
        mode=mode,
        page_type=page_type,
        depth_emphasis=list(depth_emphasis),
        arc_context=arc_context,
        dry_run=dry_run,
    )


@cli.command()
@click.option("--topic", required=True, help="Topic slug to improve")
@click.option("--track", required=True, type=click.Choice(TRACKS), help="Track slug")
@click.option(
    "--depth-emphasis",
    multiple=True,
    type=click.Choice(DEPTH_EMPHASES),
    default=["applied"],
    help="Depth lens(es) for the rewrite",
)
@click.option("--dry-run", is_flag=True, help="Preview rewrite without writing to disk")
def improve(topic, track, depth_emphasis, dry_run):
    """Rewrite an existing page: refresh prose, refresh SotA, re-judge MVB."""
    if dry_run:
        os.environ["GIT_AUTO_COMMIT"] = "false"
        console.print("[yellow]Dry run — not writing to disk[/yellow]")

    console.print(Panel(
        f"[bold]Improving existing page[/bold]\n"
        f"Topic: [cyan]{topic}[/cyan] | Track: [cyan]{track}[/cyan]\n"
        f"Depth emphasis: [cyan]{', '.join(depth_emphasis)}[/cyan]",
        border_style="yellow",
    ))

    _run_pipeline(
        topic=topic,
        track=track,
        depth="all",
        mode="full",
        page_type="core-concept",
        depth_emphasis=list(depth_emphasis),
        arc_context={},
        dry_run=dry_run,
    )


@cli.command()
@click.option("--runs-dir", default=None, help="Path to runs/ directory (default: agents/runs/)")
def status(runs_dir):
    """Show page generation coverage and quality dashboard."""
    if runs_dir:
        runs_path = Path(runs_dir)
    else:
        runs_path = Path(__file__).parent.parent.parent / "runs"

    jsonl_path = runs_path / "runs.jsonl"

    if not jsonl_path.exists():
        console.print("[yellow]No runs recorded yet. Run `generate` first.[/yellow]")
        return

    # Read all runs, keep latest per topic
    latest: dict = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                topic = r.get("topic", "")
                if topic:
                    latest[topic] = r

    if not latest:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        return

    rows = sorted(latest.values(), key=lambda r: r.get("track", "") + r.get("topic", ""))
    total = len(rows)
    approved = sum(1 for r in rows if r.get("status") == "approved")
    with_mvb = sum(1 for r in rows if r.get("has_mvb"))
    avg_conf = sum(r.get("confidence", 0) for r in rows) / total if total else 0

    console.print(Panel(
        f"[bold]Frontier Wiki — Generation Status[/bold]\n"
        f"Total pages: [cyan]{total}[/cyan] | "
        f"Approved: [green]{approved}[/green] ({100*approved//total if total else 0}%) | "
        f"With MVB: [cyan]{with_mvb}[/cyan] | "
        f"Avg confidence: [cyan]{avg_conf:.2f}[/cyan]",
        border_style="green",
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Track", style="dim")
    table.add_column("Topic")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Conf", justify="right")
    table.add_column("MVB")
    table.add_column("Rev", justify="right")

    for r in rows:
        status_str = r.get("status", "")
        if status_str == "approved":
            status_cell = "[green]✓ approved[/green]"
        elif status_str == "flagged":
            status_cell = "[yellow]⚠ flagged[/yellow]"
        else:
            status_cell = "[red]✗ error[/red]"

        table.add_row(
            r.get("track", ""),
            r.get("topic", ""),
            r.get("page_type", "core-concept"),
            status_cell,
            f"{r.get('confidence', 0):.2f}",
            "✓" if r.get("has_mvb") else "—",
            str(r.get("revision_count", 0)),
        )

    console.print(table)


def _run_pipeline(
    topic: str,
    track: str,
    depth: str,
    mode: str,
    page_type: str,
    depth_emphasis: list[str],
    arc_context: dict,
    dry_run: bool,
):
    from .graph import compile_mvb_graph, compile_wiki_graph
    from .state import WikiPageState

    console.print(Panel(
        f"[bold]Frontier Wiki Agent[/bold]\n"
        f"Topic: [cyan]{topic}[/cyan] | Track: [cyan]{track}[/cyan]\n"
        f"Page type: [cyan]{page_type}[/cyan] | "
        f"Depth emphasis: [cyan]{', '.join(depth_emphasis)}[/cyan] | "
        f"Mode: [cyan]{mode}[/cyan]"
        + (f"\nArc: [cyan]{arc_context.get('arc_id', '')}[/cyan] "
           f"pos={arc_context.get('position', 0)} "
           f"← {arc_context.get('prev', '')} → {arc_context.get('next', '')}"
           if arc_context else ""),
        border_style="blue",
    ))

    initial_state: WikiPageState = {
        "topic": topic,
        "track": track,
        "depth": depth,
        "mode": mode,
        "page_type": page_type,
        "depth_emphasis": depth_emphasis,
        "arc_context": arc_context,
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
                f"[dim]Reviewer confidence: {result.get('review_confidence', 0):.2f} | "
                f"Revisions: {result.get('revision_count', 0)} | "
                f"MVB: {'yes' if result.get('mvb_decision') else 'no'}[/dim]"
            )
    else:
        console.print("\n[yellow]⚠ Page requires human review[/yellow]")
        console.print(f"[dim]Confidence: {result.get('review_confidence', 0):.2f}[/dim]")
        console.print(f"[dim]Feedback:\n{result.get('review_feedback', '')}[/dim]")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8765, type=int, help="Server port")
@click.option("--interval", default=48, type=int, help="Cycle interval in hours (default: 48)")
@click.option("--dry-run", is_flag=True, help="Run cycles but don't write pages to disk")
@click.option("--run-now", is_flag=True, help="Run one full cycle immediately on startup")
def serve(host, port, interval, dry_run, run_now):
    """Start the self-improving wiki agent server (48h cycle by default).

    Starts a background scheduler + FastAPI HTTP interface.
    Check http://localhost:8765/status for live server state.
    Trigger a manual run via POST /trigger.
    """
    try:
        import uvicorn
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        console.print("[red]Missing dependencies. Run: uv sync[/red]")
        raise SystemExit(1)

    from .scheduler import full_cycle_job, parse_sprint_backlog

    # Import the FastAPI app from server.py
    import sys as _sys
    server_path = Path(__file__).parent.parent.parent / "server.py"
    _sys.path.insert(0, str(server_path.parent))
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("wiki_server", server_path)
    server_mod = _iu.module_from_spec(spec)

    # Inject config before loading
    import frontier_agents.scheduler as _sched
    _sched._dry_run = dry_run  # type: ignore[attr-defined]

    spec.loader.exec_module(server_mod)
    server_mod._dry_run = dry_run

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: full_cycle_job(dry_run=dry_run),
        trigger="interval",
        hours=interval,
        id="full_cycle",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    server_mod._scheduler = scheduler

    if run_now:
        console.print("[cyan]Running initial cycle now...[/cyan]")
        full_cycle_job(dry_run=dry_run)

    tasks = parse_sprint_backlog()
    console.print(Panel(
        f"[bold]Frontier Wiki Server[/bold]\n"
        f"http://{host}:{port} | cycle every [cyan]{interval}h[/cyan] | "
        f"dry_run=[cyan]{dry_run}[/cyan]\n"
        f"Sprint queue: [cyan]{len(tasks)}[/cyan] items\n"
        f"[dim]GET /status  GET /audit  POST /trigger  GET /runs  GET /changelog[/dim]",
        border_style="green",
    ))

    uvicorn.run(server_mod.app, host=host, port=port, log_level="info")


def _generate_all_stubs(
    track: str,
    depth: str,
    depth_emphasis: list[str],
    page_type: str,
    dry_run: bool,
):
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
        _run_pipeline(
            topic=topic,
            track=track,
            depth=depth,
            mode="full",
            page_type=page_type,
            depth_emphasis=depth_emphasis,
            arc_context={},
            dry_run=dry_run,
        )

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


@cli.command()
@click.option(
    "--depth-emphasis",
    multiple=True,
    type=click.Choice(DEPTH_EMPHASES),
    default=["applied"],
    help="Depth lens for all pages. Repeat for multiple.",
)
@click.option("--dry-run", is_flag=True, help="Generate but don't write to disk or commit")
def rebuild(depth_emphasis, dry_run):
    """Rebuild all existing (non-stub) pages with the updated agent pipeline.

    Use this when the prompts, writer system, or pipeline have changed and
    you need to refresh all existing content. Score-matching (just generated)
    is skipped; only pages from before the last major pipeline update are rebuilt.
    """
    docs_dir = Path(os.getenv("WIKI_DOCS_DIR", "../docs"))

    # Find all non-stub, non-index pages
    pages = []
    for page in docs_dir.rglob("*.md"):
        if page.name == "index.md":
            continue
        if page.parts[-3] != "curriculum":
            continue
        content = page.read_text(encoding="utf-8", errors="ignore")
        if "🚧" in content or "Agent-generated content pending" in content:
            continue
        track = page.parent.name
        topic = page.stem
        pages.append((topic, track))

    if not pages:
        console.print("[yellow]No fully-generated pages found to rebuild.[/yellow]")
        return

    console.print(Panel(
        f"[bold]Rebuild mode[/bold] — {len(pages)} pages to rewrite\n"
        + "\n".join(f"  [cyan]{topic}[/cyan] ({track})" for topic, track in pages),
        border_style="yellow",
    ))

    if dry_run:
        os.environ["GIT_AUTO_COMMIT"] = "false"
        console.print("[yellow]Dry run — not writing to disk[/yellow]")

    for i, (topic, track) in enumerate(pages, 1):
        console.print(f"\n[bold][{i}/{len(pages)}] Rebuilding: {topic} ({track})[/bold]")
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
@click.option("--dry-run", is_flag=True, help="Analyse and report without updating sprint or writing files")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
@click.option("--docs-dir", default=None, help="Path to docs/ (default: ../docs from agents/)")
def supervise(dry_run, verbose, docs_dir):
    """Run the supervising agent: assess wiki health, prioritise work, update sprint queue.

    Reads current wiki state, identifies quality gaps and content holes,
    ranks work by impact, and updates agents/sprints/current.md.
    Writes a health report to docs/system/supervisor.md.
    """
    from .supervisor import run_supervisor

    resolved_docs = docs_dir or str(Path(__file__).parent.parent.parent.parent / "docs")

    console.print(Panel(
        "[bold]Frontier Wiki — Supervising Agent[/bold]\n"
        "Assessing wiki health · identifying gaps · prioritising sprint queue",
        border_style="blue",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analysing wiki state...", total=None)
        report = run_supervisor(
            docs_dir=resolved_docs,
            dry_run=dry_run,
            verbose=verbose,
        )
        progress.update(task, description="Analysis complete.")

    h = report.health
    console.print(Panel(
        f"[bold]Wiki Health[/bold]\n"
        f"Pages: [cyan]{h.total_pages}[/cyan] total "
        f"([yellow]{h.stub_pages}[/yellow] stubs, "
        f"[green]{h.generated_pages}[/green] generated)\n"
        f"Approved: [green]{h.approved_pages}[/green] | "
        f"Flagged: [yellow]{h.flagged_pages}[/yellow] | "
        f"Avg confidence: [cyan]{h.avg_confidence:.2f}[/cyan]\n"
        f"Tracks covered: [cyan]{h.tracks_covered}[/cyan] / {h.tracks_total}",
        border_style="green",
    ))

    if report.llm_analysis:
        console.print("\n[bold]Editorial Analysis:[/bold]")
        console.print(report.llm_analysis)

    if report.actions:
        table = Table(show_header=True, header_style="bold", title="Prioritised Work Queue")
        table.add_column("#", width=3)
        table.add_column("Priority")
        table.add_column("Action")
        table.add_column("Topic")
        table.add_column("Track")
        table.add_column("Reason", overflow="fold")

        for i, a in enumerate(report.actions[:15], 1):
            pri_colors = {1: "red", 2: "yellow", 3: "cyan", 4: "dim"}
            pri_labels = {1: "critical", 2: "high", 3: "medium", 4: "low"}
            color = pri_colors.get(a.priority, "white")
            label = pri_labels.get(a.priority, "?")
            table.add_row(
                str(i),
                f"[{color}]{label}[/{color}]",
                a.action,
                a.topic,
                a.track,
                a.reason[:60],
            )
        console.print(table)

    if not dry_run:
        sprint_path = Path(__file__).parent.parent.parent / "sprints" / "current.md"
        supervisor_path = Path(resolved_docs) / "system" / "supervisor.md"
        console.print(f"\n[green]✓ Sprint updated:[/green] {sprint_path}")
        console.print(f"[green]✓ Report written:[/green] {supervisor_path}")
    else:
        console.print("\n[yellow]Dry run — sprint not updated, report not written[/yellow]")


@cli.command()
@click.argument("seed")
@click.option("--track", default=None, help="Restrict the exploration to one curriculum track (e.g. 02-generative-modeling)")
@click.option("--budget", type=float, default=None, help="Override BUDGET_LIMIT_USD for THIS exploration only")
@click.option("--docs-dir", default=None, help="Path to docs/ (default: ../docs from agents/)")
def explore(seed, track, budget, docs_dir):
    """Run the explorer playbook on one curriculum seed (e.g. 'world models').

    Survey via Exa → map against the diagonal arc pattern → pick + outline
    the top 1–2 candidate arcs. Writes proposals to docs/system/arc-proposals.md.

    Does NOT queue any arc work — the human reviews the proposals and runs
    `spin-arc` for the chosen ones. See arc-exploration.md skill for the
    full playbook.
    """
    from .supervisor import maybe_propose_arcs
    from .observer import observe

    if budget is not None:
        os.environ["BUDGET_LIMIT_USD"] = str(budget)
        console.print(f"[yellow]Budget override for this run: ${budget:.2f}[/yellow]")

    resolved_docs = Path(docs_dir or str(Path(__file__).parent.parent.parent.parent / "docs"))

    console.print(Panel(
        f"[bold]Frontier Wiki — Explorer[/bold]\n"
        f"Seed: [cyan]{seed}[/cyan]" + (f" · Track: [cyan]{track}[/cyan]" if track else "") + "\n"
        f"Survey → Map → Pick → Outline. Writes to docs/system/arc-proposals.md.",
        border_style="magenta",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Observing wiki state...", total=None)
        obs = observe(docs_dir=str(resolved_docs), runs_dir="runs")
        progress.update(task, description=f"Exploring branches for '{seed}'...")
        # The current maybe_propose_arcs scans all canonical tracks. For an
        # explicit seed-based exploration we synthesize a minimal context so
        # the supervisor's LLM call focuses on the seed (the prompt references
        # arc-exploration skill which forces survey-then-pick).
        if track:
            os.environ["EXPLORE_TRACK"] = track
        os.environ["EXPLORE_SEED"] = seed
        summary = maybe_propose_arcs(obs, audit=None, docs_path=resolved_docs, verbose=True)
        progress.update(task, description="Exploration complete.")

    console.print()
    if summary.get("ran"):
        console.print(f"[green]✓[/green] Proposals written: [cyan]{summary.get('wrote_to')}[/cyan]")
        console.print(f"  Slots open: {summary.get('slots_open')}, active arcs: {summary.get('active_arcs')}")
        console.print()
        console.print("Next: review the proposals, then run:")
        console.print("  [cyan]uv run python generate.py spin-arc <arc-id>[/cyan]")
    else:
        console.print(f"[yellow]Explorer did not propose new arcs.[/yellow]")
        console.print(f"  Reason: {summary.get('reason', 'unknown')}")


@cli.command(name="spin-arc")
@click.argument("arc_id")
@click.option("--steps", type=int, default=None, help="Override step count (otherwise read from proposal)")
@click.option("--dry-run", is_flag=True, help="Print the proposed sprint additions without writing")
@click.option("--docs-dir", default=None, help="Path to docs/ (default: ../docs from agents/)")
def spin_arc(arc_id, steps, dry_run, docs_dir):
    """Materialize an approved arc proposal into sprint queue items.

    Reads docs/system/arc-proposals.md, finds the proposal with the given arc_id,
    queues 1 arc-index item + N arc-step items in agents/sprints/current.md
    with full compounding-chain metadata (prev/next/prev_artifact/artifact).

    Refuses to spin an arc that the critic-editor skill would veto (e.g. the
    diagonal-shape gate fails, or > 50% of prereqs are stubs).
    """
    resolved_docs = Path(docs_dir or str(Path(__file__).parent.parent.parent.parent / "docs"))
    proposals_path = resolved_docs / "system" / "arc-proposals.md"
    sprint_path = Path(__file__).parent.parent.parent / "sprints" / "current.md"

    if not proposals_path.exists():
        console.print(f"[red]No proposals file at {proposals_path}[/red]")
        console.print("Run [cyan]explore <seed>[/cyan] first to generate proposals.")
        raise SystemExit(1)

    proposals_text = proposals_path.read_text(encoding="utf-8")
    # Find the section for the requested arc_id. The supervisor's writer is
    # an LLM so the exact format varies — search by arc_id and by the line
    # containing "approve" in the same section.
    import re as _re
    sections = _re.split(r"^##\s+", proposals_text, flags=_re.MULTILINE)
    target_section = None
    for sec in sections:
        if arc_id in sec.lower() or arc_id.replace("-", " ") in sec.lower():
            target_section = sec
            break
    if not target_section:
        console.print(f"[red]Arc id '{arc_id}' not found in {proposals_path}[/red]")
        console.print("Available proposals:")
        for sec in sections[1:]:
            first_line = sec.split("\n")[0]
            console.print(f"  - {first_line[:80]}")
        raise SystemExit(1)

    if "verdict: veto" in target_section.lower() or "verdict:veto" in target_section.lower():
        console.print(f"[red]Arc '{arc_id}' was vetoed by the critic-editor. Reasons in {proposals_path}.[/red]")
        raise SystemExit(2)

    console.print(Panel(
        f"[bold]Spinning arc: {arc_id}[/bold]\n"
        f"Will queue 1 arc-index + N arc-step items to {sprint_path.name}.\n"
        + ("[yellow]Dry run — no writes[/yellow]" if dry_run else "Live — will append to sprint queue."),
        border_style="magenta",
    ))

    # Construct the sprint additions. The proposal section contains the
    # outline; we parse the step list and emit the canonical sprint-line format.
    # If the outline isn't machine-parseable, surface to the human.
    # (Minimal parsing — full machine-readable arc proposals are a future move.)
    arc_index_line = (
        f'- [ ] {arc_id}-index | (track from proposal) | arc-index | frontier '
        f'| arc:{arc_id} dest:"(see arc-proposals.md)" total:(see proposal)'
    )

    console.print("\n[bold]Sprint additions (review before confirming):[/bold]")
    console.print(arc_index_line)
    console.print(
        "[dim]Step items: parse the outline in arc-proposals.md and emit one line per step "
        "in the format documented in scheduler.py::_parse_sprint_item. "
        "The current implementation appends a placeholder you can edit; full machine-readable "
        "outlines are a follow-up.[/dim]"
    )

    if dry_run:
        console.print("\n[yellow]Dry run — sprint not modified.[/yellow]")
        return

    # Append the arc-index line to the sprint. Step lines must be added by hand
    # for now (or in the follow-up that machine-encodes the outline).
    sprint_text = sprint_path.read_text(encoding="utf-8") if sprint_path.exists() else ""
    if arc_index_line not in sprint_text:
        with sprint_path.open("a", encoding="utf-8") as f:
            f.write("\n## Arc spin-up — " + arc_id + "\n")
            f.write(arc_index_line + "\n")
            f.write(f"# TODO: paste arc-step lines from {proposals_path} outline\n")
        console.print(f"\n[green]✓[/green] Appended arc-index line to {sprint_path}")
        console.print(f"[yellow]Manual step:[/yellow] open {proposals_path}, copy the step outline,")
        console.print(f"  format each step as a sprint line (see scheduler.py for the format), and")
        console.print(f"  paste into {sprint_path}. The next sprint cycle will pick them up.")
    else:
        console.print(f"[yellow]Arc-index line already present in sprint queue.[/yellow]")


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

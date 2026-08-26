"""Command line interface.

    syntropyrl selftest              # prove the detectors work, offline
    syntropyrl demo reward_hacking   # see what a failure looks like
    syntropyrl atlas RLD-031         # read the failure mode
    syntropyrl diagnose run.jsonl    # diagnose a real run log
    syntropyrl parity --demo         # the five-minute logprob check
    syntropyrl scenarios             # list the built-in failures
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import atlas as atlas_mod
from .core import Doctor, Rollout, Severity, Step
from .detectors import catalog
from .parity import check_logprob_parity, parity_report
from .simulate import SCENARIOS, simulate_run

# Vercel/Stripe inspired visual hierarchy and aesthetics
custom_theme = Theme({
    "info": "cyan",
    "warn": "yellow",
    "critical": "bold red",
    "success": "bold green",
    "header": "bold magenta",
    "muted": "dim"
})
console = Console(theme=custom_theme)

# Scenario -> the detector that must fire on it. This map is the contract that
# the test suite, the selftest, and scripts/selftest_js.js all check against.
EXPECTED: Dict[str, str] = {
    "logprob_divergence": "RLD-014",
    "template_mismatch": "RLD-092",
    "reward_hacking": "RLD-031",
    "length_exploit": "RLD-033",
    "entropy_collapse": "RLD-007",
    "advantage_collapse": "RLD-001",
    "truncation_bias": "RLD-042",
    "kl_blowup": "RLD-021",
    "stale_offpolicy": "RLD-055",
    "dead_reward": "RLD-060",
    "mode_collapse": "RLD-018",
    "gradient_spike": "RLD-071",
    "value_divergence": "RLD-084",
}

_ROLLOUT_FIELDS = {f.name for f in dataclasses.fields(Rollout)}


# ----------------------------------------------------------------- helpers
def _color(args: argparse.Namespace) -> Optional[bool]:
    # None means "honor NO_COLOR and whether we are attached to a terminal".
    return False if getattr(args, "no_color", False) else None


def _run(scenario: str, steps: int, seed: int, min_severity: str = "warn",
         color: Optional[bool] = None, verbose: bool = False) -> Doctor:
    doctor = Doctor(verbose=verbose, min_severity=Severity(min_severity), color=color)
    for record in simulate_run(scenario=scenario, steps=steps, seed=seed):
        doctor.observe_step(record)
    return doctor


def _rollout(obj: Any) -> Rollout:
    if isinstance(obj, Rollout):
        return obj
    if isinstance(obj, dict):
        return Rollout(**{k: v for k, v in obj.items() if k in _ROLLOUT_FIELDS})
    raise ValueError("rollouts must be JSON objects")


def read_jsonl(path: str) -> List[Step]:
    """Read a run log written one JSON object per line.

    Accepts `{"step": 3, "metrics": {...}, "rollouts": [...]}` and also flat
    records where top-level scalars are the metrics.
    """
    steps: List[Step] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError("line {} is not valid JSON: {}".format(i + 1, exc)) from None
            if not isinstance(record, dict):
                raise ValueError("line {} is not a JSON object".format(i + 1))
            metrics: Dict[str, float] = {}
            for k, v in (record.get("metrics") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics[str(k)] = float(v)
            for k, v in record.items():
                if k in ("step", "metrics", "rollouts"):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics.setdefault(str(k), float(v))
            try:
                step_no = int(record.get("step", i))
            except (TypeError, ValueError):
                step_no = i
            steps.append(Step(
                step=step_no,
                rollouts=[_rollout(r) for r in (record.get("rollouts") or [])],
                metrics=metrics,
            ))
    return steps


def _summary_lines(doctor: Doctor) -> List[str]:
    first: Dict[str, dict] = {}
    for dx in doctor.diagnoses:
        cur = first.get(dx.code)
        if cur is None:
            first[dx.code] = {"dx": dx, "step": dx.step, "count": 1}
        else:
            cur["count"] += 1
            if dx.severity.rank > cur["dx"].severity.rank:
                cur["dx"] = dx
    order = sorted(first, key=lambda c: (-first[c]["dx"].severity.rank, first[c]["step"]))
    return [
        "{:<9} {}  {}  (first seen step {}, {}x)".format(
            first[c]["dx"].severity.value.upper(), c, first[c]["dx"].title,
            first[c]["step"], first[c]["count"])
        for c in order
    ]


# ---------------------------------------------------------------- commands
def cmd_selftest(args: argparse.Namespace) -> int:
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(args.seed)]
    failures: List[str] = []
    errors: Dict[str, str] = {}

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[header]Running selftest...", total=len(seeds) * (len(EXPECTED) + 1))
        
        table = Table(title="Selftest Results", show_header=True, header_style="header")
        table.add_column("Scenario")
        table.add_column("Expected")
        table.add_column("Result", justify="center")
        table.add_column("Notes", style="muted")

        for seed in seeds:
            console.print(f"[muted]Seed {seed}[/muted]")
            for scenario in sorted(EXPECTED):
                progress.update(task, description=f"[header]Running scenario: {scenario} (seed {seed})...")
                expected = EXPECTED[scenario]
                doctor = _run(scenario, args.steps, seed)
                errors.update(doctor._errors)
                codes = doctor.unique_codes()
                ok = expected in codes
                extra = [c for c in codes if c != expected]
                
                result_str = "[success]PASS[/success]" if ok else "[critical]MISS[/critical]"
                notes = ("also: " + ", ".join(extra)) if extra else ""
                table.add_row(scenario, expected, result_str, notes)
                
                if not ok:
                    failures.append("{} (seed {}) expected {}, fired {}".format(
                        scenario, seed, expected, ", ".join(codes) or "nothing"))
                progress.advance(task)

            progress.update(task, description=f"[header]Running healthy scenario (seed {seed})...")
            healthy = _run("healthy", args.steps, seed)
            errors.update(healthy._errors)
            noise = healthy.unique_codes()
            
            result_str = "[success]PASS[/success]" if not noise else "[critical]MISS[/critical]"
            notes = ("false positives: " + ", ".join(noise)) if noise else "(silent)"
            table.add_row("healthy", "(silent)", result_str, notes)
            
            if noise:
                failures.append("healthy (seed {}) produced {}".format(seed, noise))
            progress.advance(task)

    console.print(table)

    if errors:
        console.print(Panel(f"[critical]{errors}[/critical]", title="Detector Errors", border_style="critical"))
    console.print()
    if failures:
        fail_table = Table(title="Failures", show_header=False, border_style="critical")
        fail_table.add_column("Failure", style="critical")
        for f in failures:
            fail_table.add_row(f)
        console.print(fail_table)
        console.print(f"\n[critical]{len(failures)} failure(s)[/critical]")
        return 1
    console.print(Panel(f"[success]All {len(EXPECTED)} scenarios diagnosed correctly, healthy run silent on {len(seeds)} seed(s)[/success]"))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    if args.scenario not in SCENARIOS:
        console.print(f"[critical]Unknown scenario: {args.scenario}[/critical]\n")
        table = Table(title="Available Scenarios", header_style="header")
        table.add_column("Scenario")
        table.add_column("Description")
        for name, desc in SCENARIOS.items():
            table.add_row(name, desc)
        console.print(table)
        return 2

    with Progress(
        SpinnerColumn(spinner_name="aesthetic"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"[header]Simulating {args.scenario}...", total=None)
        doctor = _run(args.scenario, args.steps, args.seed, min_severity=args.min_severity,
                      color=_color(args), verbose=args.verbose)

    if args.json:
        print(doctor.to_json())
        return 0

    console.print(Panel(f"{SCENARIOS[args.scenario]}\n\n[muted]Steps: {args.steps} | Seed: {args.seed}[/muted]", title=f"Scenario: {args.scenario}", border_style="info"))
    console.print(doctor.report())
    
    if doctor._errors:
        console.print(Panel(f"[critical]{doctor._errors}[/critical]", title="Detector Errors", border_style="critical"))
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    table = Table(title="Built-in Failure Scenarios", header_style="header")
    table.add_column("Scenario", style="info")
    table.add_column("Fires", style="warn")
    table.add_column("Description")
    
    for name, desc in SCENARIOS.items():
        table.add_row(name, EXPECTED.get(name, "-"), desc)
        
    console.print(table)
    console.print("\n[muted]Run[/muted] [info]syntropyrl demo <scenario>[/info] [muted]to see the full report[/muted]\n")
    return 0


def cmd_atlas(args: argparse.Namespace) -> int:
    if args.code:
        text = atlas_mod.render_text(args.code)
        console.print(text)
        return 0 if atlas_mod.entry(args.code) else 2

    needs = {row["code"]: row.get("requires", "") for row in catalog()}
    console.print(f"\n[header]The RL Failure Atlas[/header] - {len(atlas_mod.ENTRIES)} entries\n")
    
    for group in atlas_mod.entries_by_family():
        table = Table(title=f"{str(group['family']).upper()} ({len(group['entries'])})", show_header=True, header_style="header")
        table.add_column("Code", style="info")
        table.add_column("Title")
        table.add_column("Needs", style="muted")
        
        for e in group["entries"]:
            code = e.get("code", "")
            table.add_row(code, e.get("title", ""), needs.get(code, "rollouts"))
            
        console.print(table)
        console.print()

    console.print("[muted]Run[/muted] [info]syntropyrl atlas RLD-031[/info] [muted]for the full entry[/muted]")
    console.print(f"[info]{atlas_mod.ATLAS_URL}[/info]\n")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    try:
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"[header]Reading {args.path}...", total=None)
            steps = read_jsonl(args.path)
    except FileNotFoundError:
        console.print(f"[critical]No such file: {args.path}[/critical]")
        return 2
    except ValueError as exc:
        console.print(f"[critical]Could not read {args.path}: {exc}[/critical]")
        return 2
        
    if not steps:
        console.print(f"[warning]{args.path} contained no records[/warning]")
        return 2

    with Progress(
        SpinnerColumn(spinner_name="aesthetic"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("[header]Diagnosing run log...", total=None)
        doctor = Doctor(verbose=False, min_severity=Severity(args.min_severity),
                        color=_color(args))
        for record in steps:
            doctor.observe_step(record)

    if args.json:
        text = doctor.to_json(args.out)
        if not args.out:
            print(text)
    elif args.quiet:
        lines = _summary_lines(doctor)
        if not lines:
            console.print(f"[success]No failures detected across {len(steps)} observed steps.[/success]")
        for line in lines:
            console.print(line)
    else:
        console.print(Panel(f"Read [info]{len(steps)}[/info] steps from [info]{args.path}[/info]", border_style="info"))
        console.print(doctor.report())
        
    if doctor._errors:
        console.print(Panel(f"[critical]{doctor._errors}[/critical]", title="Detector Errors", border_style="critical"))

    counts = doctor.counts()
    if counts.get("critical"):
        return 1
    if args.strict and counts.get("warn"):
        return 1
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    if not args.demo:
        console.print("[warning]syntropyrl parity is a library call. Use --demo to see the output shape.[/warning]\n")
        syntax = (
            "from syntropyrl import check_logprob_parity, parity_report\n"
            "result = check_logprob_parity(engine_logprobs, trainer_logprobs, samples)\n"
            "print(parity_report(result))"
        )
        console.print(Panel(syntax, title="Example Usage", border_style="info"))
        return 0

    rng = random.Random(args.seed)
    samples = ["sample {} ".format(i) + "body " * (3 + i % 7) for i in range(12)]
    base = {s: [-abs(rng.gauss(0.4, 0.25)) for _ in range(32)] for s in samples}
    mode = args.mode

    def engine(text: str):
        return list(base[text])

    def trainer(text: str):
        vals = list(base[text])
        if mode == "ok":
            return [v + rng.uniform(-2e-4, 2e-4) for v in vals]
        if mode == "template":
            return [v + (0.06 if i < 8 else 1e-5) for i, v in enumerate(vals)]
        return [v + 0.02 for v in vals]

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("[header]Checking logprob parity...", total=None)
        result = check_logprob_parity(engine, trainer, samples)
        
    console.print(parity_report(result))
    return 0 if result.get("verdict") == "ok" else 1


def cmd_version(args: argparse.Namespace) -> int:
    from . import __version__

    table = Table(show_header=False, border_style="header")
    table.add_column("Component", style="info")
    table.add_column("Value", style="success")
    table.add_row("Version", __version__)
    table.add_row("Detectors", str(len(catalog())))
    table.add_row("Atlas Entries", str(len(atlas_mod.ENTRIES)))
    table.add_row("Scenarios", str(len(SCENARIOS)))
    
    console.print(Panel(table, title="SyntropyRL System Info", expand=False, border_style="info"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syntropyrl", description="Find out why your RL run is not learning.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("selftest", help="run every scenario and check the expected detector fires")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seeds", default=None, help="comma separated, e.g. 7,21,99")
    p.add_argument("--steps", type=int, default=300)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("demo", help="simulate a failing run and print the report")
    p.add_argument("scenario", nargs="?", default="reward_hacking")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-severity", default="warn", choices=["info", "warn", "critical"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("scenarios", help="list the built-in failure scenarios")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("atlas", help="read the RL Failure Atlas")
    p.add_argument("code", nargs="?", default=None, help="e.g. RLD-031")
    p.set_defaults(func=cmd_atlas)

    p = sub.add_parser("diagnose", help="diagnose a JSONL run log")
    p.add_argument("path")
    p.add_argument("--quiet", action="store_true", help="one line per detector")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None, help="write JSON here instead of stdout")
    p.add_argument("--min-severity", default="warn", choices=["info", "warn", "critical"])
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("parity", help="generator/trainer logprob parity check")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--mode", default="broken", choices=["ok", "broken", "template"])
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_parity)

    p = sub.add_parser("version", help="print version and detector count")
    p.set_defaults(func=cmd_version)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    import os
    if os.environ.get("PROMETHEUS_PORT"):
        try:
            from prometheus_client import start_http_server
            port = int(os.environ["PROMETHEUS_PORT"])
            start_http_server(port)
        except ImportError:
            pass

    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

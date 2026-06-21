#!/usr/bin/env python3
"""
Sync tool: from_dir → to_dir (exact mirror + delete extras)
Shows diffs before applying changes
Asks per-file confirmation for deletions
After sync: git add . / commit / push in to_dir
"""

import difflib
import shutil
import subprocess
from pathlib import Path

import questionary
import typer

from social_groups.general.run_commands import run_local


def get_file_content(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        # binary file or encoding issue → treat as different
        return f"<!-- binary or unreadable file: {path.name} -->"


def show_diff(old_path: Path, new_path: Path) -> str:
    """Return unified diff string (colored-ish in terminal)"""
    a = get_file_content(old_path).splitlines()
    b = get_file_content(new_path).splitlines()

    if not a and not b:
        return ""

    diff = difflib.unified_diff(
        a,
        b,
        fromfile=str(old_path.relative_to(old_path.parent.parent)),
        tofile=str(new_path.relative_to(new_path.parent.parent)),
        lineterm="",
    )

    colored = []
    for line in diff:
        if line.startswith("+"):
            colored.append(typer.style(line, fg=typer.colors.GREEN))
        elif line.startswith("-"):
            colored.append(typer.style(line, fg=typer.colors.RED))
        elif line.startswith("@@"):
            colored.append(typer.style(line, fg=typer.colors.BLUE))
        else:
            colored.append(line)
    return "\n".join(colored)


def sync_exact_with_confirmation(from_dir: Path, to_dir: Path):
    from_dir = from_dir.resolve()
    to_dir = to_dir.resolve()

    if not from_dir.is_dir():
        typer.secho(f"Source directory not found: {from_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    to_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect what should exist after sync ──
    wanted_files = set()
    for src_path in from_dir.rglob("*"):
        if src_path.is_file():
            if src_path.name == ".DS_Store":
                continue
            if src_path.suffix == ".parquet":
                continue
            rel = src_path.relative_to(from_dir)
            wanted_files.add(rel)

    # ── 2. Find files to copy / update / delete ──
    to_delete = []
    to_copy_or_update = []

    # Check existing files in target
    for dst_path in to_dir.rglob("*"):
        if dst_path.is_file():
            rel = dst_path.relative_to(to_dir)
            if rel in wanted_files:
                src_path = from_dir / rel
                if not files_are_identical(src_path, dst_path):
                    to_copy_or_update.append((src_path, dst_path))
            else:
                to_delete.append(dst_path)

    # New files (not yet in target)
    for rel in wanted_files:
        dst_path = to_dir / rel
        if not dst_path.is_file():
            to_copy_or_update.append((from_dir / rel, dst_path))

    # ── 3. Show summary and ask for confirmation ──
    if not to_copy_or_update and not to_delete:
        typer.secho("→ Directories are already in sync.", fg=typer.colors.GREEN)
        return

    typer.secho("\nPlanned changes:", fg=typer.colors.CYAN, bold=True)

    if to_copy_or_update:
        typer.secho(f"\n  Updates / new files ({len(to_copy_or_update)}):", bold=True)
        for src, dst in sorted(to_copy_or_update, key=lambda x: x[1]):
            rel = dst.relative_to(to_dir)
            action = "UPDATE" if dst.is_file() else "NEW   "
            typer.secho(f"  {action}  {rel}", fg=typer.colors.YELLOW)

    if to_delete:
        typer.secho(f"\n  Files to DELETE ({len(to_delete)}):", bold=True)
        for p in sorted(to_delete):
            rel = p.relative_to(to_dir)
            typer.secho(f"  DELETE  {rel}", fg=typer.colors.RED)
        typer.secho(f"\n  All .aux files will be deleted without prompting.", bold=True)

    if not questionary.confirm("Apply these changes?", default=False).ask():
        typer.secho("Aborted.", fg=typer.colors.RED)
        raise typer.Exit(0)

    # ── 4. Ask individually for deletions ──
    confirmed_deletes = []
    for dst_path in to_delete:
        rel = dst_path.relative_to(to_dir)
        if dst_path.suffix == ".aux":
            confirmed_deletes.append(dst_path)
            continue
        if questionary.confirm(f"Really DELETE {rel} ?", default=False).ask():
            confirmed_deletes.append(dst_path)
        else:
            typer.secho(f"  → Keeping {rel}", fg=typer.colors.YELLOW)

    # ── 5. Perform changes ──
    typer.secho("\nApplying changes...", fg=typer.colors.CYAN)

    # Copies / updates
    for src, dst in to_copy_or_update:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        typer.secho(
            f"  copied/updated {dst.relative_to(to_dir)}", fg=typer.colors.GREEN
        )

    # Deletions
    for p in confirmed_deletes:
        p.unlink()
        typer.secho(f"  deleted {p.relative_to(to_dir)}", fg=typer.colors.RED)

    typer.secho("Sync completed.", fg=typer.colors.GREEN)


def files_are_identical(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    # For small files → content compare is fine
    # For big binary files → could use hash, but keep simple
    return a.read_bytes() == b.read_bytes()


def git_commit_and_push(
    repo_dir: Path, message: str = "Auto-sync from DAGster reports"
):
    repo_dir = repo_dir.resolve()
    run_local(["git", "add", "."], cwd=repo_dir)

    # Check if there's anything to commit
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )

    typer.echo(result.stdout)

    if not result.stdout.strip():
        typer.secho("Nothing to commit.", fg=typer.colors.YELLOW)
        return

    if not questionary.confirm(
        "Do you want to commit these changes?", default=False
    ).ask():
        typer.secho("Aborted.", fg=typer.colors.RED)
        raise typer.Exit(0)

    run_local(["git", "commit", "-m", message], cwd=repo_dir)
    typer.secho("Committed changes.", fg=typer.colors.GREEN)

    typer.secho("Pushing to remote...", fg=typer.colors.CYAN)
    run_local(["git", "push"], cwd=repo_dir)
    typer.secho("Pushed successfully.", fg=typer.colors.GREEN)

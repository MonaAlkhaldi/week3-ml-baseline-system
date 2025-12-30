# Standards (Bootcamp Repos)

These conventions are intentionally boring and consistent. The goal is **low cognitive overhead** and **easy grading**.

## Repo structure
- Importable code goes in `src/<package>/...`
- Visible tests go in `tests/`
- Optional helper scripts go in `scripts/` (thin wrappers only)
- Generated artifacts:
  - use `outputs/` for scratch outputs (gitignored)
  - use `data/processed/` for regenerated datasets (gitignored)
  - commit only small, intentional deliverables in `reports/`

## Naming
- Python modules/packages: `snake_case`
- CLI commands: `kebab-case` (console scripts)
- Functions: verbs (`load_*`, `build_*`, `run_*`)
- Avoid single-letter names except for tiny loops

## Logging
- Library code: `log = logging.getLogger(__name__)`
- CLI: configure logging once with:
  - `logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")`
- Don’t use `print()` in library code; prefer logs or return values.
- Log the **inputs**, **outputs**, and **row counts** for data pipelines.

## CLI expectations
- `--help` must work for the root CLI and all subcommands.
- Errors should produce a non-zero exit code.
- CLI should print where outputs were written.
- Keep flags stable and predictable (`--input`, `--output`, `--out-dir`, `--seed`, etc.).

## Filesystem + paths
- Use `pathlib.Path` everywhere.
- Centralize repo paths in `config.py` (e.g., `Paths.from_repo_root()`).
- Never modify files under `data/raw/` in place.

## Testing
- Tests must be deterministic:
  - seed randomness
  - use `tmp_path` for writes
  - don’t require network access
- Prefer small unit tests for core logic, plus 1–2 “smoke tests” for the CLI.

## Code style
- Format: `ruff format .`
- Lint: `ruff check .`
- Keep functions small and readable.
- Write docstrings for public functions and CLI commands.

## Dependency philosophy
- Add a dependency only if it:
  1) meaningfully reduces code/bugs, and
  2) is likely to show up in real industry repos, and
  3) does not create install friction for beginners.

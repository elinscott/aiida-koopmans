# Suggested commands

Use the shared dev venv in `../koopmans2` (this repo is installed editable there):

- `cd ../koopmans2 && uv run pytest ../aiida-koopmans2/tests` — plugin tests.
- `cd ../koopmans2 && uv run ruff check ../aiida-koopmans2` — lint.
- `cd ../koopmans2 && uv run verdi -p koopmans process list -a | tail` — inspect runs (profile `koopmans`).
- After adding/renaming entry points in pyproject.toml: `uv run verdi -p koopmans daemon restart` so the daemon picks them up.

Session runs inside a `nono` sandbox — writes outside granted paths fail; ask the user instead of working around.
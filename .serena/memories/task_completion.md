# Task completion checklist

From `../koopmans2` (shared uv venv):

1. `uv run ruff check ../aiida-koopmans2`
2. `uv run pytest ../aiida-koopmans2/tests`
3. If dispatcher-facing surface changed (task names, output TypedDict keys, entry points): also run koopmans2's own tests — `uv run pytest`.
4. Entry-point changes → `uv run verdi -p koopmans daemon restart`.
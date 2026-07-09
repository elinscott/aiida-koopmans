# Tech stack

- Python ≥3.12; build: hatchling (version dynamic from `src/aiida_koopmans/__init__.py`).
- Deps: aiida-core ≥2.7,<3; aiida-workgraph ≥0.8; aiida-wannier90 ≥2.2; aiida-quantumespresso ≥4.16.
- Development happens through sibling `../koopmans2`'s uv venv (`[tool.uv.sources]` there installs this repo editable) — run tools as `uv run …` from `../koopmans2`.
- Tooling: pytest (also hatch test envs defined), ruff line-length 100, coverage.
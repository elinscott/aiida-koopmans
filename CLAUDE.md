# aiida-koopmans

AiiDA plugin for Koopmans spectral functional calculations. Holds the `@task.graph` workflow builders and — only when upstream has no equivalent — custom CalcJobs/Parsers/Data types.

## Role in the rewrite

Part of a three-repo project. See the companion [`../koopmans2/CLAUDE.md`](../koopmans2/CLAUDE.md) for the global picture. In short:

- `../koopmans/` — legacy ASE implementation. Source of truth for physics. **Read-only.**
- `../koopmans2/` — user-facing package (CLI, Pydantic input, dispatcher).
- `./` (this repo) — the plugin layer. Exports task-graph builders that `koopmans2` composes.

## Architectural rules

1. **Prefer wrapping upstream WorkChains.** Before writing a new `CalcJob`, confirm no equivalent exists in `aiida-quantumespresso` or `aiida-wannier90-workflows`. The scout (`qe-plugin-scout` agent in `../koopmans2/.claude/agents/`) handles this check.
2. **Workflow composition uses `@task.graph` + `TypedDict` outputs.** Canonical shape — see [`workgraphs/pw.py`](src/aiida_koopmans/workgraphs/pw.py):
   ```python
   class ScfNscfOutputs(TypedDict):
       scf_remote_folder: orm.RemoteData
       nscf_remote_folder: orm.RemoteData
       ...

   PwBaseTask = task(PwBaseWorkChain)  # WorkChain-as-task at module level

   @task.graph
   def PwScfNscfTask(code, structure, ..., overrides=None) -> ScfNscfOutputs:
       builder = PwBaseWorkChain.get_builder_from_protocol(...)
       data = get_dict_from_builder(builder)
       scf_outputs = PwBaseTask(**data)
       # wire downstream via dict access
       nscf_data["pw"]["parent_folder"] = scf_outputs["remote_folder"]
       ...
       return ScfNscfOutputs(...)
   ```
3. **Use `get_builder_from_protocol` where upstream supports it.** Overrides passed as nested dicts; `get_dict_from_builder` flattens the builder to kwargs.
4. **Never access task outputs by attribute.** Use `outputs["key"]`. Attribute access breaks the workgraph.
5. **`clean_workdir` must be popped** before chaining remote folders, otherwise the upstream cleanup kills downstream inputs.
6. **New Data types** (`Band`, `Bands`, `ProjectionBlock`, …) subclass `orm.Data` and register under the `aiida.data` entry point group in `pyproject.toml`.
7. **Filename convention:** `workgraphs/<qe_tool>.py`, one module per physics step (`pw.py`, `pdos.py`, `wannier90.py`, eventually `kcw.py`, `ph.py`, `kcp.py`).

## Current state

- Workgraphs present: `PwBandsTaskViaBuilder`, `PwScfNscfTask` (`workgraphs/pw.py`), `PdosTaskViaBuilder` (`workgraphs/pdos.py`), `Wannier90TaskViaBuilder`, `Wannier90OptimizeTaskViaBuilder` (`workgraphs/wannier90.py`).
- **Cleanup needed:** `calculations.py` (DiffCalculation), `parsers.py` (DiffParser), `data/__init__.py` (DiffParameters) are `aiida-plugin-cutter` template leftovers. Safe to delete once a real Koopmans CalcJob or Data type replaces them.
- No Koopmans-specific Data types defined yet — `Band`/`Bands`/`ProjectionBlock` equivalents still live in legacy `koopmans/src/koopmans/`.
- No ASE↔AiiDA conversion utilities here; those belong in `../koopmans2/src/koopmans/aiida/conversion.py`.

## Testing

- Tests live in `tests/`. Use the AiiDA test profile via `conftest.py` fixtures.
- Existing `test_calculations.py` tests the template `DiffCalculation` — delete alongside the source.
- CI: GitHub Actions, Python 3.12, PostgreSQL + RabbitMQ services (see `.github/workflows/ci.yml`).
- Lint: `ruff` format + check.

## Dependencies

- `aiida-core`
- `aiida-workgraph>=0.8.0` — task/graph decorators
- `aiida-quantumespresso>=4.16.0` — PW, Pdos, Ph WorkChains
- `aiida-wannier90>=2.2.0`, `aiida-wannier90-workflows>=2.5.0[optimization]` — Wannier pipeline

Local editable installs from sibling paths during development.

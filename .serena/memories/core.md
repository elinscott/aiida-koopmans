# aiida-koopmans core

AiiDA plugin (`aiida-koopmans`) for Koopmans functional calculations. Consumed by sibling `../koopmans2` (user-facing CLI/dispatcher) as an editable install; never depend on koopmans2 from here.

## Source map (src/aiida_koopmans/)
- `workgraphs/` — `@task.graph` builders wrapping upstream WorkChains. `pw.py` is the canonical reference (SCF+NSCF chaining). Real workflow logic lives HERE, not in koopmans2's dispatcher.
- `calculations/` — KC-specific CalcJobs only (no upstream equivalent): `kcp.py`, `wann2kcp.py`, `merge_evc.py`.
- `parsers/` — matching parsers (`koopmans.kcp`, `koopmans.wann2kcp` entry points).
- `data/` — orm.Data subclasses (mostly superseded: projections/orbitals are TypedDicts in `types.py`).
- `types.py` — TypedDicts (VariationalOrbital, projection blocks) + SpinChannel enum.
- `utils.py` — shared helpers.

## Invariants
- Compose with `@task.graph`; NEVER add WorkChain subclasses. Wrapping upstream is fine: `task(PwBaseWorkChain)` at module scope.
- Task outputs are TypedDicts; downstream wiring via `outputs["key"]`, never attribute access.
- Builder→dict via `aiida_workgraph.utils.get_dict_from_builder`; pop `clean_workdir` before chaining.
- New entry points (calculations/parsers/data) registered in `pyproject.toml` under `koopmans.*` namespace.
- Before writing any new CalcJob, confirm upstream (`aiida-quantumespresso`, `aiida-wannier90-workflows`) has no equivalent.

Workgraph gotchas: `mem:workgraph_gotchas`. Tech/deps: `mem:tech_stack`. Commands: `mem:suggested_commands`. Style: `mem:conventions`. Done-criteria: `mem:task_completion`.

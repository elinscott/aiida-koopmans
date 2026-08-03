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

   PwBaseStep = task(PwBaseWorkChain)  # WorkChain-as-task at module level

   @task.graph
   def RunScfNscf(code, structure, ..., overrides=None) -> ScfNscfOutputs:
       builder = PwBaseWorkChain.get_builder_from_protocol(...)
       data = get_dict_from_builder(builder)
       scf_outputs = PwBaseStep(**data)
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
8. **Naming convention:** case encodes what a call creates. **PascalCase** for anything whose call creates a process node — `@task.graph` builders (verb-first: `WannierizeBlock`, `RunScfNscf`, `ComputeOrbitalScreeningParameters`; `Workflow` suffix reserved for the dispatcher entry points) and `task(WorkChain/CalcJob)` constants (`Step` suffix: `KcpStep`, `PwBaseStep`). **snake_case** for leaf `@task`/calcfunction/workfunction computations (`compute_alpha_from_dscf`). No `Task`/`ViaBuilder` suffixes; internal alpha vocabulary stays `alpha`, user-facing graph names say `ScreeningParameter(s)`.

## Current state

- Workgraphs present: `RunPwBands`, `RunScfNscf` (`workgraphs/pw.py`), `RunPdos` (`workgraphs/pdos.py`), `Wannierize`, `OptimizeWannierization` (`workgraphs/wannier90.py`).
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

A workaround for a dependency is a candidate bug report, most often for
aiida-workgraph or node-graph, which are young enough that our use finds
their edges first. An annotation shaped for the framework rather than the
contract, a value coerced to survive a serializer, a socket restructured to
get past a validator: say what the defect is, which package it lives in,
and what the workaround costs — then stop. Patching upstream is the
maintainer's call; when taken, it is a branch off their main, cherry-picked
onto our fork's `patched` (which CI clones), plus an upstream pull request.
A workaround that stays says what it works around.

## Writing

One standard for everything we write: docstrings, comments, error messages,
PR bodies, commit messages, issues. Orwell's rules — short word, cut what
can be cut, active voice, no stale figure of speech — with one carve-out:
domain terms that carry a precise meaning (Wannier function,
disentanglement, socket) and upstream keyword names stay. What is forbidden
is *coined* jargon: "a pool-carrying block enters the split chain" makes
the reader learn two terms before they learn the fact. US spelling in prose
(Wannierize, behavior); upstream names keep their own (`guiding_centres`).

**Docstrings, comments, error messages.** A docstring says what the thing
does and what must hold for it to work.

- State the rule, not a picture of it: `dis_froz_max < min_k E(num_wann +
  1, k)`, not "the window is kept inside the block's own manifold".
- Say what the function does, not where its result is used.
- Document this object's contract. Another module's behaviour, and how it
  came to be this way, both go stale silently.
- No design justification and no restating the signature: both belong in
  the pull request.
- No redundant emphasis ("it is important to note", "and they must not be
  conflated").
- Imperative summary line, one line, full stop (ruff D401).
- Error messages add one rule: tell the reader what to change, in their
  vocabulary.

**PR bodies, commit messages, issues.** Public text explains; the diff
shows. One goal governs everything here: the body is a digestible
summary of what the PR does or solves, and how — an outsider gets the
whole story from it and opens the diff only for mechanics. Every rule
below serves that goal; where they conflict, clarity wins.

- Open with what the PR achieves, in plain terms; mechanics, private
  helper names and call-site detail stay in the diff.
- `### Problem / ### Changes / ### Testing` is the default shape, not a
  form: rename, add or drop headings as the change calls for.
- Bullets lend themselves to clarity when each carries one idea,
  briefly: several sentences in one bullet is a paragraph in disguise,
  and two changes joined by a semicolon are two bullets.
- Problem is a scenario an outsider can picture — never session
  codenames, database PKs or scratch paths. Testing says what each
  check discriminates, never bare pass counts.
- Grade claims (reproduced / code-read / theory) and assert only the
  reproduced ones.
- Worked examples stand alone: write the snippet a stranger could paste.
- Check for staleness before publishing.
- Squash messages in 50/72: subject ≤50 including `(#N)`, body wrapped at
  72, opening with one sentence pairing symptom and fix, then bullets.
- No Claude session URLs; the Co-Authored-By trailer stays.

**Documentation** decouples orthogonal choices: never present a default
pairing (grouping criterion ↔ screening method) as an equivalence.

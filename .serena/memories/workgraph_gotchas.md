# aiida-workgraph gotchas (statuses verified 2026-07-02 against node-graph `patched` + aiida-workgraph main)

STILL LIVE:
- `is` against a non-None object (enum member, sentinel) in a `@task.graph` body is **silently False** — graph inputs are wrapt ObjectProxies (TaggedValue). Use `==`. `x is None` is safe (None arrives unwrapped). Re-verified by live probe after the deserialize-hook fixes: still broken.
- `@task`/`@task.graph` names must not start/end with `_` (link-label rule). Now fails **loudly at build time** with ValueError (upstream #787) instead of a silent runtime skip. Same rule for explicit `name=`/`call_link_label=` strings: `[A-Za-z0-9_]+` only — no `-`, `+`, spaces.
- Custom enums passed as graph inputs must be registered under `aiida.data` entry points as EnumData (see pyproject `koopmans.*` entries), else input serialization fails with JsonableData ValueError.
- A bare `-> dict` graph return is a non-dynamic namespace: assigning keys raises "Field 'X' is not defined and this namespace is not dynamic". Use a TypedDict return (expands per-key) or a dynamic namespace annotation.
- gather → re-scatter across `@task.graph`s: assume unsupported (not re-tested).
- Map/While zones are still dispreferred: use native `for` loops / recursive `@task.graph`s (provenance-correct, simpler). Zones were hardened (gather-clone race guard, FAILED propagation) but the preference stands.

FIXED — do NOT work around anymore:
- Subscripting a raw future socket (e.g. `make(x).result['a']`) now builds an `op_getitem` task instead of raising `GraphDeferredIllegalOperationError` (node-graph `fix-subscript-operator-task`, in the `patched` branch). Verified at build time; wrapper-`@task` unpack helpers are no longer required.
- TypedDict as `@task.graph` return annotation now expands to per-key output ports (node-graph structured-model support). No `Annotated[dict, namespace(...)]` workaround needed.
- `from __future__ import annotations` in workgraph modules works (upstream #788).
- Primitive graph inputs (str, int, …) arrive usable in the body via the deserialize hooks — `str(x)` coercion workarounds (e.g. old `str(pseudo_family)`) can be dropped; verify at each use site when simplifying.
- `@task.graph` handles regained `build`/`run`/`run_get_graph`/`submit` via aiida-workgraph `GraphTaskHandle` (local commit, PR candidate branch `graph-task-handle`) after node-graph #150 made `build` graph-only by handle type.

Preference (not correctness): wrap socket arithmetic in named `@task`s for readable provenance; single-output tasks expose `.result`. That preference now also covers `op_getitem` subscript tasks — an explicit unpack `@task` with named outputs reads better in provenance than a chain of anonymous getitem nodes; choose per readability, not necessity.

# Conventions

- ruff, line-length 100.
- One module per workflow step in `workgraphs/` with its output TypedDict(s) at the top of the module.
- `task(UpstreamWorkChain)` at module scope, never inside a `@task.graph` body (defeats caching).
- Inputs built via `get_builder_from_protocol(...)` → `get_dict_from_builder(builder)`; set `data.setdefault("metadata", {})["call_link_label"] = "<step>"` for readable provenance.
- Nested override merging via `aiida_quantumespresso.workflows.protocols.utils.recursive_merge`.
- Tests in `tests/` mirror module layout (`test_kcp_workgraph.py`, `test_block_wannierize.py`, …); AiiDA fixtures, no mocking.
- Entry-point namespace: `koopmans.<name>`.

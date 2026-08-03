[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]

# aiida-koopmans

AiiDA plugin for Koopmans spectral functional calculations with Quantum ESPRESSO.

**To run a Koopmans calculation, install [`koopmans`](https://github.com/elinscott/koopmans) instead.** That package owns the input file, the command line, and the profile, code, and pseudopotential setup, and it builds the workgraphs defined here. This repository is the plugin layer underneath, for people who extend it or who drive AiiDA themselves. Its documentation assumes you know what a `CalcJob`, a `WorkChain`, and an entry point are.

## What the plugin provides

**Workflow builders.** `@task.graph` functions that compose a Koopmans calculation out of upstream `aiida-quantumespresso` and `aiida-wannier90-workflows` workchains together with the steps only this plugin has:

- **Ground state and spectra** — `RunScfNscf`, `RunPwBands`, `RunPdos`, and `DielectricTask`, which exposes the ph.x macroscopic dielectric constant that the screening step consumes.
- **Wannierization** — `Wannierize` and `OptimizeWannierization` for a whole system; `WannierizeBlocks` to Wannierize each projection block off one shared nscf; `WannierizeAndSplitBlock` to split a block into energy-isolated groups at runtime, re-Wannierize each group, and merge the products back block-diagonally.
- **Screening parameters** — `SinglepointDFPTWorkflow` drives kcw.x through wann2kcw, screen, and ham; `KoopmansDSCFWorkflow` drives kcp.x through a DFT initialization, a trial KI, a per-orbital Delta-SCF refinement, and a final KI. The periodic Delta-SCF route reaches kcp.x through `MlwfInitialization` and `FoldToSupercell`.
- **Postprocessing and surrogates** — `UnfoldAndInterpolateTask` interpolates Wannier bands onto a k-path; `TrajectoryWorkflow` trains or tests a machine-learned model of the screening parameters over a trajectory.

**CalcJobs and parsers** for the Quantum ESPRESSO tools upstream does not cover: the Koopmans fork's `kcp.x`, `wann2kcp.x`, and `merge_evc.x`, the three calculation modes of `kcw.x`, and the `wan_mode='decompose'` pass of `pw2wannier90.x`, whose staged Wannier read-back files upstream's `Pw2wannier90Calculation` cannot supply.

**Shared computation** as plain Python and numpy: Wannier product-file merging, unfolding and interpolation, machine-learning descriptors and estimators, projection and occupation accounting. The `@task` wrappers around these stay thin, so the maths is testable without a profile.

## Entry points

Seven `aiida.calculations` entry points, each paired with the `aiida.parsers` entry point of the same name:

| Entry point | CalcJob | Parser | Runs |
| --- | --- | --- | --- |
| `koopmans.kcp` | `KcpCalculation` | `KcpParser` | `kcp.x` |
| `koopmans.kcw_wann2kc` | `Wann2kcCalculation` | `Wann2kcParser` | `kcw.x`, `calculation='wann2kcw'` |
| `koopmans.kcw_screen` | `KcwScreenCalculation` | `KcwScreenParser` | `kcw.x`, `calculation='screen'` |
| `koopmans.kcw_ham` | `KcwHamCalculation` | `KcwHamParser` | `kcw.x`, `calculation='ham'` |
| `koopmans.wann2kcp` | `Wann2kcpCalculation` | `Wann2kcpParser` | `wann2kcp.x` |
| `koopmans.merge_evc` | `MergeEvcCalculation` | `MergeEvcParser` | `merge_evc.x` |
| `koopmans.pw2wannier_decompose` | `Pw2wannierDecomposeCalculation` | `Pw2wannierDecomposeParser` | `pw2wannier90.x`, `wan_mode='decompose'` |

The `aiida.data` group registers no new node class. It maps the enums the workflow builders take as inputs — this plugin's spin channel, `ElectronicType` and `SpinType` from `aiida-quantumespresso`, and the projection, disentanglement, and optimization enums from `aiida-wannier90-workflows` — onto `aiida.orm.nodes.data.enum:EnumData`, so a member can be stored as a node and appear in the provenance graph. `[project.entry-points]` in `pyproject.toml` is the exact list.

## Installation

Neither this plugin nor all of its dependencies are on PyPI yet. `[tool.uv.sources]` points at forks carrying features still working their way upstream, and each has to be checked out as a sibling of this repository:

```shell
git clone https://github.com/elinscott/aiida-koopmans
git clone --branch patched https://github.com/elinscott/aiida-workgraph
git clone --branch patched https://github.com/elinscott/node-graph
git clone --branch patched https://github.com/elinscott/aiida-wannier90-workflows
git clone https://github.com/elinscott/aiida-wannierjl
cd aiida-koopmans
uv sync
verdi presto                          # if you have no AiiDA profile yet
verdi plugin list aiida.calculations  # the koopmans.* plugins should be listed
```

Keep that list in step with `[tool.uv.sources]`, and drop an entry as its fork merges upstream.

Running a calculation also needs the Quantum ESPRESSO binaries, set up as AiiDA codes. `kcp.x` comes from [koopmans-kcp](https://github.com/epfl-theos/koopmans-kcp), `wann2kcp.x` and `merge_evc.x` from [koopmans-qe-utils](https://github.com/epfl-theos/koopmans-qe-utils), `kcw.x` from [Quantum ESPRESSO](https://gitlab.com/QEF/q-e) itself, and `pw2wannier90.x` with `wan_mode='decompose'` from its `wann-decompose` branch.

## Development

```shell
hatch test              # run the test suite
hatch fmt --check       # ruff format and lint
hatch run mypy:check    # static type checking
hatch run docs:build    # build the documentation into docs/build/html
```

`CONTRIBUTING.md` covers the variants of each command and the pre-commit hooks. The [documentation](http://aiida-koopmans.readthedocs.io/) covers the workgraph conventions this repository follows, and holds the API reference.

## License

GNU General Public License v2, matching the koopmans package.

## Contact

edwardlinscott@gmail.com

[ci-badge]: https://github.com/elinscott/aiida-koopmans/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/elinscott/aiida-koopmans/actions/workflows/ci.yml
[cov-badge]: https://codecov.io/gh/elinscott/aiida-koopmans/branch/main/graph/badge.svg
[cov-link]: https://codecov.io/gh/elinscott/aiida-koopmans/branch/main
[docs-badge]: https://readthedocs.org/projects/aiida-koopmans/badge
[docs-link]: http://aiida-koopmans.readthedocs.io/
[pypi-badge]: https://badge.fury.io/py/aiida-koopmans.svg
[pypi-link]: https://badge.fury.io/py/aiida-koopmans

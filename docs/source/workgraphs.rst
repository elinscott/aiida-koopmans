==================
Workflow builders
==================

Every workflow in this plugin is a ``@task.graph`` function in
``aiida_koopmans.workgraphs``, one module per step. A builder takes codes,
a structure, and nested override dictionaries, and returns a ``TypedDict``
of output sockets. Composition is by calling one builder from another; there
are no ``WorkChain`` subclasses here, though upstream workchains are wrapped
as tasks and called freely.

The ``koopmans`` package's dispatcher is the largest caller: it turns an
input file into a call to one of the builders below. Reading it is the
quickest way to see a complete set of arguments in context.

Ground state, bands, and spectra
================================

``RunWannierGroundState`` (``workgraphs.wannier_ground_state``) and
``RunPwBands`` (``workgraphs.pw``) chain ``PwBaseWorkChain`` and
``PwBandsWorkChain`` from ``aiida-quantumespresso``,
stamping each step's ``CONTROL.calculation`` and pinning its k-point mesh
rather than letting a later merge decide either. ``RunPdos``
(``workgraphs.pdos``) wraps ``PdosWorkChain``. ``DielectricTask``
(``workgraphs.ph``) runs an scf and then ph.x with an electric-field
perturbation only, and exposes the isotropic average of the macroscopic
dielectric tensor as ``eps_inf`` — the quantity the screening step and the
periodic-image corrections consume.

Wannierization
==============

``Wannierize`` (``workgraphs.wannier90``) wraps the
``aiida-wannier90-workflows`` workchains for a whole system.

``WannierizeBlocks`` (``workgraphs.block_wannierize``) is the form the
Koopmans workflows use: one shared scf and nscf, then a
``Wannier90WorkChain`` per projection block that skips scf and nscf and reads
the shared scratch directly. The per-block fan-out is a native ``for`` loop
over the block list inside the graph body. Results come back as a dynamic
namespace keyed by each block's stable label, so a caller that knows its own
band ordering can pick blocks out by name.

``WannierizeAndSplitBlock`` (``workgraphs.auto_wannierize``) handles a block
whose bands separate into energy-isolated groups, or straddle the occupied
and empty manifolds. It Wannierizes the block whole, splits the result with
Wannier.jl parallel transport, re-Wannierizes each group without
disentanglement, and merges the per-group ``_u.mat``, ``_hr.dat``, and
``_centres.xyz`` back into one block-diagonal set. The split decision reads
eigenvalues from a pw.x run, so neither it nor the per-group fan-out can be
drawn when the outer graph is built; both happen inside a nested graph that
runs once those eigenvalues exist.

Screening parameters
====================

Two methods compute the screening parameters.

``SinglepointDFPTWorkflow`` (``workgraphs.dfpt``) is the kcw.x route: one
shared scf and nscf, ``WannierizeBlocks`` per spin channel, then wann2kcw,
screen, and ham. ``GroupedKcwScreening`` runs the screen step once per
representative orbital and broadcasts each result across its group.

``KoopmansDSCFWorkflow`` (``workgraphs.kcp``) is the kcp.x route: a DFT
initialization, a trial KI, a Delta-SCF calculation per orbital that refines
the screening parameters, and a final KI with the converged values.
Caller-supplied initial values either seed the refinement or, with
``calculate_alpha=False``, replace it. For a periodic system the variational
orbitals are initialized from Wannier functions: ``MlwfInitialization``
(``workgraphs.mlwf_init``) Wannierizes the blocks, ``FoldToSupercell``
(``workgraphs.folding``) folds them into :math:`\Gamma`-point supercell
wavefunctions through wann2kcp.x and merge_evc.x, and the resulting kcp.x run
restarts from them.

How orbitals are grouped, so that one calculation serves several of them, is
a separate choice from which method computes them. The grouping lives in
``workgraphs.variational_orbitals`` as a partition that each criterion only
refines: a refinement splits groups and never merges them, so exact
categorical criteria compose in any order, and a tolerance is applied last,
where it cannot bridge a categorical boundary. That a route defaults to one
criterion is a default, not an equivalence; a combination that is not wired
raises rather than quietly substituting the pairing it knows.

Postprocessing and surrogates
=============================

``UnfoldAndInterpolateTask`` (``workgraphs.ui``) unfolds a supercell Wannier
Hamiltonian onto the primitive cell and interpolates its eigenvalues along a
k-path, optionally with a smooth-interpolation correction from a denser DFT
grid and a Gaussian-smeared density of states. It handles one occupied or
empty block per spin channel per graph.

``TrajectoryWorkflow`` (``workgraphs.ml``) runs the full Delta-SCF workflow
for each snapshot of a trajectory and gathers the ``(descriptor, screening
parameter)`` pairs into a single training or evaluation task. The
``self_hartree`` descriptor is read from the final KI's parsed output; the
``power_spectrum`` descriptor instead runs ``PowerSpectrumDatasetWorkflow``,
which needs the Wannier-initialized route and a pw2wannier90.x code built
with ``wan_mode='decompose'``. Under ``mode: predict`` the same decompose
pass runs as ``PowerSpectrumDescriptorWorkflow`` inside each snapshot's
screening step, where there are no computed screening parameters to pair
the descriptors with.

A trained model records the descriptor it was fitted on, the ``correction``
and ``init_orbitals`` its screening parameters were computed under, and —
for ``power_spectrum`` — the radial basis (``n_max`` / ``l_max`` / ``r_min``
/ ``r_max``) the densities were expanded on. A model whose stamps disagree
with the run asking it to predict, or which predates a stamp, raises
``ModelMismatchError``.

Gaps
====

An input that cannot take effect raises ``NotImplementedError`` or
``ValueError`` when the graph is built, naming the gap. No keyword is dropped
in silence, so an argument that survives the build is an argument that
reached a calculation.

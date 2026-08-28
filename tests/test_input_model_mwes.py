"""Minimal examples gating the port of this package onto ``input_model=``.

Each class is the smallest graph carrying one shape the port needs, written
against the node-graph and aiida-workgraph ``input-model`` branches. Its
docstring names the aiida-koopmans shape it stands for, and every assertion is
paired with a control that fails without the checkpoint it measures, so a green
test is known to measure the branch and not something else.

The module skips wholesale where ``node_graph.input_model`` is absent, so it
stays inert on the ``patched`` branches CI installs.

``from __future__ import annotations`` is deliberately absent: it hides
``NotRequired`` from ``TypedDict.__required_keys__``, which the codes namespace
is read through.
"""

import decimal
from typing import Literal

import pytest

pytest.importorskip("node_graph.input_model")

from aiida_quantumespresso.common.types import SpinType
from aiida_workgraph import WorkGraph
from aiida_workgraph import task as awg_task
from node_graph import Graph
from node_graph import task as ng_task
from node_graph.input_model import TaskInputValidationError
from pydantic import BaseModel, field_validator

from aiida_koopmans.workgraphs import unwrap_enum
from tests.fixtures import assert_graph_roundtrips, namelist_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def processes_since(moment):
    """Return the process labels of every process node created after ``moment``.

    Counting a graph's tasks is vacuous for a refusal test: a task that never
    ran leaves no node, but so does a task whose name was uniquified. A process
    node either exists or it does not.
    """
    from aiida import orm

    builder = orm.QueryBuilder()
    builder.append(
        orm.ProcessNode,
        filters={"ctime": {">": moment}},
        project=["attributes.process_label"],
    )
    return sorted(row[0] for row in builder.all())


def now():
    """Return the current time in the timezone AiiDA stamps ``ctime`` with."""
    from aiida.common import timezone

    return timezone.now()


def stub_the_rules(monkeypatch):
    """Stub node-graph's field-rule pass, leaving the type checks in place.

    The control for every checkpoint-A test: with the rules gone the same value
    must reach further into the run, which shows the assertion measures the
    rules and not the socket layer's own type check.
    """
    import node_graph.input_model as input_model

    monkeypatch.setattr(input_model, "_rule_shadow", lambda *args, **kwargs: None)


def body_report(spin):
    """Return how a graph body's ``spin`` compares against the member it stands for.

    A graph body holds its inputs wrapped in a proxy, so what the proxy wraps
    is what a comparison sees. Called from graph bodies, whose values are read
    back through the engine rather than handed over in the caller's process.
    """
    wrapped = getattr(spin, "__wrapped__", spin)
    return (
        f"kind={type(wrapped).__name__}"
        f" equals={spin == SpinType.COLLINEAR}"
        f" identity={spin is SpinType.COLLINEAR}"
        f" unwrapped={unwrap_enum(spin, SpinType) == SpinType.COLLINEAR}"
    )


def calculation_of(parameters):
    """Return the calculation mode out of a ``parameters`` value a body received.

    A body receives a model socket either as the model instance or as the plain
    mapping behind it, depending on the layer; the keyword is the same either
    way and this example is not the one that pins which.
    """
    control = getattr(parameters, "CONTROL", None)
    if control is None:
        control = parameters["CONTROL"]
    return getattr(control, "calculation", None) or control["calculation"]


# ---------------------------------------------------------------------------
# Models shared by several examples
# ---------------------------------------------------------------------------


class PhInputs(BaseModel):
    """Inputs of the dielectric leaf; ph.x has no noncollinear perturbation."""

    spin: SpinType = SpinType.NONE
    structure: str

    @field_validator("spin")
    @classmethod
    def _perturbation_exists(cls, value):
        if value in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
            raise ValueError(
                "ph.x has no electric-field perturbation for noncollinear "
                "magnetism; set spin to 'none' or 'collinear'"
            )
        return value


class PhOutputs(BaseModel):
    """What the dielectric leaf reports about the value it was handed."""

    kind: str
    value: str


class WorkflowInputs(BaseModel):
    """Inputs of a workflow accepting every spin treatment QE names."""

    spin: SpinType = SpinType.NONE
    structure: str


class SpinOrbitDefaultInputs(BaseModel):
    """A workflow whose own default is a value its leaf cannot run."""

    spin: SpinType = SpinType.SPIN_ORBIT
    structure: str


class NoneDefaultInputs(BaseModel):
    """Declare a runnable default; the control for :class:`SpinOrbitDefaultInputs`."""

    spin: SpinType = SpinType.NONE
    structure: str


ControlNscf = namelist_model(
    "ControlNscf",
    {
        "calculation": (Literal["nscf"], "nscf"),
        "prefix": (str, "aiida"),
    },
)

ControlFree = namelist_model(
    "ControlFree",
    {
        "calculation": (str, "nscf"),
        "prefix": (str, "aiida"),
    },
)


class NscfStep(BaseModel):
    """A pw.x step whose calculation mode the route owns."""

    CONTROL: ControlNscf = ControlNscf()


class FreeStep(BaseModel):
    """Leave the mode to the user; the control for :class:`NscfStep`."""

    CONTROL: ControlFree = ControlFree()


class NscfInputs(BaseModel):
    """Inputs of a leaf whose parameters carry the owned calculation mode."""

    parameters: NscfStep = NscfStep()
    structure: str


class FreeInputs(BaseModel):
    """Declare the unforced step; the control for :class:`NscfInputs`."""

    parameters: FreeStep = FreeStep()
    structure: str


class NoteOutputs(BaseModel):
    """A one-line report a graph body assembles and a leaf returns."""

    report: str


class RoundTripInputs(BaseModel):
    """One socket of every kind the round-trip has to carry."""

    spin: SpinType = SpinType.NONE
    scale: decimal.Decimal = decimal.Decimal("1.0")
    blocks: dict[str, decimal.Decimal] = {}
    label: str | None = None


# ---------------------------------------------------------------------------
# node-graph tasks
# ---------------------------------------------------------------------------


@ng_task(input_model=PhInputs, output_model=PhOutputs)
def ng_ph(spin, structure):
    """Report the type and the value the body was handed."""
    return {"kind": type(spin).__name__, "value": getattr(spin, "value", spin)}


@ng_task(input_model=NscfInputs, output_model=PhOutputs)
def ng_nscf(parameters, structure):
    """Report the calculation mode the body was handed."""
    return {"kind": type(parameters).__name__, "value": calculation_of(parameters)}


@ng_task(input_model=FreeInputs, output_model=PhOutputs)
def ng_nscf_free(parameters, structure):
    """Accept any calculation mode; the control leaf for :func:`ng_nscf`."""
    return {"kind": type(parameters).__name__, "value": calculation_of(parameters)}


@ng_task(output_model=NoteOutputs)
def ng_note(report):
    """Return the line a graph body assembled."""
    return {"report": report}


@ng_task(input_model=RoundTripInputs, output_model=NoteOutputs)
def ng_carry(spin, scale, blocks, label):
    """Report every socket the round-trip had to carry."""
    return {"report": f"{getattr(spin, 'value', spin)}|{scale}|{sorted(blocks)}|{label}"}


@ng_task.graph(input_model=WorkflowInputs)
def ng_eps(spin, structure):
    """Wire the dielectric leaf under a workflow that takes every spin."""
    return ng_ph(spin=spin, structure=structure).kind


@ng_task.graph(input_model=SpinOrbitDefaultInputs)
def ng_eps_spin_orbit_default(spin, structure):
    """Wire the same leaf under a workflow defaulting to spin-orbit."""
    return ng_ph(spin=spin, structure=structure).kind


@ng_task.graph(input_model=NoneDefaultInputs)
def ng_eps_none_default(spin, structure):
    """Wire the same leaf under a runnable default, the control for spin-orbit."""
    return ng_ph(spin=spin, structure=structure).kind


@ng_task.graph(input_model=WorkflowInputs)
def ng_inner(spin, structure):
    """Report how the body sees its spin, and hand the same value to the leaf."""
    ng_ph(spin=spin, structure=structure)
    return ng_note(report=body_report(spin)).report


@ng_task.graph(input_model=WorkflowInputs)
def ng_outer(spin, structure):
    """Pass the spin treatment down one more graph level."""
    return ng_inner(spin=spin, structure=structure)


@ng_task.graph()
def ng_annotated_inner(spin: SpinType = SpinType.NONE, structure: str = "si"):
    """Report the same from a socket declared without a model."""
    return ng_note(report=body_report(spin)).report


@ng_task.graph(input_model=NscfInputs)
def ng_nscf_route(parameters, structure):
    """Wire the leaf that owns its calculation mode."""
    return ng_nscf(parameters=parameters, structure=structure).value


@ng_task.graph(input_model=FreeInputs)
def ng_nscf_free_route(parameters, structure):
    """Wire the leaf whose calculation mode the user may write."""
    return ng_nscf_free(parameters=parameters, structure=structure).value


@ng_task.graph(input_model=RoundTripInputs)
def ng_carry_route(spin, scale, blocks, label):
    """Wire the leaf carrying one socket of every kind."""
    return ng_carry(spin=spin, scale=scale, blocks=blocks, label=label).report


# ---------------------------------------------------------------------------
# aiida-workgraph tasks
# ---------------------------------------------------------------------------


@awg_task(input_model=PhInputs, output_model=PhOutputs)
def awg_ph(spin, structure):
    """Report the type and the value the body was handed."""
    return {"kind": type(spin).__name__, "value": getattr(spin, "value", spin)}


@awg_task(output_model=NoteOutputs)
def awg_note(report):
    """Return the line a graph body assembled."""
    return {"report": report}


@awg_task(input_model=RoundTripInputs, output_model=NoteOutputs)
def awg_carry(spin, scale, blocks, label):
    """Report every socket the round-trip had to carry."""
    return {"report": f"{getattr(spin, 'value', spin)}|{scale}|{sorted(blocks)}|{label}"}


@awg_task.graph(input_model=WorkflowInputs)
def awg_eps(spin, structure):
    """Wire the dielectric leaf under a workflow that takes every spin."""
    return awg_ph(spin=spin, structure=structure).kind


@awg_task.graph(input_model=SpinOrbitDefaultInputs)
def awg_eps_spin_orbit_default(spin, structure):
    """Wire the same leaf under a workflow defaulting to spin-orbit."""
    return awg_ph(spin=spin, structure=structure).kind


@awg_task.graph(input_model=WorkflowInputs)
def awg_inner(spin, structure):
    """Report how the body sees its spin, and hand the same value to the leaf."""
    awg_ph(spin=spin, structure=structure)
    return awg_note(report=body_report(spin)).report


@awg_task.graph(input_model=WorkflowInputs)
def awg_outer(spin, structure):
    """Pass the spin treatment down one more graph level."""
    return awg_inner(spin=spin, structure=structure)


@awg_task.graph()
def awg_annotated_inner(spin: SpinType = SpinType.NONE, structure: str = "si"):
    """Report the same from a socket declared without a model."""
    return awg_note(report=body_report(spin)).report


@awg_task.graph(input_model=RoundTripInputs)
def awg_carry_route(spin, scale, blocks, label):
    """Wire the leaf carrying one socket of every kind."""
    return awg_carry(spin=spin, scale=scale, blocks=blocks, label=label).report


# ---------------------------------------------------------------------------
# 1. Membership at the owning task
# ---------------------------------------------------------------------------


class TestMembershipAtTheOwningTask:
    """ak2 shape: ``DielectricTask`` wires ph.x under a workflow taking every SpinType.

    The workflow does not know which spin treatments ph.x can run; ph.x's own
    model says so, and the refusal has to land at the line inside the graph
    body that hands the value over, before anything is submitted.
    """

    def test_ng_supported_spin_reaches_the_leaf(self):
        graph = ng_eps.build(spin=SpinType.COLLINEAR, structure="si")
        graph.run()
        leaf = next(t for t in graph.tasks if t.name.startswith("ng_ph"))
        assert leaf.outputs.kind.value == "SpinType"
        assert leaf.outputs.value.value == "collinear"

    def test_ng_unsupported_spin_refused_at_the_wiring(self):
        with pytest.raises(TaskInputValidationError) as excinfo:
            ng_eps.build(spin=SpinType.NON_COLLINEAR, structure="si")
        message = str(excinfo.value)
        assert "'ng_ph'" in message
        assert "PhInputs" in message
        assert "no electric-field perturbation" in message

    def test_ng_control_rules_stubbed_defers_to_the_run_edge(self, monkeypatch):
        stub_the_rules(monkeypatch)
        graph = ng_eps.build(spin=SpinType.NON_COLLINEAR, structure="si")
        assert any(t.name.startswith("ng_ph") for t in graph.tasks)

    def test_awg_supported_spin_runs(self, aiida_profile):
        wg = WorkGraph("mwe1-collinear")
        wg.add_task(awg_eps, name="eps", spin=SpinType.COLLINEAR, structure="si")
        wg.run()
        assert wg.process.is_finished_ok

    def test_awg_unsupported_spin_creates_no_leaf_process(self, aiida_profile):
        mark = now()
        wg = WorkGraph("mwe1-noncollinear")
        wg.add_task(awg_eps, name="eps", spin=SpinType.NON_COLLINEAR, structure="si")
        wg.run()
        assert not wg.process.is_finished_ok
        assert processes_since(mark) == ["WorkGraph<mwe1-noncollinear>"]

    def test_awg_control_rules_stubbed_reaches_the_leaf_process(self, aiida_profile, monkeypatch):
        stub_the_rules(monkeypatch)
        mark = now()
        wg = WorkGraph("mwe1-control")
        wg.add_task(awg_eps, name="eps", spin=SpinType.NON_COLLINEAR, structure="si")
        wg.run()
        assert processes_since(mark) != ["WorkGraph<mwe1-control>"]


# ---------------------------------------------------------------------------
# 3. A forced keyword as a Literal default
# ---------------------------------------------------------------------------


class TestForcedKeywordAsLiteralDefault:
    """ak2 shape: the nscf step's ``CONTROL.calculation`` is the route's to set.

    A user may restate the value the route forces; anything else has to be
    refused by name, not silently overwritten at merge time.
    """

    def test_omitted_keyword_takes_the_forced_default(self):
        graph = ng_nscf_route.build(structure="si")
        graph.run()
        leaf = next(t for t in graph.tasks if t.name.startswith("ng_nscf"))
        assert leaf.outputs.value.value == "nscf"

    def test_restating_the_forced_value_is_accepted(self):
        graph = ng_nscf_route.build(parameters={"CONTROL": {"calculation": "nscf"}}, structure="si")
        graph.run()
        leaf = next(t for t in graph.tasks if t.name.startswith("ng_nscf"))
        assert leaf.outputs.value.value == "nscf"

    def test_another_value_is_refused_naming_the_keyword(self):
        with pytest.raises(ValueError) as excinfo:
            ng_nscf_route.build(parameters={"CONTROL": {"calculation": "scf"}}, structure="si")
        message = str(excinfo.value)
        assert "parameters.CONTROL.calculation" in message
        assert "'nscf'" in message

    def test_control_an_unforced_keyword_accepts_any_value(self):
        graph = ng_nscf_free_route.build(
            parameters={"CONTROL": {"calculation": "scf"}}, structure="si"
        )
        graph.run()
        leaf = next(t for t in graph.tasks if t.name.startswith("ng_nscf_free"))
        assert leaf.outputs.value.value == "scf"

    def test_the_helper_refuses_forcing_a_keyword_it_does_not_declare(self):
        with pytest.raises(ValueError, match="not in fields"):
            namelist_model("Typo", {"calculation": (str, "scf")}, forced=frozenset({"calculaton"}))


# ---------------------------------------------------------------------------
# 5. An enum through two graph levels
# ---------------------------------------------------------------------------


class TestEnumThroughTwoGraphLevels:
    """ak2 shape: a route graph passing ``spin`` to a sub-graph that branches on it.

    A graph body holds its inputs wrapped in a proxy, so ``is`` against a
    member is False there and the port compares by value. What the proxy wraps
    is the question these examples pin: node-graph wraps the member, so ``==``
    against the member holds; aiida-workgraph's modelled route wraps the stored
    string, and ``SpinType`` is not a ``str`` enum, so ``==`` is False and only
    ``unwrap_enum`` recovers the member.
    """

    def test_ng_body_holds_the_member(self):
        graph = ng_inner.build(spin=SpinType.COLLINEAR, structure="si")
        graph.run()
        note = next(t for t in graph.tasks if t.name.startswith("ng_note"))
        report = note.outputs.report.value
        assert "kind=SpinType" in report
        assert "equals=True" in report
        assert "identity=False" in report
        assert "unwrapped=True" in report

    def test_ng_leaf_receives_the_member_through_two_levels(self):
        graph = ng_outer.build(spin=SpinType.COLLINEAR, structure="si")
        graph.run()
        inner = next(t for t in graph.tasks if t.name.startswith("ng_inner"))
        assert "equals=True" in str(inner.outputs.result.value)

    def test_awg_body_recovers_the_member_by_unwrapping(self, aiida_profile):
        wg = WorkGraph("mwe5-unwrap")
        wg.add_task(awg_outer, name="outer", spin=SpinType.COLLINEAR, structure="si")
        wg.run()
        assert wg.process.is_finished_ok
        assert "unwrapped=True" in wg.tasks.outer.outputs.result.value.value

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "aiida-workgraph input-model: a modelled graph body's socket wraps "
            "the stored string, not the enum member the field declares"
        ),
    )
    def test_awg_body_holds_the_member(self, aiida_profile):
        wg = WorkGraph("mwe5-equals")
        wg.add_task(awg_outer, name="outer", spin=SpinType.COLLINEAR, structure="si")
        wg.run()
        report = wg.tasks.outer.outputs.result.value.value
        assert "kind=SpinType" in report
        assert "equals=True" in report

    def test_awg_control_unmodelled_body_holds_the_member(self, aiida_profile):
        wg = WorkGraph("mwe5-control")
        wg.add_task(awg_annotated_inner, name="inner", spin=SpinType.COLLINEAR, structure="si")
        wg.run()
        report = wg.tasks.inner.outputs.result.value.value
        assert "kind=SpinType" in report
        assert "equals=True" in report


# ---------------------------------------------------------------------------
# 9. A graph-input default that violates an inner rule
# ---------------------------------------------------------------------------


class TestDefaultViolatingAnInnerRule:
    """ak2 shape: a route whose own default spin the leaf it wires cannot run.

    The default is written through the same path a user's value is, so it has
    to be refused at ``build``, naming the leaf -- not at decoration, where the
    leaf's model has not been consulted, and not at run.
    """

    def test_ng_violating_default_refused_at_build(self):
        with pytest.raises(TaskInputValidationError) as excinfo:
            ng_eps_spin_orbit_default.build(structure="si")
        assert "'ng_ph'" in str(excinfo.value)

    def test_ng_control_runnable_default_builds(self):
        graph = ng_eps_none_default.build(structure="si")
        assert any(t.name.startswith("ng_ph") for t in graph.tasks)

    def test_awg_violating_default_creates_no_leaf_process(self, aiida_profile):
        mark = now()
        wg = WorkGraph("mwe9")
        wg.add_task(awg_eps_spin_orbit_default, name="eps", structure="si")
        wg.run()
        assert not wg.process.is_finished_ok
        assert processes_since(mark) == ["WorkGraph<mwe9>"]


# ---------------------------------------------------------------------------
# 11. Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """ak2 shape: every graph is rebuilt from its serialized form before it runs.

    ``wg.run()`` and the daemon both go through ``to_dict``/``from_dict``, so a
    socket kind that does not survive the round-trip dies at run start even
    though construction succeeded.

    The node-graph layer is checked by reconstruction alone. Running a
    node-graph ``Graph`` rebuilt from its own dict raises in the provenance
    recorder, which expects every graph input to still wear its socket tag --
    on the ``patched`` tip too, for a plain ``str`` socket and no model in
    sight, so it is node-graph's own defect and not this port's gate.
    """

    def test_ng_modelled_graph_round_trips(self):
        graph = ng_eps.build(spin=SpinType.COLLINEAR, structure="si")
        rebuilt = Graph.from_dict(graph.to_dict())
        assert [t.name for t in rebuilt.tasks] == [t.name for t in graph.tasks]
        assert rebuilt.tasks.graph_inputs.outputs["spin"].value == SpinType.COLLINEAR

    def test_ng_every_socket_kind_survives(self):
        graph = ng_carry_route.build(
            spin=SpinType.COLLINEAR,
            scale=decimal.Decimal("2.5"),
            blocks={"occ_1": decimal.Decimal("0.5"), "emp_1": decimal.Decimal("0.25")},
            label="si",
        )
        written = Graph.from_dict(graph.to_dict()).tasks.graph_inputs.outputs
        assert written["spin"].value == SpinType.COLLINEAR
        assert written["scale"].value == decimal.Decimal("2.5")
        assert written["label"].value == "si"
        blocks = written["blocks"]
        assert sorted(socket._name for socket in blocks) == ["emp_1", "occ_1"]
        assert blocks["occ_1"].value == decimal.Decimal("0.5")

    def test_awg_modelled_graph_round_trips(self, aiida_profile):
        wg = WorkGraph("mwe11")
        wg.add_task(awg_eps, name="eps", spin=SpinType.COLLINEAR, structure="si")
        assert_graph_roundtrips(wg)

    def test_awg_every_socket_kind_survives(self, aiida_profile):
        wg = WorkGraph("mwe11-kinds")
        wg.add_task(
            awg_carry_route,
            name="carry",
            spin=SpinType.COLLINEAR,
            scale=decimal.Decimal("2.5"),
            blocks={"occ_1": decimal.Decimal("0.5"), "emp_1": decimal.Decimal("0.25")},
            label="si",
        )
        assert_graph_roundtrips(wg)
        wg.run()
        assert wg.process.is_finished_ok
        assert wg.tasks.carry.outputs.result.value.value == ("collinear|2.5|['emp_1', 'occ_1']|si")

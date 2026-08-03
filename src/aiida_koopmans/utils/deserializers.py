"""Deserializers keeping AiiDA Data sockets as nodes on PyFunction tasks."""

from __future__ import annotations


def passthrough_node(node):
    """Identity deserializer that keeps an AiiDA Data socket as a node.

    ``aiida_pythonjob``'s default deserializer eagerly converts known Data
    types (e.g. ``StructureData → ase.Atoms``); for types it doesn't know
    (e.g. ``UpfData``) it raises because they have no ``.value`` and no
    registered deserializer. koopmans tasks that need the AiiDA node
    (e.g. for ``family.get_pseudos(structure=...)``,
    ``structure.sites`` access, or passing pseudos to ``KcpCalculation``)
    register this passthrough via ``@task(deserializers=...)``.
    """
    return node


# Plug in via ``@task(deserializers=KOOPMANS_NODE_DESERIALIZERS)`` (or
# extended copies thereof) on PyFunction tasks that take AiiDA Data
# inputs but want the node, not its deserialized payload.
KOOPMANS_NODE_DESERIALIZERS = {
    "aiida.orm.nodes.data.structure.StructureData": (
        "aiida_koopmans.utils.deserializers.passthrough_node"
    ),
    "aiida_pseudo.data.pseudo.upf.UpfData": ("aiida_koopmans.utils.deserializers.passthrough_node"),
    "aiida.orm.nodes.data.singlefile.SinglefileData": (
        "aiida_koopmans.utils.deserializers.passthrough_node"
    ),
    "aiida.orm.nodes.data.array.kpoints.KpointsData": (
        "aiida_koopmans.utils.deserializers.passthrough_node"
    ),
}

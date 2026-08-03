"""Modules shared by more than one workgraph.

A module's home encodes its widest consumer: shared by several
workgraphs, it lives here; used by a single workgraph, it co-locates
with that workgraph; used beyond ``workgraphs/``, it belongs at the
package root (``aiida_koopmans/utils/``).
"""

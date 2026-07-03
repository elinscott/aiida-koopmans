"""Koopmans DFPT workflow (kcw.x): wann2kc → screen → ham.

Port target for the legacy ``koopmans/workflows/_koopmans_dfpt.py``
(``KoopmansDFPTWorkflow``). kcw.x has no upstream aiida-quantumespresso
coverage, so this module will be backed by new CalcJobs in
``aiida_koopmans.calculations.kcw`` (one kcw.x binary, three
``control.calculation`` modes: ``wann2kc``, ``screen``, ``ham``) plus
matching parsers. Input namelists come from
``pydantic_espresso.models.kcw.develop``.

Owned by the DFPT porting stream. Nothing here yet.
"""

from __future__ import annotations

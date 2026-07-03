"""Machine-learning (trajectory) workflow: per-snapshot fan-out + model train/predict.

Port target for the legacy ``koopmans/workflows/_trajectory.py``
(``TrajectoryWorkflow``) and the ``koopmans/ml/`` package (descriptor
computation, ridge-regression screening-parameter prediction). Snapshot
fan-out uses the native for-loop pattern inside a ``@task.graph`` (Map
zones are an anti-pattern here).

Owned by the ML porting stream. Nothing here yet.
"""

from __future__ import annotations

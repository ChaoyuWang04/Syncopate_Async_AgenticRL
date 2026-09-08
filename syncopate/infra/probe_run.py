"""Reserve an immutable per-probe evidence directory before running a writer."""
from pathlib import Path
import re


def reserve_run(root: Path, experiment: str, run_id: str) -> Path:
    if not re.fullmatch(r'B[0-9]{2,}', experiment) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', run_id):
        raise ValueError('Invalid experiment or run ID')
    path = root / experiment / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def reserve_attempt(root: Path, experiment: str, run_id: str) -> Path:
    """Each delivery gets a new writer directory, including Modal preemption replay.

    A logical run can contain failed and successful attempts; callers must read a
    returned exact path and verify its terminal record, never select newest/glob.
    """
    from uuid import uuid4
    if not re.fullmatch(r'B[0-9]{2,}', experiment) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', run_id):
        raise ValueError('Invalid experiment or run ID')
    path=root/experiment/run_id/('attempt-'+uuid4().hex)
    path.mkdir(parents=True,exist_ok=False)
    return path

from pathlib import Path
import pytest
from syncopate.infra.probe_run import reserve_run


def test_reservation_refuses_reuse_and_traversal(tmp_path):
    path = reserve_run(tmp_path, 'B00', 'cpu-01')
    assert path == tmp_path / 'B00' / 'cpu-01'
    with pytest.raises(FileExistsError):
        reserve_run(tmp_path, 'B00', 'cpu-01')
    for experiment, run in [('B00', '../other'), ('../B00', 'cpu-02'), ('B00', '')]:
        with pytest.raises(ValueError):
            reserve_run(tmp_path, experiment, run)


def test_attempts_never_share_writer_directory(tmp_path):
    from syncopate.infra.probe_run import reserve_attempt
    a=reserve_attempt(tmp_path,'B01','logical-run')
    b=reserve_attempt(tmp_path,'B01','logical-run')
    assert a.parent==b.parent and a!=b
    (a/'result.json').write_text('partial')
    assert not (b/'result.json').exists()
    with pytest.raises(ValueError): reserve_attempt(tmp_path,'B01','../escape')

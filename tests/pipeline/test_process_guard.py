import signal
import sys

import pytest

from syncopate.pipeline.process_guard import run_guarded


def test_normal_exit_and_real_program_error_are_not_relabelled():
    good = run_guarded([sys.executable, '-c', 'print("done")'], timeout=5)
    assert good['rc'] == 0 and good['out'] == 'done\n' and not good['timed_out']
    bad = run_guarded([sys.executable, '-c', 'raise SystemExit(7)'], timeout=5)
    assert bad['rc'] == 7 and not bad['timed_out']


def test_quality_warn_result_does_not_abort_a_normally_exiting_process(tmp_path):
    marker = tmp_path / 'result.json'
    code = f'from pathlib import Path; Path({str(marker)!r}).write_text("quality_warn"); print("kept")'
    result = run_guarded([sys.executable, '-c', code], timeout=5, completed_path=marker)
    assert result['rc'] == 0 and result['timeout_reason'] is None
    assert marker.read_text() == 'quality_warn'


@pytest.mark.parametrize('ignore_term', [False, True])
def test_completed_but_hanging_process_is_reaped_and_results_preserved(tmp_path, ignore_term):
    marker = tmp_path / 'result.json'
    code = ('import signal, time; from pathlib import Path; '
            + ('signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore_term else '')
            + f'Path({str(marker)!r}).write_text("complete"); time.sleep(30)')
    result = run_guarded([sys.executable, '-c', code], timeout=5, completed_path=marker,
                         exit_grace=.1, stop_grace=.1, poll_interval=.02)
    assert result['rc'] != 0 and result['timeout_reason'] == 'exit_after_result'
    assert result['process_returncode'] == -(signal.SIGKILL if ignore_term else signal.SIGTERM)
    assert marker.read_text() == 'complete' and result['secs'] < 5


def test_runtime_limit_without_result_is_separate():
    result = run_guarded([sys.executable, '-c', 'import time; time.sleep(30)'],
                         timeout=.1, stop_grace=.1, poll_interval=.02)
    assert result['rc'] != 0 and result['timeout_reason'] == 'runtime'


def test_timeout_reaps_child_in_same_group_not_just_the_parent(tmp_path):
    marker = tmp_path / 'result.json'
    code = (
        'import signal, subprocess, sys, time\nfrom pathlib import Path\n'
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
        'def stop(signum, frame):\n'
        '    child.wait(timeout=1)\n'
        '    print("child_reaped", flush=True)\n'
        '    raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM, stop)\n'
        f'Path({str(marker)!r}).write_text("complete")\n'
        'time.sleep(30)\n'
    )
    result = run_guarded([sys.executable, '-c', code], timeout=5, completed_path=marker,
                         exit_grace=.2, stop_grace=2, poll_interval=.02)
    assert result['timed_out'] and result['rc'] != 0
    assert 'child_reaped' in result['out'] and result['process_returncode'] == 0


def test_stale_result_is_rejected_before_start(tmp_path):
    marker = tmp_path / 'result.json'
    marker.write_text('old')
    with pytest.raises(FileExistsError):
        run_guarded(['this-command-must-not-start'], completed_path=marker)


def test_modal_uses_the_same_guard_and_records_exit_reason():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'modal_app/stack_probe.py').read_text()
    assert 'return run_guarded(cmd' in source
    assert 'completed_path=target / "result.json" if mode == "gpu" else None' in source
    assert '"timeout_reason": result["timeout_reason"]' in source

"""Mac 可运行的轻量接线检查；数值测试在完整 CPU 栈另验。"""
import os
import subprocess
import sys


def test_hook_does_not_import_torch_before_ray_assigns_gpus(tmp_path):
    result = subprocess.run([sys.executable, "-c", """
import sys
from syncopate.train.verl_patches import setup_worker
setup_worker()
assert not any(k in sys.modules for k in ('torch', 'verl', 'vllm'))
"""], capture_output=True, text=True, timeout=30,
        env={**os.environ, "SYNCOPATE_POLICY_AUDIT_DIR": str(tmp_path)})
    assert result.returncode == 0, result.stdout + result.stderr


def test_identity_probe_is_explicit_and_uses_run_owned_path():
    command = [sys.executable, "-m", "syncopate.train.launch_rl_v1", "--dry-run", "--profile", "smoke",
               "--save-path", "checkpoints/grpo/test_identity"]
    env = {**os.environ, "SYNCOPATE_CONTRACT": "v15", "SYNCOPATE_THINK": "1", "SYNCOPATE_POLICY_PROBE": "0"}
    plain = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
    assert plain.returncode == 0, plain.stderr
    assert "SYNCOPATE_POLICY_AUDIT_DIR" not in plain.stdout
    env["SYNCOPATE_POLICY_PROBE"] = "1"
    observed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
    assert observed.returncode == 0, observed.stderr
    assert "checkpoints/grpo/test_identity/policy_evidence" in observed.stdout
    unsupported = subprocess.run([*command, "--mode", "separate_async"], capture_output=True, text=True, env=env, timeout=30)
    assert unsupported.returncode != 0 and "只验 sync/FSDP2" in unsupported.stderr

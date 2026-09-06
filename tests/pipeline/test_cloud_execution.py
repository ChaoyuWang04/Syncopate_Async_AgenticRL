import json
import subprocess
import sys
from pathlib import Path

import pytest

from syncopate.pipeline.cloud_execution import (
    audit_cache_path, bind_source, cache_writer_keys, isolated_cache, prepare_cache, prepared_cache,
    reuse_prepared_cache, validate_run_id, writer_claims,
)


@pytest.mark.parametrize("value", ["", ".", "..", "../a", "/vol", "-flag", "a;echo", "a b", "a" * 97])
def test_unsafe_run_ids_cannot_escape_or_alias_run_directory(value):
    with pytest.raises(ValueError):
        validate_run_id(value)


def test_valid_run_id_is_unchanged():
    assert validate_run_id("t11_20260905a.full-1") == "t11_20260905a.full-1"


def test_modal_orchestration_imports_without_training_venv():
    root = Path(__file__).resolve().parents[2]
    # -I -S 不加载当前工作目录、venv 包或 PYTHONPATH，模拟 Modal 的外层 Python。
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); "
            "from syncopate.pipeline.cloud_execution import writer_claims, validate_run_id; "
            "from syncopate.pipeline.stages import modal_plan; "
            "assert len(modal_plan('train-all')) == 10; "
            "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = (root / "modal_app/stack_probe.py").read_text()
    assert "sys.path.insert(0, CURRENT_OVERLAY)" in source


class Registry(dict):
    def put(self, key, value, *, skip_if_exists):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def test_two_writers_cannot_claim_same_run_and_failed_claim_releases_only_own_keys():
    registry = Registry()
    with writer_claims(registry, ["run:a"], "owner-a"):
        with pytest.raises(RuntimeError):
            with writer_claims(registry, ["data", "run:a"], "owner-b"):
                pytest.fail("第二个写者不应进入")
        assert registry == {"run:a": "owner-a"}
        with writer_claims(registry, ["run:b"], "owner-b"):
            assert len(registry) == 2
    assert not registry


def test_cache_arms_are_independent_and_do_not_change_seed(tmp_path):
    seed = tmp_path / "vllm_cache"
    seed.mkdir()
    (seed / "entry").write_text("warm")
    a = isolated_cache(tmp_path, "a")
    b = isolated_cache(tmp_path, "b")
    from pathlib import Path
    (Path(a["VLLM_CACHE_ROOT"]) / "entry").write_text("arm-a")
    assert (Path(b["VLLM_CACHE_ROOT"]) / "entry").read_text() == "warm"
    assert (seed / "entry").read_text() == "warm"


def test_resume_refuses_different_source_or_an_unbound_old_run(tmp_path):
    bind_source(tmp_path, {"sha": "a"})
    bind_source(tmp_path, {"sha": "a"})
    with pytest.raises(RuntimeError):
        bind_source(tmp_path, {"sha": "b"})
    old = tmp_path / "old"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps({"stages": {}}))
    with pytest.raises(RuntimeError):
        bind_source(old, {"sha": "a"})


def test_cpu_prepares_cache_once_gpu_requires_matching_completed_source(tmp_path):
    directory = tmp_path / "prepare"
    with pytest.raises(FileNotFoundError):
        prepared_cache(directory, {"sha": "a"})
    first = prepare_cache(tmp_path, directory, {"sha": "a"})
    assert prepare_cache(tmp_path, directory, {"sha": "a"}) == first
    assert prepared_cache(directory, {"sha": "a"}) == first["env"]
    with pytest.raises(RuntimeError):
        prepared_cache(directory, {"sha": "b"})


def test_interrupted_cpu_cache_copy_does_not_leave_a_ready_marker(tmp_path, monkeypatch):
    from syncopate.pipeline import cloud_execution as cloud
    def interrupted(*args):
        raise OSError("copy interrupted")
    monkeypatch.setattr(cloud, "isolated_cache", interrupted)
    with pytest.raises(OSError):
        prepare_cache(tmp_path, tmp_path / "prepare", {"sha": "a"})
    assert not (tmp_path / "prepare/cache_ready.json").exists()


def test_reused_cache_keeps_both_sources_and_cannot_have_two_gpu_writers(tmp_path):
    old = prepare_cache(tmp_path, tmp_path / "old", {"sha": "old"})
    new = reuse_prepared_cache(tmp_path / "old", tmp_path / "new", {"sha": "new"})
    assert new["source"] == {"sha": "new"}
    assert new["reused_from"]["source"] == {"sha": "old"}
    assert json.loads((tmp_path / "old/cache_ready.json").read_text()) == old
    assert new["env"] == old["env"]
    registry = Registry()
    with writer_claims(registry, cache_writer_keys(old["env"]), "a"):
        with pytest.raises(RuntimeError):
            with writer_claims(registry, cache_writer_keys(new["env"]), "b"):
                pytest.fail("同一物理缓存不可同时写")


def test_cache_pointer_cannot_redirect_writes_to_models(tmp_path):
    env = isolated_cache(tmp_path, "safe")
    env["VLLM_CACHE_ROOT"] = str(tmp_path / "models")
    with pytest.raises(RuntimeError, match="独立"):
        cache_writer_keys(env)


def test_audit_cache_accepts_a_symlinked_volume_mount(tmp_path):
    physical = tmp_path / "physical"
    cache = physical / "_audit/run/cpu"
    cache.mkdir(parents=True)
    mount = tmp_path / "vol"
    mount.symlink_to(physical, target_is_directory=True)
    assert audit_cache_path(mount, "_audit/run/cpu") == cache.resolve()


@pytest.mark.parametrize("path", ["", "/vol/_audit/run", "../run", "_audit/../models", "models", "_audit"])
def test_audit_cache_rejects_paths_outside_its_scope(tmp_path, path):
    (tmp_path / "_audit").mkdir()
    (tmp_path / "models").mkdir()
    with pytest.raises(ValueError):
        audit_cache_path(tmp_path, path)


def test_audit_cache_rejects_an_inner_symlink_escape(tmp_path):
    (tmp_path / "_audit").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "_audit/escape").symlink_to(tmp_path / "models", target_is_directory=True)
    with pytest.raises(ValueError):
        audit_cache_path(tmp_path, "_audit/escape")


def test_cache_writer_key_uses_physical_path_for_mount_aliases(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    env = isolated_cache(physical, "run")
    mount = tmp_path / "vol"
    mount.symlink_to(physical, target_is_directory=True)
    alias = {key: str(mount / Path(value).relative_to(physical)) for key, value in env.items()}
    assert cache_writer_keys(alias) == cache_writer_keys(env)
    registry = Registry()
    with writer_claims(registry, cache_writer_keys(env), "first"):
        with pytest.raises(RuntimeError):
            with writer_claims(registry, cache_writer_keys(alias), "second"):
                pytest.fail("软链接别名不能绕过同一物理缓存的单写者限制")


def test_modal_cache_preparation_uses_the_shared_boundary_check():
    source = (Path(__file__).resolve().parents[2] / "modal_app/stack_probe.py").read_text()
    assert "previous = audit_cache_path(pathlib.Path(VOL), cache_from)" in source

import pytest

from syncopate.train.source_snapshot import archive_source, snapshot_source, source_digest, verify_orchestrator


def test_orchestrator_must_equal_frozen_source(tmp_path):
    current, frozen = tmp_path / "current.py", tmp_path / "frozen.py"
    current.write_text("same")
    frozen.write_text("same")
    assert len(verify_orchestrator(current, frozen)) == 64
    current.write_text("changed while building image")
    with pytest.raises(RuntimeError):
        verify_orchestrator(current, frozen)


def test_snapshot_survives_live_edit_and_keeps_identity(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("before")
    (tmp_path / "pyproject.toml").write_text("test")
    owner, frozen, digest = snapshot_source(tmp_path, ("src",), ("pyproject.toml",))
    try:
        (tmp_path / "src" / "a.py").write_text("after")
        assert (frozen / "src" / "a.py").read_text() == "before"
        assert source_digest(frozen, ("src",), ("pyproject.toml",)) == digest
        assert source_digest(tmp_path, ("src",), ("pyproject.toml",)) != digest
    finally:
        owner.cleanup()


def archive_inputs(tmp_path):
    root = tmp_path / "source"
    (root / "src/pkg").mkdir(parents=True)
    (root / "src/pkg.py").write_text("file before directory in lexical sort")
    (root / "src/pkg/a.py").write_text("directory before file in Path sort")
    (root / "pyproject.toml").write_text("test")
    return root, ("src",), ("pyproject.toml",)


def test_archive_reads_back_exact_snapshot_and_is_idempotent(tmp_path):
    root, dirs, files = archive_inputs(tmp_path)
    expected = source_digest(root, dirs, files)
    output = tmp_path / "source.tar.gz"
    result = archive_source(root, dirs, files, expected=expected, output=output)
    before = output.read_bytes()
    assert result["overlay_sha256"] == expected and result["members"] == 3
    assert archive_source(root, dirs, files, expected=expected, output=output) == result
    assert output.read_bytes() == before


def test_archive_wrong_source_does_not_create_output(tmp_path):
    root, dirs, files = archive_inputs(tmp_path)
    output = tmp_path / "source.tar.gz"
    with pytest.raises(ValueError, match="运行快照"):
        archive_source(root, dirs, files, expected="wrong", output=output)
    assert not output.exists()


def test_archive_existing_other_content_is_preserved_and_rejected(tmp_path):
    import io
    import tarfile
    root, dirs, files = archive_inputs(tmp_path)
    output = tmp_path / "source.tar.gz"
    with tarfile.open(output, "x:gz") as archive:
        for name in ("src/pkg.py", "src/pkg/a.py", "pyproject.toml"):
            item = tarfile.TarInfo(name)
            item.size = 3
            archive.addfile(item, io.BytesIO(b"bad"))
    before = output.read_bytes()
    with pytest.raises(ValueError, match="禁止覆盖"):
        archive_source(root, dirs, files, expected=source_digest(root, dirs, files), output=output)
    assert output.read_bytes() == before
    assert not output.with_suffix(".gz.json").exists()

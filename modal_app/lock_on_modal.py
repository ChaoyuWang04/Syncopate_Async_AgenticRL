"""在 Modal 容器里跑 `uv lock`（本机拉 GitHub 上的大轮子会超时），把 uv.lock 写回 modal_app/stack/。
    modal run modal_app/lock_on_modal.py
"""
import pathlib
import modal

STACK = pathlib.Path(__file__).resolve().parent / "stack"
app = modal.App("syncopate-stack2-lock")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "UV_HTTP_TIMEOUT": "600"})
    .add_local_file(STACK / "pyproject.toml", "/env/pyproject.toml", copy=True)
    .add_local_file(STACK / "uv.toml", "/env/uv.toml", copy=True)
)


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def lock() -> dict:
    import subprocess
    p = subprocess.run("cd /env && uv lock 2>&1 | tail -5", shell=True, capture_output=True, text=True)
    lock_txt = open("/env/uv.lock").read() if p.returncode == 0 else ""
    return {"rc": p.returncode, "log": p.stdout + p.stderr, "lock": lock_txt}


@app.local_entrypoint()
def main():
    r = lock.remote()
    print(r["log"])
    if r["rc"] == 0 and r["lock"]:
        (STACK / "uv.lock").write_text(r["lock"]); print(f"uv.lock written ({len(r['lock'])} bytes)")
    else:
        raise SystemExit(1)

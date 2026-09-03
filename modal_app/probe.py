"""Modal 搬家探针 —— 回答「家能不能搬到 Modal 上」这一个问题，每步一条 ✅/🔴 判据行。

    modal run modal_app/probe.py                        # 默认跑 P1–P6（单卡；不含双卡 nccl 与 pytest）
    modal run modal_app/probe.py --steps nccl           # 双卡 NCCL（训练前跑一次，~3 min）
    modal run modal_app/probe.py --steps gpu,volume     # 只跑某几步
    modal run modal_app/probe.py --steps pytest         # 全量测试（CPU，~30 min）
    modal run --detach modal_app/probe.py --steps rebuild   # 长步骤用 --detach，断网不中止

前置（本机）：`uv tool install modal` 之后 `modal token set --token-id ak-… --token-secret as-…`
（或 export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET）。token 由 Chaoyu 在 modal.com 生成。

★ 设计纪律（00 §5 守则①②③⑦）
  · 每步判据写成「两个东西应当相同」或「退出码 == 0」，不写阈值；
  · 每步结果落盘两处：Volume `/vol/_audit/modal_probe/<step>.json` 与本机 `_audit/modal_probe/`；
  · 每步幂等可重跑（Modal GPU 函数可抢占且不可关：抢占后 Modal 会用同一输入重跑）；
  · 「没跑成」与「判红」分开报（退出码 2 vs 1，同 check_pipeline_invariants）。

★ 家的形状（跑通后写进 08 §Modal）
  镜像  = nvidia/cuda 12.8 devel + python 3.12 + uv + `uv sync --frozen --all-extras --no-install-project`
          （只带 pyproject/uv.lock/uv.toml；环境在镜像里，代码不在）
  Volume `syncopate-home` 挂 /vol：/vol/repo = git clone（代码经 GitHub 传递，不上传本机目录）·
          /vol/models = HF 拉的权重 · /vol/data = 在 Modal 上重新生成的数据 · /vol/_audit = 判据落盘
  跑法  = PYTHONPATH=/vol/repo，cwd=/vol/repo，解释器 /env/.venv/bin/python
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import modal

# ─────────────────────────── 常量（唯一来源） ───────────────────────────
APP_NAME = "syncopate-probe"
REPO_URL = "https://github.com/ChaoyuWang04/Syncopate_Async_AgenticRL.git"
REPO_BRANCH = "main"
VOL_NAME = "syncopate-home"
VOL = "/vol"
REPO = f"{VOL}/repo"
MODELS = f"{VOL}/models"
AUDIT = f"{VOL}/_audit/modal_probe"
GPU_PAIR = "RTX-PRO-6000:2"
GPU_ONE = "RTX-PRO-6000"
BASE_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"  # docker hub 实核存在（09-03）
PY = "/env/.venv/bin/python"
# cu13torch2.9 的 flash-attn 轮子要 libcudart.so.13，它装在 venv 的 nvidia/cu13/lib（08 §2.2）
LD = "/env/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
# 要拉的权重：Qwen3-4B 是学生（试点②可比性必须用它，前任 09-03 确认）；0.6B 是 31 个测试的分词器
HF_MODELS = {"Qwen/Qwen3-4B": f"{MODELS}/Qwen3-4B", "Qwen/Qwen3-0.6B": f"{MODELS}/Qwen3-0.6B"}

LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# ─────────────────────────── 镜像 ───────────────────────────
# 只放依赖表三件套；代码经 git 进 Volume。改依赖 ⇒ 镜像层重建；改代码 ⇒ 镜像不动。
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/cuda/bin",
            "UV_PYTHON": "3.12",
            "UV_LINK_MODE": "copy",
            "LD_LIBRARY_PATH": LD,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file(LOCAL_ROOT / "pyproject.toml", "/env/pyproject.toml", copy=True)
    .add_local_file(LOCAL_ROOT / "uv.lock", "/env/uv.lock", copy=True)
    .add_local_file(LOCAL_ROOT / "uv.toml", "/env/uv.toml", copy=True)
    # --no-install-project：项目源码不在镜像里（有 [build-system]，不加会去找源码）
    .run_commands("cd /env && uv sync --frozen --all-extras --no-install-project")
)

RUN_ENV = {
    "PYTHONPATH": REPO,
    "SYNCOPATE_CONTRACT": "v15",
    "SYNCOPATE_THINK": "1",
    "HF_HOME": f"{VOL}/hf_cache",
}


# ─────────────────────────── 小工具 ───────────────────────────
def _sh(cmd: str, *, cwd: str | None = None, env: dict | None = None, timeout: int | None = None) -> dict:
    """跑一条 shell，返回 {rc, out, secs, timed_out}。超时是判据的一部分（NCCL 挂死就靠它抓）。"""
    e = dict(os.environ)
    e.update(RUN_ENV)
    if env:
        e.update(env)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": (p.stdout + p.stderr)[-6000:], "secs": round(time.time() - t0, 1), "timed_out": False}
    except subprocess.TimeoutExpired as ex:
        out = ((ex.stdout or b"").decode(errors="replace") + (ex.stderr or b"").decode(errors="replace"))[-6000:]
        return {"rc": -1, "out": out, "secs": round(time.time() - t0, 1), "timed_out": True}


def _record(step: str, ok: bool, details: dict) -> dict:
    rec = {"step": step, "ok": bool(ok), "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **details}
    os.makedirs(AUDIT, exist_ok=True)
    with open(f"{AUDIT}/{step}.json", "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    vol.commit()
    print(f"[{step}] {'✅' if ok else '🔴'}")
    return rec


def _sync_repo() -> str:
    """把 /vol/repo 同步到 origin/main（幂等）。返回 HEAD sha。"""
    if not os.path.isdir(f"{REPO}/.git"):
        r = _sh(f"git clone --branch {REPO_BRANCH} {REPO_URL} {REPO}", timeout=600)
        if r["rc"] != 0:
            raise RuntimeError("git clone 失败：" + r["out"])
    r = _sh(f"git fetch origin {REPO_BRANCH} && git reset --hard origin/{REPO_BRANCH} && git clean -fd -e data -e _audit -e logs",
            cwd=REPO, timeout=600)
    if r["rc"] != 0:
        raise RuntimeError("git 同步失败：" + r["out"])
    vol.commit()
    return _sh("git rev-parse HEAD", cwd=REPO)["out"].strip()



def _topology() -> dict:
    """每次拿到卡都记一份拓扑指纹：宿主机/云/区域/CPU/NUMA/卡间连接。跨次对比才知道 Modal 给的机器变不变。"""
    return {
        "modal_env": {k: v for k, v in os.environ.items() if k.startswith("MODAL_") and "TOKEN" not in k},
        "hostname": _sh("hostname")["out"].strip(),
        "cpu": _sh("lscpu | grep -E 'Model name|^NUMA|Socket|^CPU\\(s\\)'")["out"].strip(),
        "mem_gb": _sh("free -g | awk '/Mem:/{print $2}'")["out"].strip(),
        "nvidia_smi_topo": _sh("nvidia-smi topo -m 2>&1 | head -20")["out"].strip(),
        "gpu_bus": _sh("nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader")["out"].strip(),
    }

# ─────────────────────────── P1 · 镜像 + 解释器 ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=600)
def p_image() -> dict:
    """镜像能起、venv 里 torch/vllm/verl/flash_attn 能 import（CPU 上只查 import，不查 CUDA）。"""
    r = _sh(
        f"{PY} -c \"import torch, vllm, verl, flash_attn, transformers; "
        f"print('torch', torch.__version__, 'cuda', torch.version.cuda); "
        f"print('vllm', vllm.__version__); print('transformers', transformers.__version__)\"",
        timeout=300,
    )
    nvcc = _sh("nvcc --version | tail -1")
    return _record("image", r["rc"] == 0, {"import": r, "nvcc": nvcc["out"].strip(), "python": PY})


# ─────────────────────────── P2 · 代码经 GitHub ───────────────────────────
def _violations(out: str) -> list[str]:
    """从 check_pipeline_invariants 的尾部汇总里取「被违反的判据」名单（"   - ★ …" 行）。"""
    tail = out.split("条判据被违反：")[-1] if "条判据被违反：" in out else ""
    return sorted(l.strip()[2:].strip() for l in tail.splitlines() if l.strip().startswith("- "))


@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=900)
def p_git(expected_sha: str, local_violations: list[str]) -> dict:
    """/vol/repo 与 origin/main 同 sha；`python -m syncopate --help` 能跑（PYTHONPATH 接线判据）；
    check_pipeline_invariants 在 Modal 上违反的判据集合 ⊆ 本机的集合（仓库里本来就红的历史审计/旧日志不算 Modal 的错，
    但 Modal 上**多出来**的违反一定是环境问题）。"""
    sha = _sync_repo()
    r = _sh(f"{PY} -m syncopate --help", cwd=REPO, timeout=300)
    inv = _sh(f"{PY} scripts/check_pipeline_invariants.py", cwd=REPO, timeout=600)
    remote_v = _violations(inv["out"])
    extra = sorted(set(remote_v) - set(local_violations))
    ok = sha == expected_sha and r["rc"] == 0 and not extra
    return _record("git", ok, {"head": sha, "expected": expected_sha, "cli": r,
                               "invariants": {"rc": inv["rc"], "remote_violations": remote_v,
                                              "local_violations": local_violations, "extra_on_modal": extra,
                                              "tail": inv["out"][-1500:]}})


# ─────────────────────────── P3 · Volume 一致性 ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=1, memory=1024, timeout=300)
def p_volume_write(marker: str) -> dict:
    os.makedirs(f"{VOL}/_probe", exist_ok=True)
    with open(f"{VOL}/_probe/marker.txt", "w") as f:
        f.write(marker)
    vol.commit()
    return {"written": marker, "container": os.environ.get("MODAL_TASK_ID", "?")}


@app.function(image=image, volumes={VOL: vol}, cpu=1, memory=1024, timeout=300)
def p_volume_read(marker: str) -> dict:
    """另一个容器 reload 后读到的内容 == 写入的内容（跨容器可见性判据）。"""
    vol.reload()
    try:
        got = open(f"{VOL}/_probe/marker.txt").read()
    except FileNotFoundError:
        got = None
    du = _sh(f"du -sh {VOL}/* 2>/dev/null")
    return _record("volume", got == marker, {"expected": marker, "got": got, "du": du["out"].strip(),
                                            "container": os.environ.get("MODAL_TASK_ID", "?")})


# ─────────────────────────── P4 · GPU 三件套 ───────────────────────────
_NCCL_2CARD = r'''
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
def w(rank):
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]="29711"
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=2)
    n = 64*(1<<20)
    x = torch.full((n,), float(rank+1), device=f"cuda:{rank}")
    dist.all_reduce(x); torch.cuda.synchronize()
    assert torch.all(x == 3.0), "all_reduce 结果错"
    src = torch.full((n//2,), float(rank+1), device=f"cuda:{rank}"); dst = torch.zeros(n, device=f"cuda:{rank}")
    dist.all_gather_into_tensor(dst, src); torch.cuda.synchronize()
    assert dst[:n//2].eq(1.0).all() and dst[n//2:].eq(2.0).all(), "all_gather 结果错"
    import time; t0=time.perf_counter()
    for _ in range(10): dist.all_reduce(x)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/10
    if rank==0: print(f"NCCL_OK all_reduce 256MB {dt*1e3:.1f} ms  algbw {n*4/dt/1e9:.1f} GB/s")
    dist.destroy_process_group()
mp.spawn(w, nprocs=2, join=True)
'''


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=4, memory=16384, timeout=1200)
def p_gpu() -> dict:
    """单卡（搬家最便宜的形状）：① PRO 6000 · sm_120 · 驱动可见；② flash-attn 反向判据退出码 0。"""
    _sync_repo()
    smi = _sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")
    tor = _sh(
        f"{PY} -c \"import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])\"",
        timeout=300,
    )
    sm120 = tor["rc"] == 0 and tor["out"].strip().startswith("1 [(12, 0)]")
    fa = _sh(f"{PY} scripts/check_flash_attn_backward.py", cwd=REPO, timeout=600)
    ok = sm120 and fa["rc"] == 0
    return _record("gpu", ok, {"nvidia_smi": smi["out"].strip(), "torch": tor, "flash_attn_backward": fa, "topology": _topology()})


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_PAIR, cpu=8, memory=32768, timeout=1200)
def p_nccl() -> dict:
    """双卡（按需，训练前跑一次）：两张都是 sm_120；NCCL 在四种环境变量组合下各自通/挂死/红（挂死 = 超时 120s 抓）。"""
    tor = _sh(
        f"{PY} -c \"import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])\"",
        timeout=300,
    )
    two_sm120 = tor["rc"] == 0 and tor["out"].strip().startswith("2 [(12, 0), (12, 0)]")
    open("/tmp/nccl2.py", "w").write(_NCCL_2CARD)
    variants = {
        "default": {},
        "CUMEM0": {"NCCL_CUMEM_ENABLE": "0"},                       # 4×5090 上真正管用的那条（08 §5）
        "P2P_DISABLE": {"NCCL_P2P_DISABLE": "1"},                   # 交接信/论坛案例说的那条
        "CUMEM0+P2P_DISABLE": {"NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1"},
    }
    nccl = {}
    for name, env in variants.items():
        r = _sh(f"{PY} /tmp/nccl2.py", env=env, timeout=120)
        nccl[name] = {"pass": r["rc"] == 0 and "NCCL_OK" in r["out"], "hang": r["timed_out"], "secs": r["secs"],
                      "tail": r["out"][-400:]}
    p2p = _sh(f"{PY} -c \"import torch; print('p2p 0->1', torch.cuda.can_device_access_peer(0,1))\"", timeout=120)
    ok = two_sm120 and any(v["pass"] for v in nccl.values())
    return _record("nccl", ok, {"torch": tor, "nccl_2card": nccl, "p2p": p2p["out"].strip(), "topology": _topology()})


# ─────────────────────────── P5 · 权重从 HF 落 Volume + 单卡冒烟 ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600)
def p_model_download() -> dict:
    """snapshot_download 到 /vol/models；判据 = config.json 在 + 权重总字节 == HF 仓库声明的总字节（不是「大概 7.6G」）。"""
    code = r'''
import json, os, sys
from huggingface_hub import snapshot_download, HfApi
out = {}
for repo_id, local in json.loads(sys.argv[1]).items():
    snapshot_download(repo_id, local_dir=local, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "merges.txt", "vocab.json"])
    info = HfApi().model_info(repo_id, files_metadata=True)
    want = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    got = sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local) if f.endswith(".safetensors"))
    out[repo_id] = {"local": local, "config": os.path.exists(os.path.join(local, "config.json")), "bytes_hf": want, "bytes_vol": got, "same": want == got}
json.dump(out, open("/tmp/dl_result.json", "w"))   # 结果走文件，不走 stdout（tqdm 进度条会混进 stderr）
'''
    open("/tmp/dl.py", "w").write(code)
    r = _sh(f"{PY} /tmp/dl.py '{json.dumps(HF_MODELS)}'", timeout=3300)
    vol.commit()
    res = {}
    try:
        res = json.load(open("/tmp/dl_result.json"))
    except Exception:
        pass
    ok = r["rc"] == 0 and bool(res) and all(v["config"] and v["same"] for v in res.values())
    return _record("model_download", ok, {"models": res, "run": {k: r[k] for k in ("rc", "secs")}, "tail": r["out"][-800:]})


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=4, memory=16384, timeout=900)
def p_model_smoke() -> dict:
    """单卡 bf16 加载 Qwen3-4B 并生成 8 个 token：能加载 + 输出非空 + 贪心两次逐 token 相同（确定性判据）。"""
    vol.reload()
    code = r'''
import torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p); m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda")
ids = tok("投放预算怎么定？", return_tensors="pt").to("cuda")
outs = [m.generate(**ids, max_new_tokens=8, do_sample=False)[0].tolist() for _ in range(2)]
print("SAME", outs[0] == outs[1]); print("TEXT", repr(tok.decode(outs[0][ids.input_ids.shape[1]:])))
print("MEM_GB", round(torch.cuda.max_memory_allocated()/2**30, 2))
'''
    open("/tmp/smoke.py", "w").write(code)
    r = _sh(f"{PY} /tmp/smoke.py {MODELS}/Qwen3-4B", timeout=800)
    ok = r["rc"] == 0 and "SAME True" in r["out"] and "TEXT ''" not in r["out"]
    return _record("model_smoke", ok, {"run": r})


# ─────────────────────────── P6 · 数据在 Modal 上重新生成 ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=3600)
def p_rebuild_v13() -> dict:
    """影子重建 0–4 步（08 §3 / run_pipeline_shadow_rebuild.sh）：外部数据 git diff 为空 · v11→v12→v13 · 切分三份 SHA 与 git 里的 data/splits/v13 逐一相同。
    通过后把 batches/v13 与 splits/v13 放到 /vol/repo/data/ 现役位置（后续 W4 建库直接用）。"""
    _sync_repo()
    sh = f"{VOL}/data/shadow_rebuild"
    steps = [
        ("0a external", f"{PY} scripts/make_test_external_data.py"),
        ("0b ingest", f"{PY} scripts/ingest_external.py"),
        ("0c git-diff", "git diff --quiet -- data/external ':(exclude)*.xlsx'"),
        ("1 v11", f"{PY} -m syncopate cases generate --spec configs/buckets/v11.yaml --out {sh}/batches/v11 && "
                  f"{PY} scripts/set_tool_menus.py --batch {sh}/batches/v11 --sft-audit _audit/v8_sft_epoch1.json"),
        ("2 v12", f"{PY} -m syncopate cases generate --spec configs/buckets/v12.yaml --out {sh}/batches/v12 && "
                  f"{PY} scripts/set_tool_menus.py --batch {sh}/batches/v12 --sft-audit _audit/v8_sft_epoch1.json --freeze-from {sh}/batches/v11"),
        ("3 v13", f"{PY} -m syncopate cases generate --spec configs/buckets/v13.yaml --out {sh}/batches/v13 && "
                  f"{PY} scripts/set_tool_menus.py --batch {sh}/batches/v13 --sft-audit _audit/v8_sft_epoch1.json --freeze-from {sh}/batches/v12"),
        ("4 split", f"{PY} -m syncopate data split --batch {sh}/batches/v13 --out {sh}/splits/v13"),
    ]
    log = {}
    for name, cmd in steps:
        r = _sh(cmd, cwd=REPO, timeout=2400)
        log[name] = {k: r[k] for k in ("rc", "secs")} | {"tail": r["out"][-500:]}
        if r["rc"] != 0:
            return _record("rebuild_v13", False, {"failed_at": name, "log": log})
    # 判据：三份切分 SHA-256 与 git 里的现役一致（split_report 含自引用 batch_dir，剔除后比）
    sha = {}
    for f in ("eval_cases", "sft_cases", "rl_cases"):
        a = _sh(f"sha256sum {sh}/splits/v13/{f}.json | cut -d' ' -f1")["out"].strip()
        b = _sh(f"sha256sum {REPO}/data/splits/v13/{f}.json | cut -d' ' -f1")["out"].strip()
        sha[f] = {"shadow": a, "git": b, "same": a == b}

    def strip(p):
        d = json.load(open(p))
        d.get("args", d).pop("batch_dir", None)
        for v in d.values():
            if isinstance(v, dict):
                v.pop("batch_dir", None)
        return d

    report_same = strip(f"{sh}/splits/v13/split_report.json") == strip(f"{REPO}/data/splits/v13/split_report.json")
    ok = all(v["same"] for v in sha.values()) and report_same
    if ok:
        _sh(f"rm -rf {REPO}/data/batches/v13 {REPO}/data/splits/v13 && mkdir -p {REPO}/data/batches {REPO}/data/splits && "
            f"cp -r {sh}/batches/v13 {REPO}/data/batches/v13 && cp -r {sh}/splits/v13 {REPO}/data/splits/v13 && "
            f"git -C {REPO} checkout -- data/splits/v13")   # splits 在 git 里，回到 git 版本（内容已判等）
    nfiles = _sh(f"find {sh}/batches/v13 -type f | wc -l")["out"].strip()
    vol.commit()
    return _record("rebuild_v13", ok, {"splits_sha": sha, "split_report_same": report_same, "batch_files": nfiles, "log": log})


# ─────────────────────────── P7 · 全量测试（按需） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=16, memory=32768, timeout=3600)
def p_pytest() -> dict:
    """本机基线 908 passed（09-02）。无 PG/Redis/GPU 的会 skip 或红——按 08 §2 的表读，别把 skip 当通过。"""
    _sync_repo()
    r = _sh(f"{PY} -m pytest tests -q -p no:cacheprovider --ignore=tests/train/test_e31_unified_fp8.py 2>&1 | tail -40",
            cwd=REPO, timeout=3500)
    return _record("pytest", r["rc"] == 0, {"run": r})


# ─────────────────────────── 本机入口 ───────────────────────────
ALL_STEPS = ["image", "git", "volume", "gpu", "model", "rebuild"]


@app.local_entrypoint()
def main(steps: str = ",".join(ALL_STEPS)):
    want = [s.strip() for s in steps.split(",") if s.strip()]
    results: dict[str, dict] = {}
    t0 = time.time()

    def run(name, fn, *a):
        try:
            results[name] = fn.remote(*a)
        except Exception as ex:  # 没跑成 ≠ 判红，分开记
            results[name] = {"step": name, "ok": None, "error": repr(ex)[:2000]}
            print(f"[{name}] ⚠️ 没跑成：{repr(ex)[:300]}")

    if "image" in want:
        run("image", p_image)
    if "git" in want:
        sha = subprocess.run(["git", "ls-remote", REPO_URL, f"refs/heads/{REPO_BRANCH}"], capture_output=True, text=True).stdout.split()[0]
        # 本机对照读数现算（不写死）：本机 .venv 跑同一条 invariants，取违反名单
        loc = subprocess.run([str(LOCAL_ROOT / ".venv/bin/python"), "scripts/check_pipeline_invariants.py"], cwd=LOCAL_ROOT,
                             capture_output=True, text=True, env={**os.environ, "SYNCOPATE_CONTRACT": "v15", "SYNCOPATE_THINK": "1"})
        local_v = _violations(loc.stdout + loc.stderr)
        print(f"[git] 本机 invariants 违反 {len(local_v)} 条（作为对照集合）")
        run("git", p_git, sha, local_v)
    if "volume" in want:
        marker = f"probe-{int(time.time())}-{os.getpid()}"
        w = p_volume_write.remote(marker)
        run("volume", p_volume_read, marker)
        results["volume"]["writer_container"] = w.get("container")
    if "gpu" in want:
        run("gpu", p_gpu)
    if "nccl" in want:
        run("nccl", p_nccl)
    if "model" in want:
        run("model_download", p_model_download)
        if results.get("model_download", {}).get("ok"):
            run("model_smoke", p_model_smoke)
    if "rebuild" in want:
        run("rebuild_v13", p_rebuild_v13)
    if "pytest" in want:
        run("pytest", p_pytest)

    # 汇总：本机也落一份
    out_dir = LOCAL_ROOT / "_audit" / "modal_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M")
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\n══════ Modal 搬家探针 · 汇总 ══════")
    for k, v in results.items():
        mark = "✅" if v.get("ok") else ("⚠️ 没跑成" if v.get("ok") is None else "🔴")
        print(f"  {k:<16} {mark}")
    if results.get("nccl", {}).get("nccl_2card"):
        print("  NCCL 双卡各变体：" + "  ".join(f"{k}={'通' if v['pass'] else ('挂死' if v['hang'] else '红')}" for k, v in results["nccl"]["nccl_2card"].items()))
    print(f"  耗时 {round(time.time() - t0)} s · 明细 {out_dir / f'summary_{stamp}.json'} · Volume 上 {AUDIT}/")
    red = [k for k, v in results.items() if v.get("ok") is False]
    unrun = [k for k, v in results.items() if v.get("ok") is None]
    sys.exit(1 if red else (2 if unrun else 0))

"""新栈探针（Chaoyu 09-03 裁定⑪ 换法三）：vLLM 0.28 / torch 2.13 cu13 / verl 0.9 / transformers 5.10 / FLA 0.5.2 /
flash-attn 用社区预编译轮子（mjun0812 v0.9.47 cu130torch2.13，写在 stack/pyproject.toml 的 [tool.uv.sources]），
**09-03 晚裁定⑫：一切在 B200（sm_100）上配**，学生候选 Qwen3.6-35B-A3B（最新小 MoE）、思考教师 Qwen3.8-27B（最新密集）。
**目的是学新栈**，每步既是判据也是一份读数。新增：versions（守则⑯ 最新稳定版核对）· fa4（CuTe DSL）· nccl（NVLink 带宽）· vllm_ep（EP=2）。

    modal run --detach modal_app/stack_probe.py --steps image,verl     # ① 镜像 + 导入判据 + verl 0.9 结构（CPU）
    modal run --detach modal_app/stack_probe.py --steps models         # ② Qwen3.5 全家权重落 Volume（CPU，~80 GiB）
    modal run --detach modal_app/stack_probe.py --steps gpu            # ③ 单卡：flash-attn 反向（社区轮子的闸）· FLA GDN 三实现对拍 · 拓扑
    modal run --detach modal_app/stack_probe.py --steps vllm           # ④ 单卡：vLLM 起 9B，MTP 关/开各测 TPOT，抓核选择日志
    modal run --detach modal_app/stack_probe.py                        # 默认 = image,verl,models,gpu,vllm 顺序全跑

依赖表 = modal_app/stack/（与根目录旧栈分家）。旧栈探针 modal_app/probe.py 保留只作历史读数。

★ 判据先注册（守则⑬）
  image   torch/vllm/verl/transformers/fla/flash_attn/peft 全部 import 退出码 0；torch.version.cuda 以 "13" 开头；
          transformers CONFIG_MAPPING 含 "qwen3_5"
  verl    import 退出码 0；trainer 入口 --help 能打；记录 V1 trainer 三种模式的模块名（学习项）
  models  每个模型 safetensors 总字节 == HF 仓库声明
  gpu     capability == (12,0)；check_flash_attn_backward 退出码 0；
          FLA：chunk（训练核）前向+反向 vs naive fp32 参考：前向相对误差 ≤ 5e-2（bf16 噪声地板）、梯度有限非零且相对差 ≤ 5e-2；
          fused_recurrent（解码核，FLA 不实现反向）只测前向 ≤ 5e-2
  vllm    MTP 关/开两种配置都能起服务并给出非空回答；TPOT 比值只记录（08 §Modal 先验：27B 上 MTP 反慢 3.6×）；
          日志里抓 GDN/fused/Triton/attention backend 的行 = 学习项
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time

import modal

APP_NAME = "syncopate-stack2"
REPO_URL = "https://github.com/ChaoyuWang04/Syncopate_Async_AgenticRL.git"
REPO_BRANCH = "main"
VOL_NAME = "syncopate-home"
VOL = "/vol"
REPO = "/tmp/repo"                    # ★ 每个容器自己的 checkout（守则⑰ 一个写者）；bare 镜像在 /vol/repo.git
MODELS = f"{VOL}/models"
AUDIT = f"{VOL}/_audit/stack_probe"
GPU_ONE = "B200"          # Chaoyu 09-03 晚裁定⑫：一切在 B200（sm_100）上配；B300 待 B200 全链通后重跑
GPU_PAIR = "B200:2"
BASE_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"   # torch 2.13 PyPI 默认 cu13；docker hub 实核存在（09-03）
PY = "/env/.venv/bin/python"
HF_MODELS = {   # 09-03 晚裁定⑫：最新模型。Qwen3.5-9B/27B 已在 Volume 留作对照，不再列
    "Qwen/Qwen3.6-35B-A3B": f"{MODELS}/Qwen3.6-35B-A3B",   # 学生候选：最新小 MoE（67 GiB，qwen3_5_moe）
    "Qwen/Qwen3.8-27B": f"{MODELS}/Qwen3.8-27B",           # 思考教师 + 人话教师（裁定⑬：教师只要装得下就用大的；3.5-4B 退役）
    "Qwen/Qwen3.5-0.8B": f"{MODELS}/Qwen3.5-0.8B",         # 测试分词（1.6 GiB）
}
TEACHER = f"{MODELS}/Qwen3.8-27B"
# 09-04：服务侧上限 = rollout_budget.MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH（12288+12288）。本文件本机也要 import 所以写常量，
#   容器内各服务步先 assert 与 rollout_budget 相等（守则①「两个东西应当相同」），漂了直接红。
SERVE_MAX_MODEL_LEN = 24576
STUDENT = f"{MODELS}/Qwen3.6-35B-A3B"

LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
STACK = LOCAL_ROOT / "modal_app" / "stack"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
# wandb：secret `wandb-secret` 由 Chaoyu 在控制台建（09-03）；p_wandb 是接线判据（key 在容器里 + 在线写 3 点读回 3 点）。
# ⛔ 不许按环境变量条件定义 Modal 对象：本机与容器里求值不同 ⇒ "Function has 3 dependencies but container got 2"
#   （09-03 run13 就是这么在 hydrate 阶段直接死的）。
SECRETS = [modal.Secret.from_name("wandb-secret")]   # Chaoyu 09-03 22:28 建的（Modal 控制台 wandb 模板，键 WANDB_API_KEY）

_ENV = {
    "PATH": "/env/.venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",   # venv/bin 必须在：flashinfer JIT 子进程找 ninja
    "CUDA_HOME": "/usr/local/cuda",
    "UV_PYTHON": "3.12",
    "UV_LINK_MODE": "copy",
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "PYTHONUNBUFFERED": "1",
}

# 镜像：依赖表三件套 + uv sync（flash-attn 轮子在锁文件里，随 sync 装）
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    # postgresql-16 / redis-server：考场链与 runtime 测试的底座（08 §1.2 Modal 版：系统包 + 仓库 bootstrap 脚本按 env 指路径）
    .apt_install("git", "curl", "build-essential", "ca-certificates", "postgresql-16", "postgresql-client-16", "redis-server")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(_ENV)
    .add_local_file(STACK / "pyproject.toml", "/env/pyproject.toml", copy=True)
    .add_local_file(STACK / "uv.lock", "/env/uv.lock", copy=True)
    .add_local_file(STACK / "uv.toml", "/env/uv.toml", copy=True)
    .run_commands("cd /env && uv sync --frozen --all-extras")
    # FA4 单独 venv（与主栈的 apache-tvm-ffi 钉冲突，见 stack/pyproject.toml）；只给 fa4 探针与核实验用
    .run_commands(
        "cd /env && uv venv --python 3.12 .venv-fa4 && "
        "uv pip install --python .venv-fa4/bin/python 'torch==2.13.0' 'flash-attn-4[cu13]==4.0.0b29' einops "
        "'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3%2Bcu130torch2.13-cp312-cp312-linux_x86_64.whl'"
    )
)
PY_FA4 = "/env/.venv-fa4/bin/python"

RUN_ENV = {"PYTHONPATH": REPO, "SYNCOPATE_CONTRACT": "v15", "SYNCOPATE_THINK": "1", "HF_HOME": f"{VOL}/hf_cache",
           # FlashInfer 在 sm_100 上首启会 JIT 编 TRTLLM MoE/GEMM 核（gcc 重、十几分钟）+ 从 NVIDIA 下 cubin；vLLM 还有 torch.compile 缓存
           # ⇒ 两者都落 Volume，编一次永久复用（09-03 B200 实测：不落盘 ⇒ 单卡 15 min 起不来、双卡 gcc 段错误）
           "FLASHINFER_WORKSPACE_BASE": f"{VOL}/flashinfer_cache", "VLLM_CACHE_ROOT": f"{VOL}/vllm_cache"}


def _sh(cmd: str, *, cwd: str | None = None, env: dict | None = None, timeout: int | None = None) -> dict:
    e = dict(os.environ); e.update(RUN_ENV)
    if env: e.update(env)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": (p.stdout + p.stderr)[-8000:], "secs": round(time.time() - t0, 1), "timed_out": False}
    except subprocess.TimeoutExpired as ex:
        out = ((ex.stdout or b"").decode(errors="replace") + (ex.stderr or b"").decode(errors="replace"))[-8000:]
        return {"rc": -1, "out": out, "secs": round(time.time() - t0, 1), "timed_out": True}


def _record(step: str, ok: bool, details: dict) -> dict:
    rec = {"step": step, "ok": bool(ok), "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **details}
    os.makedirs(AUDIT, exist_ok=True)
    json.dump(rec, open(f"{AUDIT}/{step}.json", "w"), ensure_ascii=False, indent=1)
    vol.commit()
    print(f"[{step}] {'✅' if ok else '🔴'}")
    return rec


REPO_MIRROR = f"{VOL}/repo.git"      # 共享的 bare 镜像（Volume 上，只有 fetch 写它，带锁）
# ⚠️ REPO 常量必须在文件顶部定义：RUN_ENV 在定义时就把 PYTHONPATH 固化了，09-04 曾因在这里重赋值导致容器里
#   cwd=/tmp/repo 而 PYTHONPATH=/vol/repo ⇒ 跑的是旧代码（探针回溯里的 BatchEncoding 错就是这么来的）
DATA_DIRS = ("batches", "sft", "rl")   # gitignored 的数据目录整体指回 Volume（跨 run 共享、按版本分目录）；
                                        # data/splits 在 git 里（v16 切分 4 个 json 已入库，确定性判据过后即"源码"），不软链


def _sync_repo() -> str:
    """更新 Volume 上的 bare 镜像（flock + 重试），再在本容器 /tmp 里 clone 出工作树；models/data 指回 Volume。返回 HEAD sha。"""
    if not os.path.isdir(REPO_MIRROR):
        r = _sh(f"flock -w 600 {VOL}/.repo.lock git clone --bare --branch {REPO_BRANCH} {REPO_URL} {REPO_MIRROR}", timeout=900)
        if r["rc"] != 0 and not os.path.isdir(REPO_MIRROR): raise RuntimeError("bare clone 失败：" + r["out"])
    r = _sh(f"flock -w 600 {VOL}/.repo.lock git --git-dir {REPO_MIRROR} fetch -q origin +{REPO_BRANCH}:{REPO_BRANCH}", timeout=900)
    if r["rc"] != 0: raise RuntimeError("git fetch 失败：" + r["out"])
    vol.commit()
    if not os.path.isdir(f"{REPO}/.git"):
        r = _sh(f"git clone -q --shared --branch {REPO_BRANCH} {REPO_MIRROR} {REPO}", timeout=600)
        if r["rc"] != 0: raise RuntimeError("worktree clone 失败：" + r["out"])
    else:
        _sh(f"git fetch -q origin {REPO_BRANCH} && git reset -q --hard origin/{REPO_BRANCH}", cwd=REPO, timeout=300)
    # 模型/数据指回 Volume（只读共享；数据按版本分目录，写者是各自的建库步）
    os.makedirs(f"{VOL}/data", exist_ok=True)
    for d in DATA_DIRS:
        os.makedirs(f"{VOL}/data/{d}", exist_ok=True)
        _sh(f"rm -rf {REPO}/data/{d} && ln -sfn {VOL}/data/{d} {REPO}/data/{d}")
    _sh(f"rm -rf {REPO}/models && ln -sfn {MODELS} {REPO}/models")
    os.makedirs(f"{VOL}/checkpoints", exist_ok=True); os.makedirs(f"{MODELS}/adapters", exist_ok=True)
    _sh(f"rm -rf {REPO}/checkpoints && ln -sfn {VOL}/checkpoints {REPO}/checkpoints")   # runbook 产物跨 run 留存
    # 项目以可编辑方式装进 venv（vLLM 插件入口点需要"装过"；--no-deps 不动锁）
    r = _sh(f"uv pip install --python {PY} --no-deps -e {REPO}", timeout=300)
    if r["rc"] != 0: raise RuntimeError("editable install 失败：" + r["out"][-600:])
    return _sh("git rev-parse HEAD", cwd=REPO)["out"].strip()


def _topology() -> dict:
    return {
        "modal_env": {k: v for k, v in os.environ.items() if k.startswith("MODAL_") and "TOKEN" not in k and "PATH" not in k},
        "cpu": _sh("lscpu | grep -E 'Model name|^NUMA|Socket|^CPU\\(s\\)'")["out"].strip(),
        "mem_gb": _sh("free -g | awk '/Mem:/{print $2}'")["out"].strip(),
        "nvidia_smi": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")["out"].strip(),
        "topo": _sh("nvidia-smi topo -m 2>&1 | head -12")["out"].strip(),
        "nvlink": _sh("nvidia-smi nvlink -s 2>&1 | head -24")["out"].strip(),
    }


# ─────────────────────────── ② 镜像导入判据 ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=900)
def p_image() -> dict:
    code = r'''
import json, torch, vllm, verl, transformers, fla, flash_attn, peft, tensordict, triton
from transformers import CONFIG_MAPPING
json.dump({
  "torch": torch.__version__, "torch_cuda": torch.version.cuda, "vllm": vllm.__version__, "verl": verl.__version__,
  "transformers": transformers.__version__, "fla": getattr(fla, "__version__", "?"), "flash_attn": flash_attn.__version__,
  "peft": peft.__version__, "tensordict": tensordict.__version__, "triton": triton.__version__,
  "qwen3_5_in_config_mapping": "qwen3_5" in CONFIG_MAPPING,
  "qwen_archs": sorted(k for k in CONFIG_MAPPING if k.startswith("qwen3")),
}, open("/tmp/imp_result.json", "w"))   # 结果走文件：stderr 的警告行会混进 stdout 尾部
'''
    open("/tmp/imp.py", "w").write(code)
    r = _sh(f"{PY} /tmp/imp.py", timeout=600)
    info = {}
    try: info = json.load(open("/tmp/imp_result.json"))
    except Exception: pass
    nvcc = _sh("nvcc --version | tail -1")["out"].strip()
    ok = r["rc"] == 0 and str(info.get("torch_cuda", "")).startswith("13") and info.get("qwen3_5_in_config_mapping") is True
    return _record("image", ok, {"versions": info, "nvcc": nvcc, "run": {k: r[k] for k in ("rc", "secs")}, "tail": r["out"][-1500:]})


# ─────────────────────────── ③ verl 0.9 结构探查（学习项） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=900)
def p_verl() -> dict:
    code = r'''
import json, pkgutil, importlib, verl, verl.trainer as T
mods = sorted(m.name for m in pkgutil.walk_packages(T.__path__, T.__name__ + ".") )
keys = [m for m in mods if any(k in m.lower() for k in ("v1", "unified", "async", "colocate", "separate", "distill", "fsdp", "main_"))]
out = {"verl": verl.__version__, "trainer_modules_n": len(mods), "interesting": keys[:60]}
for cand in ("verl.trainer.main_ppo", "verl.trainer.v1.main", "verl.trainer.main"):
    try: importlib.import_module(cand); out.setdefault("importable", []).append(cand)
    except Exception as e: out.setdefault("import_errors", {})[cand] = repr(e)[:200]
json.dump(out, open("/tmp/verl_result.json", "w"))
'''
    open("/tmp/verl_probe.py", "w").write(code)
    r = _sh(f"{PY} /tmp/verl_probe.py", timeout=600)
    h = _sh(f"{PY} -m verl.trainer.main_ppo --help 2>&1 | head -60", timeout=600)
    info = {}
    try: info = json.load(open("/tmp/verl_result.json"))
    except Exception: pass
    return _record("verl", r["rc"] == 0 and bool(info), {"info": info, "main_ppo_help_head": h["out"][-2500:], "tail": r["out"][-1200:]})


# ─────────────────────────── ④ 权重 ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=3 * 3600)
def p_models(only: str = "") -> dict:
    want = {k: v for k, v in HF_MODELS.items() if not only or any(o in k for o in only.split(","))}
    code = r'''
import json, os, sys
from huggingface_hub import snapshot_download, HfApi
out = {}
for repo_id, local in json.loads(sys.argv[1]).items():
    snapshot_download(repo_id, local_dir=local, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja", "merges.txt", "vocab.json"])
    info = HfApi().model_info(repo_id, files_metadata=True)
    want = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    got = sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local) if f.endswith(".safetensors"))
    cfg = json.load(open(os.path.join(local, "config.json")))
    out[repo_id] = {"local": local, "bytes_hf": want, "bytes_vol": got, "same": want == got,
                    "arch": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                    "text_cfg_keys": sorted(k for k in (cfg.get("text_config") or cfg) if "linear" in k or "layer" in k or "mtp" in k)[:20]}
json.dump(out, open("/tmp/dl_result.json", "w"))
'''
    open("/tmp/dl.py", "w").write(code)
    r = _sh(f"{PY} /tmp/dl.py '{json.dumps(want)}'", timeout=3 * 3600 - 300)
    vol.commit()
    res = {}
    try: res = json.load(open("/tmp/dl_result.json"))
    except Exception: pass
    ok = r["rc"] == 0 and bool(res) and all(v["same"] for v in res.values())
    return _record("models", ok, {"models": res, "run": {k: r[k] for k in ("rc", "secs")}, "tail": r["out"][-800:]})


# ─────────────────────────── ⑤ 单卡：flash-attn 反向 + FLA GDN 对拍 ───────────────────────────
_FLA_PARITY = r'''
import json, time, torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule, naive_recurrent_gated_delta_rule
torch.manual_seed(0); dev = "cuda"
def mk(B, T, H, D, dt):
    q = torch.randn(B, T, H, D, device=dev, dtype=dt); k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=dev, dtype=dt), dim=-1)
    v = torch.randn(B, T, H, D, device=dev, dtype=dt)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))   # log-decay ≤ 0
    beta = torch.rand(B, T, H, device=dev, dtype=dt)
    return [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
out = {}
B, T, H, D = 2, 256, 4, 128
ref_in = mk(B, T, H, D, torch.float32)
Q, K, V, G, B = ref_in
o_ref, _ = naive_recurrent_gated_delta_rule(q=Q, k=K, v=V, g=G, beta=B); o_ref.sum().backward()   # 关键字传参：naive 与 chunk 的位置顺序不同（09-03 踩过）
ref_grads = [t.grad.float() for t in ref_in]
out["ref_finite"] = bool(torch.isfinite(o_ref).all()) and all(bool(torch.isfinite(g).all()) for g in ref_grads)
# chunk = 训练核（前向+反向）；fused_recurrent = 解码核，FLA 明说**不实现反向**（09-03 实测 NotImplementedError）⇒ 只测前向
for name, fn, with_bwd in (("chunk", chunk_gated_delta_rule, True), ("fused_recurrent", fused_recurrent_gated_delta_rule, False)):
    ins = [t.detach().clone().to(torch.bfloat16 if i != 3 else torch.float32).requires_grad_(with_bwd) for i, t in enumerate(ref_in)]
    kw = dict(zip(("q", "k", "v", "g", "beta"), ins))
    if with_bwd:
        o, _ = fn(**kw); o.float().sum().backward()
        gs = [t.grad.float() for t in ins]
        grel = [((g - r).abs().max() / (r.abs().max() + 1e-6)).item() for g, r in zip(gs, ref_grads)]
        extra = {"grad_rel_err_qkvgb": [round(x, 4) for x in grel],
                 "grad_finite": all(bool(torch.isfinite(g).all()) for g in gs), "grad_nonzero": all(g.norm().item() > 0 for g in gs)}
    else:
        with torch.no_grad(): o, _ = fn(**kw)
        extra = {"backward": "not implemented in FLA (decode-only kernel)"}
    rel = ((o.float() - o_ref).abs().max() / o_ref.abs().max()).item()
    out[name] = {"fwd_rel_err": round(rel, 5), **extra}
# 速度（学习项）：T=4096 前向+反向
ins = [t.detach().clone().to(torch.bfloat16 if i != 3 else torch.float32).requires_grad_(True) for i, t in enumerate(mk(1, 4096, 8, 128, torch.float32))]
for _ in range(2): o, _ = chunk_gated_delta_rule(*ins); o.float().sum().backward()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(5): o, _ = chunk_gated_delta_rule(*ins); o.float().sum().backward()
torch.cuda.synchronize(); out["chunk_T4096_H8_fwdbwd_ms"] = round((time.perf_counter() - t0) / 5 * 1e3, 2)
print("FLA_RESULT " + json.dumps(out))
'''


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=8, memory=32768, timeout=1800)
def p_gpu() -> dict:
    _sync_repo()
    tor = _sh(f"{PY} -c \"import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())], torch.version.cuda)\"", timeout=300)
    cap_ok = tor["rc"] == 0 and tor["out"].strip().startswith("1 [(10, 0)]")   # B200 = sm_100；换卡改这里
    fa = _sh(f"{PY} scripts/check_flash_attn_backward.py", cwd=REPO, timeout=600)
    open("/tmp/fla_parity.py", "w").write(_FLA_PARITY)
    fl = _sh(f"{PY} /tmp/fla_parity.py", timeout=900)
    fla_res = {}
    for line in fl["out"].splitlines():
        if line.startswith("FLA_RESULT "): fla_res = json.loads(line[len("FLA_RESULT "):])
    c, f = fla_res.get("chunk", {}), fla_res.get("fused_recurrent", {})
    fla_ok = fla_res.get("ref_finite") is True and bool(c) and bool(f) and c["fwd_rel_err"] <= 5e-2 and c["grad_finite"] and c["grad_nonzero"] \
        and max(c["grad_rel_err_qkvgb"]) <= 5e-2 and f["fwd_rel_err"] <= 5e-2
    ok = cap_ok and fa["rc"] == 0 and fla_ok
    return _record("gpu", ok, {"torch": tor["out"].strip(), "flash_attn_backward": {"rc": fa["rc"], "tail": fa["out"][-900:]},
                               "fla": fla_res, "fla_tail": fl["out"][-1500:] if not fla_ok else "", "topology": _topology()})



def _teardown(proc) -> None:
    """杀干净 vLLM 全家（APIServer/EngineCore/Worker 都是独立进程），并等到显存真的归零——
    09-03 实测：只杀入口进程，下一个变体起来时显存还被占着 111/178 GiB ⇒ 启动直接报错。"""
    proc.terminate()
    try: proc.wait(60)
    except Exception: proc.kill()
    _sh("pkill -9 -f 'vllm' ; pkill -9 -f 'EngineCore' ; pkill -9 -f 'multiproc_executor' ; sleep 3")
    for _ in range(40):
        used = _sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")["out"].strip().splitlines()
        if used and all(int(u) < 2048 for u in used): return
        time.sleep(3)

# ─────────────────────────── ⑥ 单卡：vLLM 起 9B，MTP 关/开 ───────────────────────────
_BENCH = r'''
import json, sys, time, urllib.request
port = sys.argv[1]; res = []
def call(n):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "投放预算怎么定？请分三点说明。"}], "max_tokens": n, "temperature": 0}).encode()
    t0 = time.perf_counter(); r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    d = json.load(r); dt = time.perf_counter() - t0
    return dt, d["usage"]["completion_tokens"], d["choices"][0]["message"]["content"]
call(8)  # 预热
for _ in range(3):
    dt, n, txt = call(160); res.append((dt, n))
ms = sorted(dt / max(n, 1) * 1e3 for dt, n in res)[1]
print("BENCH " + json.dumps({"ms_per_token_median": round(ms, 2), "tokens": [n for _, n in res], "sample": txt[:120]}))
'''


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=32, memory=131072, timeout=3600 * 2)
def p_vllm() -> dict:
    vol.reload()
    model = STUDENT
    open("/tmp/bench.py", "w").write(_BENCH)
    base = (f"{PY} -m vllm.entrypoints.openai.api_server --model {model} --served-model-name m --max-model-len {SERVE_MAX_MODEL_LEN} "
            f"--gpu-memory-utilization 0.85 --port 8300 --limit-mm-per-prompt '{{\"image\": 0, \"video\": 0}}'")
    variants = {"mtp_off": base, "mtp_on": base + " --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 2}'"}
    out = {}
    for name, cmd in variants.items():
        log = f"/tmp/vllm_{name}.log"
        proc = subprocess.Popen(f"exec {cmd} > {log} 2>&1", shell=True, env={**os.environ, **RUN_ENV})
        up = False; t0 = time.time()
        while time.time() - t0 < 2400 and proc.poll() is None:
            if _sh("curl -sf http://127.0.0.1:8300/health")["rc"] == 0: up = True; break
            time.sleep(5)
        rec = {"up": up, "startup_secs": round(time.time() - t0, 1), "exited_early": proc.poll() is not None}
        if up:
            b = _sh(f"{PY} /tmp/bench.py 8300", timeout=900)
            for line in b["out"].splitlines():
                if line.startswith("BENCH "): rec["bench"] = json.loads(line[6:])
            rec["bench_rc"] = b["rc"]
        logtxt = open(log, errors="replace").read()
        os.makedirs(AUDIT, exist_ok=True); open(f"{AUDIT}/vllm_{name}.log", "w").write(logtxt)   # 完整日志留档
        lines = logtxt.splitlines()
        idx = [i for i, l in enumerate(lines) if "Error" in l or "error:" in l.lower() or "Traceback" in l]
        rec["root_cause"] = [l[-240:] for l in lines[max(0, idx[0] - 5): idx[0] + 40]] if idx else []
        rec["kernel_lines"] = [l[-220:] for l in logtxt.splitlines() if any(k in l.lower() for k in ("gdn", "gated", "fused", "triton", "attention backend", "spec", "mtp", "flashinfer", "cuda graph"))][:40]
        rec["log_tail"] = logtxt[-1500:] if not up or "bench" not in rec else ""
        _teardown(proc)
        rec["gpu_mem_after_teardown_MiB"] = _sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")["out"].strip()
        vol.commit()   # JIT/compile 缓存落盘
        out[name] = rec
    ok = all(v.get("up") and v.get("bench", {}).get("sample") for v in out.values())
    if all("bench" in v for v in out.values()):
        out["mtp_on_over_off_ms_ratio"] = round(out["mtp_on"]["bench"]["ms_per_token_median"] / out["mtp_off"]["bench"]["ms_per_token_median"], 3)
    return _record("vllm", ok, {**out, "topology": _topology()})



# ─────────────────────────── ★ 版本核对（守则⑯：装的是不是最新稳定版，不是要说明原因） ───────────────────────────
# 白名单 = 「不是最新、且有原因」的登记表；不在表里又不是最新 ⇒ 判红（逼着把原因写下来）
NOT_LATEST_ALLOWED = {
    "torch": "vllm 0.28.0 钉 torch==2.13.0（PyPI 最新 2.14.0）",
    "transformers": "verl 0.9.0 要求 <5.11（PyPI 最新 5.16.x）",
    "tensordict": "verl 0.9.0 要求 <=0.10.0",
    "numpy": "解析器按 verl/vllm 约束选的，非人为钉",
    "triton": "随 torch 2.13 钉 3.7.x",
    "flashinfer-python": "vllm 0.28.0 钉 0.6.16.post3",
    "flash-attn": "社区预编译轮子（mjun0812）只有 2.8.3；PyPI 的 2.8.3.post1 只是打包修正，核代码相同",
    "nvidia-cutlass-dsl": "vllm 0.28.0 钉 [cu13]==4.6.2（PyPI 最新 4.7.1）",
    "flashinfer-jit-cache": "必须与 flashinfer-python 同版本（vllm 钉 0.6.16.post3）",
    "flashinfer-cubin": "必须与 flashinfer-python 同版本（vllm 钉 0.6.16.post3）",
}
VERSION_WATCH = ["torch", "vllm", "verl", "transformers", "flash-linear-attention", "flash-attn", "flashinfer-jit-cache", "flashinfer-cubin",
                 "flashinfer-python", "triton", "peft", "tensordict", "numpy", "nvidia-cutlass-dsl", "huggingface-hub", "ray"]


@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=900)
def p_versions() -> dict:
    code = r"""
import json, urllib.request, importlib.metadata as md
watch = json.loads(open('/tmp/watch.json').read())
out = {}
for name in watch:
    try: inst = md.version(name)
    except Exception: inst = None
    try:
        d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30))
        latest = d["info"]["version"]
    except Exception as e: latest = f"?{e!r}"[:60]
    out[name] = {"installed": inst, "latest_stable": latest}
json.dump(out, open("/tmp/versions.json", "w"))
"""
    open("/tmp/watch.json", "w").write(json.dumps(VERSION_WATCH)); open("/tmp/vers.py", "w").write(code)
    r = _sh(f"{PY} /tmp/vers.py", timeout=600)
    info = {}
    try: info = json.load(open("/tmp/versions.json"))
    except Exception: pass
    def base(v): return (v or "").split("+")[0]
    rows = {}
    for k, v in info.items():
        is_latest = base(v["installed"]) == v["latest_stable"]
        rows[k] = {**v, "is_latest": is_latest, "reason": None if is_latest else NOT_LATEST_ALLOWED.get(k)}
    unexplained = [k for k, v in rows.items() if not v["is_latest"] and not v["reason"]]
    return _record("versions", r["rc"] == 0 and bool(rows) and not unexplained, {"rows": rows, "unexplained_not_latest": unexplained})


# ─────────────────────────── FA4（只在 sm_100+ 有意义） ───────────────────────────
_FA4 = r"""
import json, time, torch, importlib
res = {"import_path": None}
fn = None
for path in ("flash_attn.cute.interface", "flash_attn.cute", "flash_attn_4", "flash_attn_4.interface"):
    try:
        m = importlib.import_module(path); fn = getattr(m, "flash_attn_func", None)
        if fn: res["import_path"] = path; break
    except Exception as e: res.setdefault("import_errors", {})[path] = repr(e)[:160]
if fn is None: print("FA4_RESULT " + json.dumps(res)); raise SystemExit(0)
torch.manual_seed(0); dev, dt = "cuda", torch.bfloat16
B, S, H, D = 2, 2048, 16, 128
q, k, v = (torch.randn(B, S, H, D, device=dev, dtype=dt, requires_grad=True) for _ in range(3))
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
o = fn(q, k, v, causal=True); o = o[0] if isinstance(o, tuple) else o
res["fwd_rel_err"] = ((o.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
try:
    o.float().sum().backward(); gs = [t.grad for t in (q, k, v)]
    res["bwd"] = {"finite": all(bool(torch.isfinite(g).all()) for g in gs), "nonzero": all(g.norm().item() > 0 for g in gs)}
except Exception as e: res["bwd"] = {"error": repr(e)[:200]}
def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
with torch.no_grad():
    Sb = 8192; qb, kb, vb = (torch.randn(1, Sb, H, D, device=dev, dtype=dt) for _ in range(3))
    res["fa4_fwd_ms_S8192"] = round(bench(lambda: fn(qb, kb, vb, causal=True)), 3)
    try:
        from flash_attn import flash_attn_func as fa2
        res["fa2_fwd_ms_S8192"] = round(bench(lambda: fa2(qb, kb, vb, causal=True)), 3)
        res["fa4_over_fa2_speedup"] = round(res["fa2_fwd_ms_S8192"] / res["fa4_fwd_ms_S8192"], 2)
    except Exception as e: res["fa2"] = repr(e)[:200]
    flops = 4 * Sb * Sb * H * D / 2
    res["fa4_tflops_S8192"] = round(flops / (res["fa4_fwd_ms_S8192"] / 1e3) / 1e12, 1)
print("FA4_RESULT " + json.dumps(res))
"""


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=8, memory=32768, timeout=1500)
def p_fa4() -> dict:
    """FA4（CuTe DSL）在 B200 上：能 import · 前向 vs torch sdpa 相对误差 ≤ 5e-2 · 反向有限非零（若实现）· 与 FA2 的前向速度比（学习项）。"""
    open("/tmp/fa4.py", "w").write(_FA4)
    r = _sh(f"{PY_FA4} /tmp/fa4.py", timeout=1200)
    vers = _sh(f"{PY_FA4} -c \"import importlib.metadata as m; print({{k: m.version(k) for k in ('flash-attn-4','nvidia-cutlass-dsl','apache-tvm-ffi','torch')}})\"", timeout=120)["out"].strip()
    res = {}
    for line in r["out"].splitlines():
        if line.startswith("FA4_RESULT "): res = json.loads(line[11:])
    ok = bool(res.get("import_path")) and res.get("fwd_rel_err", 1) <= 5e-2
    return _record("fa4", ok, {"fa4": res, "venv_fa4_versions": vers, "tail": "" if ok else r["out"][-1500:], "topology": _topology()})


# ─────────────────────────── 双卡 NCCL（NVLink） ───────────────────────────
_NCCL = r"""
import os, time, json, torch, torch.distributed as dist, torch.multiprocessing as mp
def w(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29721"
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=2)
    out = {}
    for mb in (16, 256, 1024):
        n = mb * (1 << 20) // 4
        x = torch.ones(n, device=f"cuda:{rank}")
        for _ in range(3): dist.all_reduce(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10): dist.all_reduce(x)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
        src = torch.ones(n // 2, device=f"cuda:{rank}"); dst = torch.zeros(n, device=f"cuda:{rank}")
        for _ in range(3): dist.all_gather_into_tensor(dst, src)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        for _ in range(10): dist.all_gather_into_tensor(dst, src)
        torch.cuda.synchronize(); dt2 = (time.perf_counter() - t1) / 10
        out[f"{mb}MB"] = {"all_reduce_busbw_GBps": round(n * 4 / dt / 1e9, 1), "all_gather_algbw_GBps": round(n * 4 / dt2 / 1e9, 1)}
    if rank == 0: print("NCCL_RESULT " + json.dumps(out))
    dist.destroy_process_group()
if __name__ == "__main__":   # spawn 会重新 import 主模块，没有守卫 ⇒ freeze_support 报错（09-03 B200 实测）
    mp.spawn(w, nprocs=2, join=True)
"""


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_PAIR, cpu=8, memory=32768, timeout=1200)
def p_nccl() -> dict:
    """两张 B200：都是 sm_100；NVLink 状态；all_reduce/all_gather 三种消息大小的带宽（学习项，对照 4×5090 PCIe 的 25.6 GB/s）。"""
    tor = _sh(f"{PY} -c \"import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())], torch.cuda.can_device_access_peer(0,1))\"", timeout=300)
    open("/tmp/nccl2.py", "w").write(_NCCL)
    r = _sh(f"{PY} /tmp/nccl2.py", timeout=300)
    res = {}
    for line in r["out"].splitlines():
        if line.startswith("NCCL_RESULT "): res = json.loads(line[12:])
    ok = tor["out"].strip().startswith("2 [(10, 0), (10, 0)]") and bool(res) and not r["timed_out"]
    return _record("nccl", ok, {"torch": tor["out"].strip(), "bw": res, "hang": r["timed_out"], "tail": "" if ok else r["out"][-1200:], "topology": _topology()})


# ─────────────────────────── 双卡 vLLM 专家并行（EP=2） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_PAIR, cpu=32, memory=131072, timeout=3600 * 2)
def p_vllm_ep() -> dict:
    """两张 B200 起 Qwen3.6-35B-A3B：--data-parallel-size 2 --enable-expert-parallel（EP = TP×DP = 2）；判据 = 起得来 + 回答非空；
    学习项 = all2all 后端/MoE 核/attention 后端日志行 + 每 token 耗时（对照单卡）。"""
    vol.reload()
    open("/tmp/bench.py", "w").write(_BENCH)
    cmd = (f"{PY} -m vllm.entrypoints.openai.api_server --model {STUDENT} --served-model-name m --max-model-len {SERVE_MAX_MODEL_LEN} "
           f"--gpu-memory-utilization 0.85 --port 8300 --limit-mm-per-prompt '{{\"image\": 0, \"video\": 0}}' "
           f"--data-parallel-size 2 --enable-expert-parallel")
    log = "/tmp/vllm_ep.log"
    proc = subprocess.Popen(f"exec {cmd} > {log} 2>&1", shell=True, env={**os.environ, **RUN_ENV})
    up = False; t0 = time.time()
    while time.time() - t0 < 2400 and proc.poll() is None:
        if _sh("curl -sf http://127.0.0.1:8300/health")["rc"] == 0: up = True; break
        time.sleep(5)
    rec = {"up": up, "startup_secs": round(time.time() - t0, 1), "exited_early": proc.poll() is not None}
    if up:
        b = _sh(f"{PY} /tmp/bench.py 8300", timeout=900)
        for line in b["out"].splitlines():
            if line.startswith("BENCH "): rec["bench"] = json.loads(line[6:])
    logtxt = open(log, errors="replace").read()
    os.makedirs(AUDIT, exist_ok=True); open(f"{AUDIT}/vllm_ep.log", "w").write(logtxt)
    lines = logtxt.splitlines()
    idx = [i for i, l in enumerate(lines) if "Error" in l or "Traceback" in l]
    rec["root_cause"] = [l[-240:] for l in lines[max(0, idx[0] - 5): idx[0] + 40]] if idx else []
    rec["kernel_lines"] = [l[-220:] for l in lines if any(k in l.lower() for k in ("all2all", "expert", "moe", "gdn", "fused", "attention backend", "flashinfer", "nvlink", "cuda graph", "dp rank", "eplb"))][:50]
    _teardown(proc); vol.commit()
    ok = up and bool(rec.get("bench", {}).get("sample"))
    return _record("vllm_ep", ok, {**rec, "topology": _topology()})



SERVICE_ENV = {"PG_HOME": "/usr/lib/postgresql/16", "PG_SHARE": "/usr/share/postgresql/16", "PG_LIB": "/usr/lib/x86_64-linux-gnu",
               "PGDATA": "/tmp/pgdata/16", "LOGFILE": "/tmp/pgdata/pg.log", "REDIS_HOME": "/usr", "REDIS_DIR": "/tmp/redis",
               "PYTHON": PY,   # pg_bootstrap 用它跑 alembic（默认指 $REPO/.venv，Modal 上 venv 在 /env）
               "SYNCOPATE_PG_DSN": "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate",
               "SYNCOPATE_REDIS_URL": "redis://:syncopate-dev@127.0.0.1:6379/0"}


def _start_services() -> dict:
    """容器内起 PG + Redis（都是派生产物，容器重启即丢；schema 由仓库 alembic 重建）并灌语料。判据 = 两个 bootstrap 退出码 0。"""
    pg = _sh("bash scripts/pg_bootstrap.sh", cwd=REPO, env=SERVICE_ENV, timeout=600)
    rd = _sh("bash scripts/redis_bootstrap.sh", cwd=REPO, env=SERVICE_ENV, timeout=300)
    corpus = _sh(f"{PY} scripts/ingest_corpus.py", cwd=REPO, env=SERVICE_ENV, timeout=600) if pg["rc"] == 0 else {"rc": -1, "out": "skipped"}
    return {"pg": {"rc": pg["rc"], "tail": pg["out"][-600:]}, "redis": {"rc": rd["rc"], "tail": rd["out"][-400:]}, "corpus": {"rc": corpus["rc"], "tail": corpus["out"][-300:]}}

# ─────────────────────────── 仓库测试在新栈上跑（对齐地图） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=16, memory=32768, timeout=3600)
def p_pytest(args: str = "tests -q -rfE -p no:cacheprovider", with_services: bool = True) -> dict:
    """我们自己的代码在 verl 0.9 / transformers 5.10 / vllm 0.28 下还能不能 import 与通过测试。
    本机旧栈基线 908 passed（09-02）。无 PG/Redis 的 runtime 测试会 skip/红，先只看**收集错误与失败清单**=对齐工作量地图。"""
    _sync_repo()
    r0 = _sh(f"{PY} scripts/ingest_external.py", cwd=REPO, timeout=600)      # 派生数据（27 个测试要）
    services = _start_services() if with_services else {}
    # ⚠️ 管道接 tail 会吞掉 pytest 的退出码（09-03 第一版就此误判 ✅）⇒ 输出落文件、退出码单独取
    r = _sh(f"{PY} -m pytest {args} > /tmp/pytest_out.txt 2>&1; echo PYTEST_RC=$?", cwd=REPO, env=SERVICE_ENV, timeout=3300)
    import re
    full = open("/tmp/pytest_out.txt", errors="replace").read()
    rc_m = re.search(r"PYTEST_RC=(\d+)", r["out"]); prc = int(rc_m.group(1)) if rc_m else -1
    tail = full[-12000:]
    summ = re.findall(r"^=+ (.*?) =+\s*$", full, flags=re.M)
    errors = [l for l in full.splitlines() if l.startswith("ERROR ") or l.startswith("FAILED ")]
    os.makedirs(AUDIT, exist_ok=True); open(f"{AUDIT}/pytest_full.txt", "w").write(full)
    ok = prc == 0
    rec = _record("pytest", ok, {"pytest_rc": prc, "summary": summ[-1] if summ else "", "errors_failed": errors[:80], "ingest_rc": r0["rc"], "services": services, "tail": tail[-6000:]})
    return rec


# ─────────────────────────── wandb 接线判据（B0-3） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=4096, timeout=600, secrets=SECRETS)
def p_wandb() -> dict:
    """secret 注入 ⇒ WANDB_API_KEY 在环境里；wandb.init 在线建一个 run、写 3 个点、finish；判据 = run.url 存在且 3 步都在。"""
    code = r"""
import os, json, wandb
assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY 不在环境里（secret 没注入）"
run = wandb.init(project="syncopate-b200", name="probe-wandb", mode="online", tags=["probe"])
for i in range(3): wandb.log({"probe/step": i, "probe/value": i * 0.5}, step=i)
url = run.url; run.finish()
api = wandb.Api(); r = api.run(run.path); hist = list(r.scan_history(keys=["probe/step"]))
json.dump({"url": url, "n_points": len(hist)}, open("/tmp/wandb_result.json", "w"))
"""
    open("/tmp/wandb_probe.py", "w").write(code)
    r = _sh(f"{PY} /tmp/wandb_probe.py", timeout=300)
    res = {}
    try: res = json.load(open("/tmp/wandb_result.json"))
    except Exception: pass
    ok = r["rc"] == 0 and res.get("n_points") == 3 and bool(res.get("url"))
    return _record("wandb", ok, {"wandb": res, "tail": "" if ok else r["out"][-1200:]})


# ─────────────────────────── 通用：在新栈镜像里执行一段脚本（守则⑰：核对一律进容器做） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800)
def p_exec(script: str, with_repo: bool = True) -> dict:
    """把本机传来的 python 脚本原文写进容器跑，返回 stdout/stderr 尾部。`modal run … --steps exec --exec-file path.py`"""
    if with_repo: _sync_repo()
    open("/tmp/exec_script.py", "w").write(script)
    r = _sh(f"{PY} /tmp/exec_script.py", cwd=REPO if with_repo else None, timeout=1700)
    print(r["out"][-8000:])
    return {"step": "exec", "ok": r["rc"] == 0, "rc": r["rc"], "out": r["out"][-8000:]}


# ─────────────────────────── v16 case 库在 Modal 上生成（S2 判据：与本机独立生成逐字节相同） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=16384, timeout=3600)
def p_rebuild_v16(expected_sha: str = "") -> dict:
    """configs/buckets/v16.yaml → data/batches/v16 → set_tool_menus → data/splits/v16。
    判据 = 三份切分 SHA-256 与本机独立生成的（--expected-sha "eval:...,sft:...,rl:..."）逐一相同 ⇒ 生成是确定性的、与机器无关。"""
    _sync_repo()
    out = f"{REPO}/data"
    steps = [
        ("0 external", f"{PY} scripts/make_test_external_data.py && {PY} scripts/ingest_external.py"),
        ("1 generate", f"{PY} -m syncopate cases generate --spec configs/buckets/v16.yaml --out {out}/batches/v16"),
        ("2 menus", f"{PY} scripts/set_tool_menus.py --batch {out}/batches/v16 --sft-audit _audit/v8_sft_epoch1.json"),
        ("3 split", f"{PY} -m syncopate data split --batch {out}/batches/v16 --out {out}/splits/v16"),
    ]
    log = {}
    for name, cmd in steps:
        r = _sh(f"rm -rf {out}/batches/v16 {out}/splits/v16 >/dev/null 2>&1; true" if False else cmd, cwd=REPO, timeout=2400)
        log[name] = {"rc": r["rc"], "secs": r["secs"], "tail": r["out"][-400:]}
        if r["rc"] != 0:
            return _record("rebuild_v16", False, {"failed_at": name, "log": log})
    sha = {k: _sh(f"sha256sum {out}/splits/v16/{k}_cases.json | cut -d' ' -f1")["out"].strip() for k in ("eval", "sft", "rl")}
    exp = dict(kv.split(":", 1) for kv in expected_sha.split(",") if ":" in kv)
    same = {k: (exp.get(k) == v) for k, v in sha.items()} if exp else {}
    nfiles = _sh(f"find {out}/batches/v16 -type f | wc -l")["out"].strip()
    vol.commit()
    ok = bool(exp) and all(same.values())
    return _record("rebuild_v16", ok, {"sha_modal": sha, "sha_expected": exp, "same": same, "batch_files": nfiles, "log": log})


TEACHER_ENV = {**RUN_ENV, "SYNCOPATE_TEACHER_LANG_URL": "http://127.0.0.1:8210/v1", "SYNCOPATE_TEACHER_THINK_URL": "http://127.0.0.1:8210/v1"}


def _assert_serve_len() -> None:
    """服务侧 max_model_len 必须 == rollout_budget 派生值（在容器里 import 仓库代码核）。"""
    r = _sh(f"{PY} -c \"from syncopate.train.rollout_budget import MAX_PROMPT_LENGTH as p, MAX_RESPONSE_LENGTH as r; print(p+r)\"", cwd=REPO, env=RUN_ENV)
    got = (r["out"].strip().splitlines() or ["?"])[-1]
    assert got == str(SERVE_MAX_MODEL_LEN), f"🔴 服务侧 max_model_len {SERVE_MAX_MODEL_LEN} != rollout_budget 派生 {got}（两处漂了）"


def _start_teacher(log: str = "/tmp/vllm_teacher.log", wait_s: int = 1500):
    _assert_serve_len()
    """Qwen3.8-27B 教师 @8210（两角色同端点）。返回 (proc, up, secs)。build_v16 与 teacher_diag 共用。"""
    cmd = (f"{PY} -m vllm.entrypoints.openai.api_server --model {TEACHER} --served-model-name t --max-model-len {SERVE_MAX_MODEL_LEN} "
           f"--gpu-memory-utilization 0.90 --port 8210 --limit-mm-per-prompt '{{\"image\": 0, \"video\": 0}}' --max-num-seqs 64")
    proc = subprocess.Popen(f"exec {cmd} > {log} 2>&1", shell=True, env={**os.environ, **RUN_ENV})
    up = False; t0 = time.time()
    while time.time() - t0 < wait_s and proc.poll() is None:
        if _sh("curl -sf http://127.0.0.1:8210/health")["rc"] == 0: up = True; break
        time.sleep(5)
    return proc, up, round(time.time() - t0, 1)


# ─────────────────────────── S3-diag · 27B 教师原始思考画像（先量后动，Chaoyu 09-04 放行） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=16, memory=65536, timeout=2 * 3600, secrets=SECRETS)
def p_teacher_diag(n: int = 20, samples: int = 4, max_tokens: int = 4096, arm: str = "base", with_behavior: bool = True) -> dict:
    """判据（预注册，见 scripts/v16_teacher_think_diag.py 顶部）：这是**测量**不是门槛——产出 closed_within_900_rate /
    cjk_below_0.5_rate / action_match_rate 三个数与预注册判读；同容器再跑一遍行为类探针（现在带丢弃原因计数）。
    ok = 两个脚本退出码 0 且 diag.json 落盘（读数本身不判红绿）。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16"; os.makedirs(aud, exist_ok=True)
    proc, up, secs = _start_teacher()
    rec: dict = {"teacher_up": up, "teacher_startup_secs": secs}
    if not up:
        _teardown(proc); return _record("teacher_diag", False, {**rec, "log_tail": open("/tmp/vllm_teacher.log", errors="replace").read()[-2000:]})
    try:
        sfx = "" if arm == "base" else f"_{arm}"
        r = _sh(f"{PY} scripts/v16_teacher_think_diag.py --teacher http://127.0.0.1:8210/v1 --n {n} --samples {samples} --max-tokens {max_tokens} --out {aud} --arm {arm} "
                f"> {aud}/teacher_think_diag{sfx}.log 2>&1; echo RC=$?", cwd=REPO, env=TEACHER_ENV, timeout=3600)
        rec["diag_rc"] = int(re.search(r"RC=(\d+)", r["out"]).group(1))
        rec["diag_tail"] = open(f"{aud}/teacher_think_diag{sfx}.log", errors="replace").read()[-2500:]
        try: rec["agg"] = json.load(open(f"{aud}/teacher_think_diag{sfx}.json"))["agg"]
        except Exception as ex: rec["agg_err"] = repr(ex)[:200]
        rec["behavior_rc"] = 0
        b = None if not with_behavior else _sh(f"{PY} scripts/v15_w3_behavior_think_probe.py --n 20 --teacher http://127.0.0.1:8210/v1 > {aud}/behavior_think_probe.log 2>&1; echo RC=$?", cwd=REPO, env=TEACHER_ENV, timeout=1800)
        if b is not None:
            rec["behavior_rc"] = int(re.search(r"RC=(\d+)", b["out"]).group(1))
            rec["behavior_tail"] = open(f"{aud}/behavior_think_probe.log", errors="replace").read()[-1500:]
            _sh(f"cp _audit/v15_w3/behavior_think_probe.json {aud}/ 2>/dev/null; true", cwd=REPO)
    finally:
        _teardown(proc); open(f"{aud}/teacher_diag_vllm.log", "w").write(open("/tmp/vllm_teacher.log", errors="replace").read()[-20000:]); vol.commit()
    ok = rec.get("diag_rc") == 0 and rec.get("behavior_rc") == 0 and "agg" in rec
    return _record("teacher_diag" if arm == "base" else f"teacher_diag_{arm}", ok, rec)


# ─────────────────────────── S3 · v16 训练集建库（B200 单卡：Qwen3.8-27B 教师 + 26 §W4 七步） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=16, memory=65536, timeout=3 * 3600, secrets=SECRETS)
def p_build_v16(skip_probe: bool = False, gates: str = "strict") -> dict:
    """W4 七步（v16/B200 版）：① 教师起服务（Qwen3.8-27B @8210，两角色同端点）② 行为类 think 探针（≥70% 才收）
    ③ 旧缓存已在 git 里改名作废（*.pre_v16.json，不许命中）④ 建库 tee 入 /vol/_audit/v16/build.log
    ⑤ 判据：[同形] 不同形 0 · [CoT-v15] 命中率行在 · 出厂体检 ✅ · prompt 预算零截断 · 产物 grep "[DRY" = 0
    ⑥ 新池画像（budget_table）⑦ 画廊 markdown 落盘（给 Chaoyu 逐条看）。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16"; os.makedirs(aud, exist_ok=True)
    env = TEACHER_ENV
    log = "/tmp/vllm_teacher.log"
    proc, up, secs = _start_teacher(log)
    rec: dict = {"teacher_up": up, "teacher_startup_secs": secs}
    if not up:
        open(f"{aud}/teacher.log", "w").write(open(log, errors="replace").read()); _teardown(proc); vol.commit()
        return _record("build_v16", False, {**rec, "log_tail": open(log, errors="replace").read()[-2000:]})
    try:
        # 教师缓存（v16_ballast/l2l1/fam/cot/defs/chat_mat 由建库脚本写在 data/u_route/，容器本地）⇒ 建库前从 Volume 取回、建库后存回，断点不从零
        # 裁定⑭（09-04）：run14–16 的 v15_* 缓存含 4B/8B 旧物料（l2 复用 reply、cot 复用 think、defs/chat 是 4B 产物）⇒ 搬进 pre_v16_run16/ 留档，不再取回
        cache_dir = f"{aud}/cache"; os.makedirs(f"{cache_dir}/pre_v16_run16", exist_ok=True)
        _sh(f"mv {cache_dir}/v15_*.json {cache_dir}/pre_v16_run16/ 2>/dev/null; true")
        _sh(f"cp -n {cache_dir}/v16_*.json data/u_route/ 2>/dev/null; true", cwd=REPO)
        if not skip_probe:
            r = _sh(f"{PY} scripts/v15_w3_behavior_think_probe.py --n 20 --teacher http://127.0.0.1:8210/v1 2>&1 | tee {aud}/behavior_think_probe.log | tail -20", cwd=REPO, env=env, timeout=1800)
            rec["behavior_probe"] = {"rc": r["rc"], "tail": r["out"][-1200:]}
            _sh(f"cp _audit/v15_w3/behavior_think_probe.json {aud}/ 2>/dev/null; true", cwd=REPO)
        # 旧物料不许命中（裁定⑭）：判据 = 建库脚本源码里不再出现任何旧缓存/物料文件名（前任 09-04 核对：原判据数的是取回的新缓存，第二次起必红）
        # 只数代码行（注释里提到旧名不算——run17 实测 3 条注释把判据判红）
        stale = _sh("grep -vE '^\\s*#' scripts/u_build_v14_5.py | grep -cE 'open\\(.*(v15_(cot_rows|l2l1_rows|ballast_replies|fam_rows|materials)|v145_(defs|chat_mat))\\.json|cand_v13r2_e1/' || true", cwd=REPO)["out"].strip()
        rec["stale_caches_present"] = int(stale or 0)
        # ★ 09-04 固定管线：探针不再自己拼命令，只调 runbook 的 stage（本机 == 云上）；教师由本函数起（进程管理），runbook 检测到已在线会跳过
        # gates=report：闸观察模式（所有闸都算不中断、产物落 _audit/v16/report、末尾汇总）——一次看全貌，不再红一道停一道
        b = _sh(f"bash scripts/v16_pipeline.sh sft-data > {aud}/sft_data_stage.log 2>&1; echo BUILD_RC=$?", cwd=REPO, env={**env, "PY": PY, "U_BUILD_GATES": gates}, timeout=2 * 3600)
        _sh(f"cp _audit/v16/build.log {aud}/build.log 2>/dev/null; cp _audit/v16/gallery.md {aud}/gallery.md 2>/dev/null; true", cwd=REPO)
        blog = open(f"{aud}/build.log", errors="replace").read() if os.path.exists(f"{aud}/build.log") else open(f"{aud}/sft_data_stage.log", errors="replace").read()
        rec["build_rc"] = int(re.search(r"BUILD_RC=(\d+)", b["out"]).group(1)) if re.search(r"BUILD_RC=(\d+)", b["out"]) else -1
        rec["build_secs"] = b["secs"]
        rec["judge_lines"] = [l[-200:] for l in blog.splitlines() if any(k in l for k in ("[同形]", "[CoT-v15]", "出厂体检", "份额", "密度", "✅", "🔴"))][-40:]
        rec["build_tail"] = blog[-2500:]
        if rec["build_rc"] == 0:
            # 三项都已由 runbook sft-data 跑过（顺序：建库 → prompt 预算 → 隔离复核 → 画廊）；这里只从 stage 日志取读数、复核画廊无占位
            slog = open(f"{aud}/sft_data_stage.log", errors="replace").read()
            rec["prompt_budget"] = {"rc": 0 if "零截断" in slog or "over=0" in slog or "✅" in slog else 1, "tail": slog[-800:]}
            rec["isolation"] = {"rc": 0 if "越桶 0" in slog and "🔴" not in slog.split("[隔离]")[-1][:200] else 1, "tail": slog[-600:]}
            rec["gallery"] = {"tail": "", "dry_hits": int(_sh(f"grep -c '\\[DRY' {aud}/gallery.md || true")["out"].strip() or 0)}
            bt = _sh(f"{PY} scripts/v15_w3_budget_table.py 2>&1 | tail -12", cwd=REPO, env=env, timeout=1200)
            rec["budget_table"] = bt["out"][-1200:]
            rec["parquet"] = _sh("ls -la data/sft/v16/ && ls -la data/u_route/ | grep -E 'v15_(cot|l2l1|ballast)'", cwd=REPO)["out"][-800:]
    finally:
        _sh(f"cp data/u_route/v16_*.json {aud}/cache/ 2>/dev/null; true", cwd=REPO)
        _teardown(proc); open(f"{aud}/teacher.log", "w").write(open(log, errors="replace").read()[-20000:]); vol.commit()
    ok = (rec.get("build_rc") == 0 and rec.get("stale_caches_present", 1) == 0
          and any("不同形 0" in l or "不同形: 0" in l for l in rec.get("judge_lines", []))
          and rec.get("prompt_budget", {}).get("rc") == 0 and rec.get("gallery", {}).get("dry_hits", 1) == 0
          and rec.get("isolation", {}).get("rc") == 0)
    return _record("build_v16", ok, rec)


# ─────────────────────────── S4 · SFT 冒烟（B200 单卡：Qwen3.6-35B-A3B + LoRA attn_shared，N 步） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=16, memory=131072, timeout=3 * 3600, secrets=SECRETS)
def p_sft_smoke(max_steps: int = 30, use_wandb: bool = False, arm: str = "v16_smoke", train_file: str = "", val_file: str = "", epochs: int = 1) -> dict:
    """判据（26 W4′ S4）：loss 有限且末窗均值 < 首窗均值 · grad_norm 全有限 · 可训参数量 ≈ 37M±20% · 存档能被 peft 加载且 ΔW>0 ·
    峰值显存 < 180 GB · 吞吐 tok/s 记录（学习项）。
    arm="mech_dry"（09-04 并行机制冒烟）：容器里先 U_BUILD_DRY=6 演练出 _audit/v16/dry_rows.parquet（"[DRY" 占位、不调教师），
    只验 LoRA 模块名/参数量/显存/吞吐/存档这些**与数据内容无关**的机制；产物目录带 arm 名，**不是候选**。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16/sft_{arm}"; os.makedirs(aud, exist_ok=True)
    out = f"{VOL}/checkpoints/sft/{arm}"; _sh(f"rm -rf {out}")
    rec: dict = {"arm": arm}
    if arm == "mech_dry" and not train_file:
        d = _sh(f"U_BUILD_DRY=6 {PY} scripts/u_build_v14_5.py > {aud}/dry_build.log 2>&1; echo RC=$?", cwd=REPO, env=RUN_ENV, timeout=1800)
        rec["dry_build_rc"] = int(re.search(r"RC=(\d+)", d["out"]).group(1)); rec["dry_build_tail"] = open(f"{aud}/dry_build.log", errors="replace").read()[-1500:]
        if rec["dry_build_rc"] != 0 or not os.path.exists(f"{REPO}/_audit/v16/dry_rows.parquet"):
            vol.commit(); return _record("sft_smoke", False, rec)
        _sh(f"cp _audit/v16/dry_rows.parquet {aud}/", cwd=REPO)
        train_file = val_file = "_audit/v16/dry_rows.parquet"
    train_file = train_file or "data/sft/v16/train.parquet"; val_file = val_file or "data/sft/v16/val.parquet"
    wandb_flag = "" if use_wandb else "--no-wandb"
    if arm == "mech_dry":   # 机制冒烟：占位数据，仍走 sft.py 本体（runbook 的 smoke 档要真 parquet）
        cmd = (f"{PY} -m syncopate.train.sft --model {STUDENT} --train-file {train_file} --val-file {val_file} "
               f"--out {out} --epochs {epochs} --batch-size 1 --grad-accum 8 --max-steps {max_steps} {wandb_flag} --wandb-run sft_{arm} "
               f"> {aud}/sft_smoke.log 2>&1; echo SFT_RC=$?")
    else:                   # 09-04 固定管线：runbook 的 sft-train（smoke 档 = 30 步冒烟；candidate 档 = 正式）
        cmd = f"bash scripts/v16_pipeline.sh --profile {'smoke' if arm == 'v16_smoke' else 'candidate'} sft-train > {aud}/sft_smoke.log 2>&1; echo SFT_RC=$?"
    r = _sh(cmd, cwd=REPO, env=RUN_ENV, timeout=3 * 3600 - 300)
    log = open(f"{aud}/sft_smoke.log", errors="replace").read()
    rc = int(re.search(r"SFT_RC=(\d+)", r["out"]).group(1)) if re.search(r"SFT_RC=(\d+)", r["out"]) else -1
    losses = [float(x) for x in re.findall(r"train/loss[=: ]+([0-9.]+)", log)] or [float(x) for x in re.findall(r"loss=([0-9.]+)", log)]
    gnorms = [float(x) for x in re.findall(r"grad_norm[=: ]+([0-9.eE+-]+)", log)]
    trainable = re.findall(r"可训练 ([0-9.]+)M", log)
    peak = re.findall(r"peak_memory_gb[=: ]+([0-9.]+)", log) or re.findall(r"显存峰值[^0-9]*([0-9.]+)", log)   # sft.py 打的是「显存峰值 74.1 GB」（mech_dry 实测判据量错对象）
    dw = re.findall(r"\|\|ΔW\|\|/\|\|W\|\| = ([0-9.]+)%", log)
    import math
    k = max(1, len(losses) // 4)
    loss_ok = bool(losses) and all(math.isfinite(x) for x in losses) and (sum(losses[-k:]) / k) < (sum(losses[:k]) / k)
    gn_ok = bool(gnorms) and all(math.isfinite(x) for x in gnorms)
    tr_ok = bool(trainable) and 30 <= float(trainable[0]) <= 45
    peak_ok = (not peak) or float(peak[-1]) < 180
    ok = rc == 0 and loss_ok and gn_ok and tr_ok and peak_ok
    rec.update({"rc": rc, "secs": r["secs"], "train_file": train_file, "n_loss_points": len(losses), "loss_first_last": (losses[:3], losses[-3:]), "grad_norm_minmax": (min(gnorms), max(gnorms)) if gnorms else None,
           "trainable_M": trainable[:1], "peak_memory_gb": peak[-1:], "delta_w_pct": dw[-1:], "judge": {"loss": loss_ok, "grad": gn_ok, "trainable": tr_ok},
           "lora_targets_line": [l for l in log.splitlines() if "[lora-targets]" in l][:1], "tail": log[-3000:], "topology": _topology()})
    vol.commit()
    return _record(f"sft_smoke_{arm}" if arm != "v16_smoke" else "sft_smoke", ok, rec)


# ─────────────────────────── S5 · 考场 v4 单容器（B200 单卡：vLLM 学生端点 + PG/Redis + API + worker + 四遍 + 判卷） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=16, memory=98304, timeout=4 * 3600, secrets=SECRETS)
def p_exam_v4(model: str = "", adapter: str = "", arm: str = "v16_smoke", passes: int = 1, concurrency: int = 4, limit: int = 0) -> dict:
    """26 §W5 起链五步的容器版：0 seed_demo --check（7 条 campaign）1 无陈旧 worker 2 起端点(:8100)+API(:8000)+worker
    3 u_exam_run 四遍（每遍落 jsonl，可重入）4 u_exam_judge_v4 5 v15_gate_triage。判据 = 每遍 rc 0 · 判卷器 rc 0 · triage 出表。
    model 默认学生底座；adapter 给 LoRA 目录则 vLLM --enable-lora（served 名仍为 model）。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16/exam_{arm}"; os.makedirs(aud, exist_ok=True)
    model = model or STUDENT
    # 前任 09-04 核对：① 容器里 PG 是新的，--check 只查不播 ⇒ 先播种再 --check；② vLLM 挂 adapter 时 LoRA 的 served 名必须
    #   与 SYNCOPATE_DECIDER_MODEL 一致（decider 按 /v1/models 核名，原写法 basename(model) 与完整路径对不上）⇒ 统一叫 v16_adapter
    served = "v16_adapter" if adapter else model
    sv = _start_services(); rec: dict = {"services": {k: v.get("rc") for k, v in sv.items()}, "arm": arm, "model": model, "adapter": adapter, "served": served}
    if sv["pg"]["rc"] != 0: return _record("exam_v4", False, rec)
    env = {**RUN_ENV, **SERVICE_ENV, "SYNCOPATE_DECIDER_URL": "http://127.0.0.1:8100", "SYNCOPATE_DECIDER_TOKENIZER": model,
           "SYNCOPATE_DECIDER_MODEL": served, "SYNCOPATE_API_DB_POOL": "12"}
    rs = _sh(f"{PY} scripts/seed_demo_data.py", cwd=REPO, env=env, timeout=600); rec["seed"] = {"rc": rs["rc"], "tail": rs["out"][-300:]}
    r0 = _sh(f"{PY} scripts/seed_demo_data.py --check", cwd=REPO, env=env, timeout=600); rec["seed_check"] = {"rc": r0["rc"], "tail": r0["out"][-300:]}
    lora = f" --enable-lora --max-lora-rank 64 --lora-modules {served}={adapter}" if adapter else ""
    vcmd = (f"{PY} -m vllm.entrypoints.openai.api_server --model {model} --served-model-name {model} --max-model-len {SERVE_MAX_MODEL_LEN} "
            f"--gpu-memory-utilization 0.85 --port 8100 --limit-mm-per-prompt '{{\"image\": 0, \"video\": 0}}'{lora}")
    vlog = "/tmp/vllm_exam.log"
    _assert_serve_len()
    vproc = subprocess.Popen(f"exec {vcmd} > {vlog} 2>&1", shell=True, env={**os.environ, **RUN_ENV})
    up = False; t0 = time.time()
    while time.time() - t0 < 1500 and vproc.poll() is None:
        if _sh("curl -sf http://127.0.0.1:8100/health")["rc"] == 0: up = True; break
        time.sleep(5)
    rec["endpoint_up"] = up; rec["endpoint_secs"] = round(time.time() - t0, 1)
    if not up:
        open(f"{aud}/vllm.log", "w").write(open(vlog, errors="replace").read()[-20000:]); _teardown(vproc); vol.commit()
        return _record("exam_v4", False, rec)
    api = subprocess.Popen(f"exec {PY} -m uvicorn syncopate.runtime.api:app --host 127.0.0.1 --port 8000 --workers 2 > {aud}/api.log 2>&1", shell=True, cwd=REPO, env={**os.environ, **env})
    wrk = subprocess.Popen(f"exec {PY} -m syncopate.runtime.worker --org-id org_demo --worker-id v16-exam --daily-cost-cap-micros 10000000000 > {aud}/worker.log 2>&1", shell=True, cwd=REPO, env={**os.environ, **env})
    time.sleep(20)
    try:
        runs = []
        for i in range(1, passes + 1):
            lim = f" --limit {limit}" if limit else ""
            r = _sh(f"{PY} scripts/u_exam_run.py --exam context_v4 --arm {arm}_r{i} --concurrency {concurrency}{lim} > {aud}/exam_r{i}.log 2>&1; echo RC=$?", cwd=REPO, env=env, timeout=3 * 3600)
            rc = int(re.search(r"RC=(\d+)", r["out"]).group(1)); runs.append({"pass": i, "rc": rc, "secs": r["secs"]})
            _sh(f"cp logs/u_route/run_{arm}_r{i}_*.jsonl {aud}/ 2>/dev/null; true", cwd=REPO)
        rec["runs"] = runs
        jl = " ".join(f"logs/u_route/run_{arm}_r{i}_context_v4.jsonl" for i in range(1, passes + 1))
        j = _sh(f"{PY} scripts/u_exam_judge_v4.py --context {jl} > {aud}/judge.log 2>&1; echo RC=$?", cwd=REPO, env=env, timeout=1800)
        rec["judge_rc"] = int(re.search(r"RC=(\d+)", j["out"]).group(1)); rec["judge_tail"] = open(f"{aud}/judge.log", errors="replace").read()[-2500:]
        _sh(f"cp logs/u_route/judged_*{arm}* {aud}/ 2>/dev/null; true", cwd=REPO)
        t = _sh(f"{PY} scripts/v15_gate_triage.py > {aud}/triage.log 2>&1; echo RC=$?", cwd=REPO, env=env, timeout=600)
        rec["triage_rc"] = int(re.search(r"RC=(\d+)", t["out"]).group(1)); rec["triage_tail"] = open(f"{aud}/triage.log", errors="replace").read()[-1500:]
    finally:
        for pr in (wrk, api):
            pr.terminate()
        _teardown(vproc); open(f"{aud}/vllm.log", "w").write(open(vlog, errors="replace").read()[-20000:]); vol.commit()
    ok = up and rec["seed_check"]["rc"] == 0 and all(x["rc"] == 0 for x in rec.get("runs", [])) and rec.get("judge_rc") == 0
    return _record(f"exam_v4_{arm}" if arm != "v16_smoke" else "exam_v4", ok, rec)

# ─────────────────────────── S6 · RL（verl 0.9 V1）：键名判据（CPU）+ 冒烟（B200×2） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=1800, secrets=SECRETS)
def p_rl_cfg(extra: str = "") -> dict:
    """S6 前置（CPU，零 GPU 费）：① `syncopate data build --pool rl` 造 data/rl/v16（判据：train/val 行数 = 切分 rl 桶 ⁄ val_every）
    ② `launch_rl_v1 --cfg-only`：Hydra 只合成配置不起 Ray ⇒ 任何键名在 0.9 里不存在会在这里红，不烧 GPU。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16/rl"; os.makedirs(aud, exist_ok=True)
    rec: dict = {}
    if not os.path.exists(f"{REPO}/data/rl/v16/train.parquet"):
        b = _sh(f"{PY} -m syncopate data build --pool rl --batch data/batches/v16 --split-dir data/splits/v16 --out data/rl/v16 --val-every 5 > {aud}/build_rl.log 2>&1; echo RC=$?", cwd=REPO, env=RUN_ENV, timeout=1200)
        rec["build_rl_rc"] = int(re.search(r"RC=(\d+)", b["out"]).group(1)); rec["build_rl_tail"] = open(f"{aud}/build_rl.log", errors="replace").read()[-800:]
    try: rec["rl_manifest"] = json.load(open(f"{REPO}/data/rl/v16/manifest.json"))
    except Exception as ex: rec["rl_manifest_err"] = repr(ex)[:200]
    # 键名判据用 smoke 档（candidate 档会先断言合并 SFT 模型存在——那是另一段的事）
    r = _sh(f"{PY} -m syncopate.train.launch_rl_v1 --profile smoke --cfg-only --logger console {extra} > {aud}/cfg_only.log 2>&1; echo RC=$?", cwd=REPO, env=RUN_ENV, timeout=900)
    rec["cfg_rc"] = int(re.search(r"RC=(\d+)", r["out"]).group(1))
    log = open(f"{aud}/cfg_only.log", errors="replace").read()
    rec["cfg_tail"] = log[-3000:]
    rec["cfg_has_v1"] = ("trainer_mode: sync" in log) and ("syncopate_adcampaign" in log)
    vol.commit()
    ok = rec["cfg_rc"] == 0 and rec["cfg_has_v1"] and "rl_manifest" in rec
    return _record("rl_cfg", ok, rec)


@app.function(image=image, volumes={VOL: vol}, gpu=GPU_PAIR, cpu=32, memory=262144, timeout=3 * 3600, secrets=SECRETS)
def p_rl_smoke(steps: int = 2, gpus: int = 2, extra: str = "", arm: str = "v16_smoke") -> dict:
    """S6 冒烟（判据在 launch_rl_v1 顶部注册）：每步退出码 0 · `[pool] 动态分池启用` 在 worker 侧 · loss/grad 有限 · reward 非全 0 ·
    权重同步行 · LoRA-only ckpt 落盘。产物 /vol/checkpoints/grpo/<arm>（一个写者）；日志 /vol/_audit/v16/rl/<arm>.log。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16/rl"; os.makedirs(aud, exist_ok=True)
    save = f"{VOL}/checkpoints/grpo/{arm}"; _sh(f"rm -rf {save}")
    if not os.path.exists(f"{REPO}/data/rl/v16/train.parquet"):
        _sh(f"{PY} -m syncopate data build --pool rl --batch data/batches/v16 --split-dir data/splits/v16 --out data/rl/v16 --val-every 5 > {aud}/build_rl.log 2>&1", cwd=REPO, env=RUN_ENV, timeout=1200)
    # 09-04 固定管线：smoke 档走 runbook（默认值即注册值）；额外参数只给探索臂（extra）
    if arm == "v16_smoke" and not extra:
        cmd = f"bash scripts/v16_pipeline.sh --profile smoke rl-train > {aud}/{arm}.log 2>&1; echo RL_RC=$?"
    else:
        cmd = (f"{PY} -m syncopate.train.launch_rl_v1 --profile smoke --steps {steps} --gpus {gpus} --experiment rl_{arm} --save-path {save} --logger console,wandb {extra} "
               f"> {aud}/{arm}.log 2>&1; echo RL_RC=$?")
    r = _sh(cmd, cwd=REPO, env=RUN_ENV, timeout=3 * 3600 - 600)
    log = open(f"{aud}/{arm}.log", errors="replace").read()
    rc = int(re.search(r"RL_RC=(\d+)", r["out"]).group(1)) if re.search(r"RL_RC=(\d+)", r["out"]) else -1
    import math
    losses = [float(x) for x in re.findall(r"actor/pg_loss[\'\"]?[:=]\s*([-0-9.eE+]+)", log)]
    gn = [float(x) for x in re.findall(r"actor/grad_norm[\'\"]?[:=]\s*([-0-9.eE+]+)", log)]
    rew = [float(x) for x in re.findall(r"critic/score/mean[\'\"]?[:=]\s*([-0-9.eE+]+)", log)]
    rec = {"arm": arm, "rc": rc, "secs": r["secs"], "steps": steps, "pool_line": "[pool] 动态分池启用" in log, "n_loss": len(losses), "losses": losses[:6],
           "grad_norms": gn[:6], "score_mean": rew[:6], "weight_sync_lines": len(re.findall(r"update_weights|checkpoint_engine|weights synced", log, flags=re.I)),
           "ckpt_dirs": _sh(f"find {save} -maxdepth 3 -type d | head -20")["out"], "lora_files": _sh(f"find {save} -name 'adapter_model*' -o -name 'lora*' | head")["out"],
           "traceback": ("Traceback" in log), "tail": log[-4000:], "topology": _topology()}
    # 收尾杀全家（Ray/vLLM）等显存归零
    _sh("ray stop --force >/dev/null 2>&1; pkill -9 -f 'vllm' ; pkill -9 -f EngineCore ; pkill -9 -f ray:: ; sleep 5; true")
    vol.commit()
    ok = rc == 0 and rec["pool_line"] and bool(losses) and all(math.isfinite(x) for x in losses) and all(math.isfinite(x) for x in gn) and (not rew or any(x != 0 for x in rew))
    return _record(f"rl_smoke_{arm}", ok, rec)


# ─────────────────────────── S7 · OPD 冒烟（B200×2：学生@0 · 教师+锚@1；逐 token 反 KL） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_PAIR, cpu=16, memory=196608, timeout=2 * 3600, secrets=SECRETS)
def p_opd_smoke(max_steps: int = 5, adapter: str = "", arm: str = "v16_smoke", batch: int = 4) -> dict:
    """S7 冒烟：opd.py（v16 版：--adapter 可空=底座新建 LoRA；教师 27B；vocab 断言）跑 --max-steps N。
    判据：rc 0 · `[opd-vocab] ✓` · `[opd-mask]` 非零 · 每步 KL 有限 · 零掩码对照断言过（--probe-every 命中一次）· adapter 落盘。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16/opd"; os.makedirs(aud, exist_ok=True)
    out = f"{VOL}/checkpoints/opd/{arm}"; _sh(f"rm -rf {out}")
    ad = f" --adapter {adapter}" if adapter else ""
    # 09-04 固定管线：smoke 档走 runbook；给了 adapter 则 candidate 档
    cmd = f"bash scripts/v16_pipeline.sh --profile {'candidate' if adapter else 'smoke'} opd-train > {aud}/{arm}.log 2>&1; echo OPD_RC=$?"
    r = _sh(cmd, cwd=REPO, env=RUN_ENV, timeout=2 * 3600 - 300)
    log = open(f"{aud}/{arm}.log", errors="replace").read()
    rc = int(re.search(r"OPD_RC=(\d+)", r["out"]).group(1)) if re.search(r"OPD_RC=(\d+)", r["out"]) else -1
    import math
    kls = [float(x) for x in re.findall(r"kl_(?:chat|task)/tok=([-0-9.eE+]+)", log)][:20]   # opd.py 的 step 行格式（首跑正则量错对象）
    rec = {"arm": arm, "rc": rc, "secs": r["secs"], "vocab_ok": "[opd-vocab]" in log, "mask_lines": len(re.findall(r"\[opd-mask\]", log)),
           "probe_lines": len(re.findall(r"零掩码|zero-mask|\[opd-probe\]", log)), "kls": kls, "trainable": re.findall(r"可训练 ([0-9.]+)M", log)[:1],
           "adapter_files": _sh(f"find {out} -name 'adapter_model*' | head")["out"], "traceback": "Traceback" in log, "tail": log[-4000:], "topology": _topology()}
    vol.commit()
    rec["skipped_steps"] = len(re.findall(r"集体跳步", log)); rec["real_steps"] = len(re.findall(r"\] step \d+ ep\d+ kl_chat", log))
    ok = rc == 0 and rec["vocab_ok"] and all(math.isfinite(x) for x in kls) and bool(rec["adapter_files"].strip())
    return _record(f"opd_smoke_{arm}", ok, rec)


# ─────────────────────────── E-think · CoT 开/关 A/B（B200 单卡一臂；两臂并行各起一台） ───────────────────────────
@app.function(image=image, volumes={VOL: vol}, gpu=GPU_ONE, cpu=16, memory=131072, timeout=5 * 3600, secrets=SECRETS)
def p_eval_ab(think: int = 0, arm: str = "", model: str = "", adapter: str = "", samples: int = 8, limit: int = 0, families: str = "", gpu_util: float = 0.85) -> dict:
    """26 §W4′ E-think：同一份 eval_local（冻结 EVAL 342 题 × samples），只变 SYNCOPATE_THINK。判据 J1–J6 在 26 里预注册，
    本步只产读数（/vol/_audit/v16/eval/<arm>.json）+ 有效性（rc 0 · 行数 == 题数）。分析在本机 scripts/v16_think_ab_report.py。"""
    _sync_repo()
    aud = f"{VOL}/_audit/v16/eval"; os.makedirs(aud, exist_ok=True)
    model = model or STUDENT; arm = arm or f"think_{'on' if think else 'off'}"
    env = {**RUN_ENV, "SYNCOPATE_THINK": str(int(think))}
    ad = f" --adapter {adapter}" if adapter else ""
    lim = f" --limit {limit}" if limit else ""
    fam = f" --families {families}" if families else ""
    cmd = (f"{PY} -m syncopate.train.eval_local --model {model}{ad} --samples-per-case {samples} --gpu-util {gpu_util}{lim}{fam} "
           f"--out _audit/v16/eval/{arm}.json > {aud}/{arm}.log 2>&1; echo RC=$?")
    t0 = time.time()
    r = _sh(cmd, cwd=REPO, env=env, timeout=5 * 3600 - 600)
    rc = int(re.search(r"RC=(\d+)", r["out"]).group(1)) if re.search(r"RC=(\d+)", r["out"]) else -1
    _sh(f"cp _audit/v16/eval/{arm}.json {aud}/ 2>/dev/null; true", cwd=REPO)
    log = open(f"{aud}/{arm}.log", errors="replace").read()
    rec = {"arm": arm, "think": think, "model": model, "rc": rc, "wall_secs": round(time.time() - t0, 1), "samples": samples,
           "think_mode_line": [l for l in log.splitlines() if "[think-mode]" in l][:1], "summary_tail": log[-3500:], "topology": _topology()}
    try:
        j = json.load(open(f"{aud}/{arm}.json")); rec["n_rows"] = len(j["rows"]); rec["mean_reward"] = sum(x["reward"] for x in j["rows"]) / max(1, len(j["rows"]))
    except Exception as ex: rec["json_err"] = repr(ex)[:200]
    vol.commit()
    ok = rc == 0 and rec.get("n_rows", 0) > 0
    return _record(f"eval_ab_{arm}", ok, rec)


# ─────────────────────────── 本机入口 ───────────────────────────
ALL_STEPS = ["image", "verl", "versions", "models", "gpu", "fa4", "nccl", "vllm", "vllm_ep"]


@app.local_entrypoint()
def main(steps: str = ",".join(ALL_STEPS), models_only: str = "", pytest_args: str = "tests -q -rfE -p no:cacheprovider", exec_file: str = "", expected_sha: str = "", max_steps: int = 30, exam_model: str = "", exam_adapter: str = "", exam_arm: str = "v16_smoke", exam_passes: int = 1, exam_limit: int = 0, sft_arm: str = "v16_smoke", sft_train_file: str = "", sft_val_file: str = "", sft_epochs: int = 1, diag_n: int = 20, diag_samples: int = 4, diag_max_tokens: int = 4096, diag_arm: str = "base", rl_steps: int = 2, rl_gpus: int = 2, rl_extra: str = "", rl_arm: str = "v16_smoke", opd_steps: int = 5, opd_adapter: str = "", opd_arm: str = "v16_smoke", ab_think: int = 0, ab_arm: str = "", ab_samples: int = 8, ab_limit: int = 0, ab_families: str = "", ab_adapter: str = "", build_gates: str = "strict"):
    want = [s.strip() for s in steps.split(",") if s.strip()]
    results: dict[str, dict] = {}
    t0 = time.time()

    def run(name, fn, *a, **kw):
        try:
            results[name] = fn.remote(*a, **kw)
        except Exception as ex:
            results[name] = {"step": name, "ok": None, "error": repr(ex)[:2000]}
            print(f"[{name}] ⚠️ 没跑成：{repr(ex)[:300]}")

    if "image" in want: run("image", p_image)
    if "verl" in want: run("verl", p_verl)
    if "models" in want: run("models", p_models, models_only)
    if "versions" in want: run("versions", p_versions)
    if "gpu" in want: run("gpu", p_gpu)
    if "fa4" in want: run("fa4", p_fa4)
    if "nccl" in want: run("nccl", p_nccl)
    if "vllm" in want: run("vllm", p_vllm)
    if "vllm_ep" in want: run("vllm_ep", p_vllm_ep)
    if "pytest" in want: run("pytest", p_pytest, pytest_args)
    if "wandb" in want: run("wandb", p_wandb)
    if "exec" in want and exec_file: run("exec", p_exec, open(exec_file).read())
    if "rebuild_v16" in want: run("rebuild_v16", p_rebuild_v16, expected_sha)
    if "build_v16" in want: run("build_v16", p_build_v16, False, build_gates)
    if "teacher_diag" in want: run("teacher_diag", p_teacher_diag, diag_n, diag_samples, diag_max_tokens, diag_arm, diag_arm == "base")
    if "sft_smoke" in want: run("sft_smoke", p_sft_smoke, max_steps, False, sft_arm, sft_train_file, sft_val_file, sft_epochs)
    if "exam_v4" in want: run("exam_v4", p_exam_v4, exam_model, exam_adapter, exam_arm, exam_passes, 4, exam_limit)
    if "rl_cfg" in want: run("rl_cfg", p_rl_cfg, rl_extra)
    if "rl_smoke" in want: run("rl_smoke", p_rl_smoke, rl_steps, rl_gpus, rl_extra, rl_arm)
    if "opd_smoke" in want: run("opd_smoke", p_opd_smoke, opd_steps, opd_adapter, opd_arm)
    if "eval_ab" in want: run("eval_ab", p_eval_ab, ab_think, ab_arm, "", ab_adapter, ab_samples, ab_limit, ab_families)

    out_dir = LOCAL_ROOT / "_audit" / "stack_probe"; out_dir.mkdir(parents=True, exist_ok=True)
    # 09-04：并行多臂时两个 run 同一分钟收尾会互相覆盖（exam_plumb 被 rl_cfg 盖掉过）⇒ 文件名带秒 + 步名
    stamp = time.strftime("%Y-%m-%d_%H%M%S") + "_" + "-".join(want)[:40]
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\n══════ 新栈探针 · 汇总 ══════")
    for k, v in results.items():
        print(f"  {k:<10} {'✅' if v.get('ok') else ('⚠️ 没跑成' if v.get('ok') is None else '🔴')}")
    if results.get("fa4", {}).get("fa4", {}).get("fa4_over_fa2_speedup"):
        print(f"  FA4/FA2 前向加速 = {results['fa4']['fa4']['fa4_over_fa2_speedup']}× · FA4 {results['fa4']['fa4'].get('fa4_tflops_S8192')} TFLOPS")
    if results.get("vllm", {}).get("mtp_on_over_off_ms_ratio") is not None:
        print(f"  MTP 开/关 每 token 耗时比 = {results['vllm']['mtp_on_over_off_ms_ratio']}（>1 = MTP 更慢）")
    print(f"  耗时 {round(time.time() - t0)} s · 明细 {out_dir / f'summary_{stamp}.json'} · Volume {AUDIT}/")
    red = [k for k, v in results.items() if v.get("ok") is False]
    unrun = [k for k, v in results.items() if v.get("ok") is None]
    sys.exit(1 if red else (2 if unrun else 0))

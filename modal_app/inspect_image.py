"""看一个现成 docker 镜像里装了什么（给选基础镜像用）。  modal run modal_app/inspect_image.py --tag verlai/verl:uv.cu130.dev3"""
import modal
app = modal.App("syncopate-inspect-image")


@app.function(cpu=2, memory=8192, timeout=900)
def noop(): pass


@app.local_entrypoint()
def main(tag: str = "verlai/verl:uv.cu130.dev3"):
    img = modal.Image.from_registry(tag, add_python=None) if False else modal.Image.from_registry(tag)
    f = modal.Function.from_local(inspect, image=img) if hasattr(modal.Function, "from_local") else None
    import subprocess, json
    r = subprocess.run(["modal", "shell", "--image", tag, "--cmd",
        "sh -lc 'which python python3 uv pip; python3 -V; (pip list 2>/dev/null || python3 -m pip list 2>/dev/null || uv pip list 2>/dev/null) | grep -i -E \"^(torch|vllm|verl|transformers|megatron|transformer.engine|flash|flashinfer|sglang|ray|nvidia-cutlass|apex|deep.ep|triton) \"; nvcc --version 2>/dev/null | tail -1; ls /opt /workspace 2>/dev/null | head; env | grep -i -E \"cuda|torch|path\" | head'",
        "--no-pty"], capture_output=True, text=True, timeout=800)
    print(r.stdout[-6000:]); print(r.stderr[-2000:])


def inspect(): pass

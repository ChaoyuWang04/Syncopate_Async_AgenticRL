"""B06 derived dependency image; production lock remains unchanged."""
from __future__ import annotations
import modal
import sys
if not modal.is_local():sys.path.insert(0,'/opt/syncopate-current')
from modal_app.probability_batch import image,vol,execute
fast_image=image.run_commands(
    'gcc --version','g++ --version',
    'CC=gcc CXX=g++ CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 uv pip install --python /env/.venv/bin/python --no-deps --no-build-isolation causal-conv1d==1.7.0')
app=modal.App('syncopate-gdn-fastpath-batch')
@app.function(image=fast_image,volumes={'/vol':vol},cpu=4,memory=16384,timeout=1200,retries=0)
def cpu(run_id):return execute(run_id,'fast_cpu',runtime_image=fast_image,dispatch_file=__file__)
@app.function(image=fast_image,volumes={'/vol':vol},gpu='B200',cpu=16,memory=131072,timeout=1800,retries=0)
def gpu(run_id,preflight):return execute(run_id,'fast',preflight,runtime_image=fast_image,dispatch_file=__file__)
@app.local_entrypoint()
def main(run_id:str,phase:str='cpu',preflight:str=''):
    import json
    if phase not in {'cpu','gpu'}:raise ValueError('unsupported phase before allocation')
    result=cpu.remote(run_id) if phase=='cpu' else gpu.remote(run_id,preflight)
    print(json.dumps(result,indent=2))
    if not result['ok']:raise RuntimeError('Probe failed: '+result['evidence_path'])

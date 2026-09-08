"""B00: test the installed upstream metric, without model or generation claims."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys


def main(output: Path):
    import torch
    import verl
    from verl import DataProto
    from verl.trainer.ppo import metric_utils
    torch.set_num_threads(2)
    source = Path(inspect.getfile(metric_utils)).read_bytes()
    # Exact upstream blob pinned by the preregistration (release and main identical).
    blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
    assert blob == 'c9c8c1e92e44130fddd5c7d1be51d55b15d59cea', blob
    (output / 'metric_utils.py').write_bytes(source)
    rows = []
    for width, lengths, reasons in [(4, [2, 4], ['stop', 'stop']),
                                   (12, [2, 4], ['stop', 'stop']),
                                   (12, [2, 12], ['stop', 'length']),
                                   (12, [2, 12], ['stop', 'stop'])]:
        mask = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
        responses = (torch.arange(width)[None, :] + 10).expand(2, -1).clone() * mask
        prompts = torch.ones(2, 3, dtype=torch.long)
        batch = DataProto.from_dict(tensors={
            'prompts': prompts, 'responses': responses,
            'attention_mask': torch.cat([torch.ones_like(prompts), mask.long()], dim=1),
            'response_mask': mask,
            **{key: torch.ones(2, width) for key in
               ('token_level_scores', 'token_level_rewards', 'advantages', 'returns')},
        })
        metrics = metric_utils.compute_data_metrics(batch, use_critic=False)
        actual = metrics['response_length/clip_ratio']
        assert actual == sum(n == width for n in lengths) / 2
        assert metrics['response_length/max'] == max(lengths)
        rows.append({'width': width, 'lengths': lengths, 'raw_tokens': [r[m].tolist() for r, m in zip(responses, mask)],
                     'synthetic_finish_reason': reasons, 'generation_budget': 12,
                     'boundary_fraction': actual, 'length_finish_fraction': reasons.count('length') / 2})
    assert rows[0]['raw_tokens'] == rows[1]['raw_tokens']
    assert rows[2]['boundary_fraction'] == rows[3]['boundary_fraction'] == .5
    checks = subprocess.run([sys.executable, '-m', 'pytest', 'tests/train/test_rl_run_gate.py',
                             'tests/infra/test_probe_run.py', '-q'], text=True, capture_output=True)
    (output / 'pytest.txt').write_text(checks.stdout + checks.stderr)
    assert checks.returncode == 0, checks.stdout + checks.stderr
    result = {'ok': True, 'torch': torch.__version__, 'verl': verl.__version__,
              'metric_git_blob': blob, 'source_sha256': hashlib.sha256(source).hexdigest(),
              'rows': rows, 'pytest': checks.stdout, 'scope': 'metric function and local consumer; no live rollout'}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))

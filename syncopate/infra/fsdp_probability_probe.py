"""Real verl FSDP2 reference probabilities, with identical local batches on both ranks.

Run with torchrun --standalone --nproc-per-node=2 -m
syncopate.infra.fsdp_probability_probe --input INPUT --output OUTPUT.
No optimizer, backward, update, or DP-throughput measurement is performed.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import traceback


def validate_input(payload):
    from syncopate.infra.probability_probe import MODEL, MODEL_ID, validate_tokens
    if payload['model_path'] != str(MODEL) or payload['model'] != MODEL_ID:
        raise ValueError('unexpected model identity')
    rows = payload['tokens']
    validate_tokens(rows, payload['config']['text_config']['vocab_size'])
    if len(rows) != 2 or any(len(row) != 64 for row in rows) or rows[0] == rows[1]:
        raise ValueError('expected two distinct 64-token rows')
    if payload['positions'] != list(range(64)) or payload['attention_mask'] != [1] * 64:
        raise ValueError('positions or attention mask changed')
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    if digest != payload['tokens_sha256']:
        raise ValueError('token digest mismatch')
    return rows


def assert_root(trace, rows):
    expected_positions = [list(range(len(row))) for row in rows]
    if (trace['tokens'] != rows or trace['positions'] not in (expected_positions, [expected_positions] * 3, [expected_positions] * 4)
            or trace['attention_mask'] != [[1] * len(row) for row in rows]):
        raise ValueError('model root did not consume the exact local batch')


def validate_parameter_dtypes(inventory, fp32_names):
    """Only explicit upstream model declarations may justify FP32 parameters."""
    if not inventory or not any(item['ndim'] >= 2 for item in inventory.values()):
        raise ValueError('missing major model weights')
    for name, item in inventory.items():
        if item['dtype'] == 'torch.bfloat16':
            continue
        if item['dtype'] == 'torch.float32' and name in fp32_names and item['ndim'] < 2:
            continue
        raise ValueError('unexpected parameter dtype: ' + name + ': ' + item['dtype'])


def chosen_probabilities(values):
    # verl returns a rolled label for the final token too; omit that boundary.
    if len(values) != 64 or not all(math.isfinite(x) and x <= 0 for x in values):
        raise ValueError('expected 64 finite nonpositive engine log probabilities')
    return values[:-1]


def build_configs(model_path):
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, FSDPEngineConfig
    return dict(
        model_config=HFModelConfig(
            path=str(model_path), load_tokenizer=False, enable_gradient_checkpointing=False,
            use_remove_padding=False, use_fused_kernels=False, lora_rank=0,
            override_config={'attn_implementation': 'sdpa'}),
        engine_config=FSDPEngineConfig(
            strategy='fsdp2', fsdp_size=2, dtype='bf16', model_dtype='bf16',
            mixed_precision={'param_dtype': 'bf16', 'reduce_dtype': 'fp32', 'buffer_dtype': 'bf16'},
            use_dynamic_bsz=False, micro_batch_size_per_gpu=2, infer_micro_batch_size_per_gpu=2,
            use_torch_compile=False, forward_only=True, offload_policy=False,
            param_offload=False, optimizer_offload=False, grad_offload=False,
            use_remove_padding=False, use_fused_kernels=False),
        optimizer_config=None,
        checkpoint_config=CheckpointConfig(save_contents=['model'], load_contents=['model']))


def make_batch(rows):
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    values = {
        key: torch.nested.as_nested_tensor([torch.tensor(row, dtype=torch.long) for row in entries],
                                         layout=torch.jagged)
        for key, entries in {
            'input_ids': rows,
            'position_ids': [list(range(len(row))) for row in rows],
            'loss_mask': [[0] + [1] * (len(row) - 1) for row in rows],
        }.items()
    }
    values['temperature'] = torch.ones(len(rows), dtype=torch.float32)
    data = TensorDict(values, batch_size=[len(rows)])
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=len(rows),
                        use_remove_padding=False, use_fused_kernels=False,
                        return_model_output=True, calculate_entropy=False)
    return data


def cpu(input_file, output):
    """Target-image contract check: real verl config and input preparation, no CUDA."""
    import torch
    from types import SimpleNamespace
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.utils import tensordict_utils as tu
    from importlib.metadata import version
    inputs = json.loads(input_file.read_text())
    rows = validate_input(inputs)
    configs = build_configs(inputs['model_path'])
    assert configs['engine_config'].forward_only and configs['optimizer_config'] is None
    assert configs['model_config'].lora_rank == 0
    # Exercise the official stateless preparation method before paying for GPUs.
    shell = SimpleNamespace(pad_to_length=False, use_ulysses_sp=False)
    for batch in [rows[:1], rows]:
        data = make_batch(batch)
        assert tu.get_non_tensor_data(data, 'micro_batch_size_per_gpu', None) == len(batch)
        model_inputs, _ = FSDPEngineWithLMHead.prepare_model_inputs(shell, data)
        assert_root({name: model_inputs[key].tolist() for name, key in
                     [('tokens', 'input_ids'), ('positions', 'position_ids'),
                      ('attention_mask', 'attention_mask')]}, batch)
    assert not torch.cuda.is_initialized()
    result = {'ok': True, 'cuda_initialized': False, 'config_and_official_input_preparation': True,
              'versions': {p: version(p) for p in ['torch', 'transformers', 'verl', 'tensordict']}}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fsdp2-cpu.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def worker(input_file, output):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor
    from importlib.metadata import version
    from verl.workers.engine import EngineRegistry
    from syncopate.infra.probability_probe import compare_logprobs, verify_model

    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    result = {'rank': rank, 'local_rank': local_rank, 'ok': False, 'records': []}
    result_path = output / f'fsdp2-rank{rank}.json'
    try:
        inputs = json.loads(input_file.read_text())
        rows = validate_input(inputs)
        if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
            raise RuntimeError('exactly two visible GPUs and two ranks required')
        torch.cuda.set_device(local_rank)
        torch.set_num_threads(1)
        torch.manual_seed(137)
        if torch.cuda.get_device_capability() != (10, 0) or 'B200' not in torch.cuda.get_device_name():
            raise RuntimeError('this bounded worker requires B200')
        dist.init_process_group('nccl', timeout=timedelta(seconds=180),
                                device_id=torch.device('cuda', local_rank))
        # Rank zero validates the complete existing manifest before either rank loads.
        verification = [None]
        if rank == 0:
            try:
                verify_model(inputs)
                verification[0] = {'ok': True}
            except Exception:
                verification[0] = {'ok': False, 'error': traceback.format_exc()}
        dist.broadcast_object_list(verification, src=0)
        if not verification[0]['ok']:
            raise RuntimeError(verification[0])
        configs = build_configs(inputs['model_path'])
        engine = EngineRegistry.new(model_type='language_model', backend='fsdp2', **configs)
        engine.initialize()
        if (not isinstance(engine.module, FSDPModule) or engine.get_data_parallel_size() != 2
                or engine.optimizer is not None or engine.lr_scheduler is not None or engine._is_lora):
            raise RuntimeError('FSDP2 forward-only identity/optimizer contract failed')
        params = list(engine.module.named_parameters())
        sharded = [(name, p) for name, p in params if isinstance(p, DTensor)]
        if not sharded or any('lora_' in name for name, _ in params):
            raise RuntimeError('missing real sharded parameters or unexpected LoRA')
        inventory = {name: {'dtype': str(p.dtype), 'ndim': p.ndim, 'shape': list(p.shape)}
                     for name, p in params if p.is_floating_point()}
        declarations = []
        allowed_fp32 = set()
        for prefix, module in engine.module.named_modules():
            for attribute in ['_keep_in_fp32_modules', '_keep_in_fp32_modules_strict']:
                for pattern in getattr(module, attribute, None) or []:
                    declarations.append({'module': prefix, 'attribute': attribute, 'name': pattern})
                    # Match a declared module/parameter path component, never fuzzy substrings.
                    for name in inventory:
                        relative = name[len(prefix) + 1:] if prefix and name.startswith(prefix + '.') else name
                        if (not prefix or name.startswith(prefix + '.')) and (
                                relative == pattern or relative.endswith('.' + pattern)
                                or ('.' + pattern + '.') in ('.' + relative + '.')):
                            allowed_fp32.add(name)
        result.update(parameter_dtypes=inventory, upstream_fp32_declarations=declarations,
                      allowed_fp32_parameters=sorted(allowed_fp32))
        validate_parameter_dtypes(inventory, allowed_fp32)
        result.update(
            engine='fsdp2', engine_class=f'{type(engine).__module__}.{type(engine).__name__}',
            model_class=type(engine.module).__name__, tokens_sha256=inputs['tokens_sha256'],
            revision=inputs['revision'], input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
            device={'name': torch.cuda.get_device_name(), 'index': torch.cuda.current_device(),
                    'capability': list(torch.cuda.get_device_capability())},
            versions={p: version(p) for p in ['torch', 'transformers', 'verl', 'tensordict']},
            source_sha256={inspect.getfile(type(engine)): hashlib.sha256(
                Path(inspect.getfile(type(engine))).read_bytes()).hexdigest()},
            sharded_parameter_count=len(sharded), optimizer_absent=True,
            cpu_offload_policy=bool(getattr(engine, '_uses_fsdp2_cpu_offload_policy', False)),
            data_distribution='identical local batches on both DP ranks; duplicated work, not throughput',
            shard_example={'name': sharded[0][0], 'global_shape': list(sharded[0][1].shape),
                           'local_shape': list(sharded[0][1].to_local().shape),
                           'placements': [str(x) for x in sharded[0][1].placements]})
        traces = []
        def observe(module, args, kwargs):
            trace = {name: kwargs[key].detach().cpu().tolist() for name, key in
                     [('tokens', 'input_ids'), ('positions', 'position_ids'), ('attention_mask', 'attention_mask')]}
            traces.append(trace)
            with (output / f'fsdp2-root-rank{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(trace) + '\n')
        hook = engine.module.register_forward_pre_hook(observe, with_kwargs=True)
        for batch in [rows[:1]] * 3 + [rows] * 3:
            before = len(traces)
            with engine.eval_mode(), torch.no_grad():
                actual = engine.infer_batch(make_batch(batch), loss_function=None)
            if len(traces) != before + 1:
                raise RuntimeError('requested batch was split or root forward was bypassed')
            assert_root(traces[-1], batch)
            probabilities = actual['model_output']['log_probs']
            if len(probabilities.unbind()) != len(batch):
                raise RuntimeError('output batch size mismatch')
            lp = chosen_probabilities(probabilities.unbind()[0].detach().float().cpu().tolist())
            copies = [None] * 2
            dist.all_gather_object(copies, lp)
            rank_difference=compare_logprobs(*copies)
            if rank_difference['max_abs'] > 1e-5:
                raise RuntimeError('identical local batches disagree across ranks: '+str(rank_difference))
            result['records'].append({'batch_size': len(batch), 'logprobs': lp,
                                      'observed_same_batch': True,
                                      'cross_rank': compare_logprobs(*copies)})
            result_path.write_text(json.dumps(result, indent=2) + '\n')
        hook.remove()
        result.update(ok=True, peak_allocated=torch.cuda.max_memory_allocated(),
                      updates=0, backward_calls=0)
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        if dist.is_initialized():
            dist.destroy_process_group()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (cpu if args.cpu else worker)(args.input, args.output)

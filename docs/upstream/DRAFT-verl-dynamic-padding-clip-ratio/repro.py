"""CPU-only: identical response lengths, different padding, different clip_ratio.

No model, dataset, generation, or project imports. This is a metric reproduction,
not proof that every verl caller supplies dynamically padded tensors.
"""
import json

import torch
from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics


def measure(width):
    lengths = torch.tensor([2, 4])
    mask = torch.arange(width)[None, :] < lengths[:, None]
    prompts = torch.ones(2, 3, dtype=torch.long)
    tensors = {
        "prompts": prompts, "responses": torch.zeros(2, width, dtype=torch.long),
        "attention_mask": torch.cat([torch.ones_like(prompts), mask.long()], dim=1),
        "response_mask": mask,
        **{key: torch.ones(2, width) for key in
           ("token_level_scores", "token_level_rewards", "advantages", "returns")},
    }
    data = DataProto.from_dict(tensors=tensors)
    metrics = compute_data_metrics(data, use_critic=False)
    return {key: metrics[key] for key in ("response_length/max", "response_length/clip_ratio")}


def main():
    torch.set_num_threads(2)
    dynamic, fixed = measure(4), measure(12)
    result = {"response_lengths": [2, 4], "registered_response_budget": 12,
              "dynamic_padding_width_4": dynamic, "fixed_padding_width_12": fixed}
    print(json.dumps(result))
    assert dynamic["response_length/max"] == fixed["response_length/max"] == 4
    assert dynamic["response_length/clip_ratio"] == 0.5
    assert fixed["response_length/clip_ratio"] == 0.0
    return result


if __name__ == "__main__":
    main()

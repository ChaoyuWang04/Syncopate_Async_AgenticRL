import copy
import hashlib
import json
import unittest

from syncopate.infra.fsdp_probability_probe import assert_root, chosen_probabilities, validate_input, validate_parameter_dtypes
from syncopate.infra.probability_probe import MODEL, MODEL_ID


class FSDPProbabilityContract(unittest.TestCase):
    def payload(self):
        rows = [list(range(64)), list(range(1, 65))]
        return dict(model_path=str(MODEL), model=MODEL_ID, tokens=rows,
                    config={'text_config': {'vocab_size': 100}}, positions=list(range(64)),
                    attention_mask=[1]*64, tokens_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest())

    def test_input_rejects_identity_digest_shape_and_masks(self):
        original = self.payload()
        self.assertEqual(validate_input(original), original['tokens'])
        for key, value in [('model', 'other'), ('tokens_sha256', 'bad'),
                           ('positions', [0]*64), ('attention_mask', [0]*64),
                           ('tokens', [list(range(64))]*2)]:
            bad = copy.deepcopy(original)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_input(bad)

    def test_root_rejects_split_reordered_masked_or_shifted_batch(self):
        rows = self.payload()['tokens']
        trace = dict(tokens=rows, positions=[list(range(64))]*2, attention_mask=[[1]*64]*2)
        assert_root(trace, rows)
        assert_root(dict(trace, positions=[trace['positions']]*3), rows)
        for key, value in [('tokens', rows[:1]), ('tokens', rows[::-1]),
                           ('positions', [list(range(1,65))]*2), ('attention_mask', [[0]*64]*2)]:
            bad = copy.deepcopy(trace)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_root(bad, rows)

    def test_excludes_only_last_cross_sequence_label(self):
        values = [-float(x) for x in range(64)]
        self.assertEqual(chosen_probabilities(values), values[:63])
        for bad in [values[:-1], values+[0], values[:-1]+[float('nan')], values[:-1]+[1]]:
            with self.assertRaises(ValueError):
                chosen_probabilities(bad)

    def test_dtype_requires_explicit_fp32_exception(self):
        inventory = {'embed.weight': {'dtype': 'torch.bfloat16', 'ndim': 2},
                     'layer.A_log': {'dtype': 'torch.float32', 'ndim': 1}}
        validate_parameter_dtypes(inventory, {'layer.A_log'})
        with self.assertRaises(ValueError):
            validate_parameter_dtypes(inventory, set())
        inventory['embed.weight']['dtype'] = 'torch.float32'
        with self.assertRaises(ValueError):
            validate_parameter_dtypes(inventory, {'layer.A_log', 'embed.weight'})

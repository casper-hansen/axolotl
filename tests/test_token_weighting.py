"""
Test token-based dataset weighting functionality
"""

import unittest
from unittest.mock import patch

from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.utils.data.token_weighting import _count_tokens, merge_datasets
from axolotl.utils.dict import DictDefault


class TestTokenWeighting(unittest.TestCase):
    """Test token-based dataset weighting"""

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        
        self.dataset1 = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4], [5, 6, 7], [8, 9]],  # 4 + 3 + 2 = 9 tokens
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1], [1, 1]],
            "labels": [[1, 2, 3, 4], [5, 6, 7], [8, 9]]
        })
        
        self.dataset2 = Dataset.from_dict({
            "input_ids": [[10, 11], [12, 13, 14, 15]],  # 2 + 4 = 6 tokens
            "attention_mask": [[1, 1], [1, 1, 1, 1]],
            "labels": [[10, 11], [12, 13, 14, 15]]
        })
        
        self.cfg = DictDefault({
            "seed": 42,
            "shuffle_merged_datasets": True
        })

    def test_count_tokens_exact(self):
        """Test exact token counting for small datasets"""
        count = _count_tokens(self.dataset1)
        self.assertEqual(count, 9)
        
        count = _count_tokens(self.dataset2)
        self.assertEqual(count, 6)

    def test_count_tokens_estimated(self):
        """Test estimated token counting for large datasets"""
        large_dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3]] * 3000,  # 9000 tokens total
            "attention_mask": [[1, 1, 1]] * 3000,
            "labels": [[1, 2, 3]] * 3000
        })
        
        count = _count_tokens(large_dataset, sample_size=100)
        self.assertGreater(count, 8000)
        self.assertLess(count, 10000)

    def test_merge_datasets_no_weights(self):
        """Test merging datasets without weights (should behave like normal concatenation)"""
        datasets = [self.dataset1, self.dataset2]
        configs = [
            DictDefault({"path": "dataset1"}),
            DictDefault({"path": "dataset2"})
        ]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        self.assertEqual(len(merged), 5)  # 3 + 2 samples

    def test_merge_datasets_upsample(self):
        """Test upsampling with weight > 1"""
        datasets = [self.dataset1, self.dataset2]
        configs = [
            DictDefault({"path": "dataset1", "weight": 2.0, "weight_strategy": "upsample"}),
            DictDefault({"path": "dataset2"})
        ]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        self.assertGreater(len(merged), 5)

    def test_merge_datasets_downsample(self):
        """Test downsampling with weight < 1"""
        datasets = [self.dataset1, self.dataset2]
        configs = [
            DictDefault({"path": "dataset1", "weight": 0.5, "weight_strategy": "downsample"}),
            DictDefault({"path": "dataset2"})
        ]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        self.assertLess(len(merged), 5)

    def test_merge_datasets_invalid_strategy(self):
        """Test handling of invalid weight strategy"""
        datasets = [self.dataset1]
        configs = [
            DictDefault({"path": "dataset1", "weight": 2.0, "weight_strategy": "invalid"})
        ]
        
        with patch('axolotl.utils.data.token_weighting.LOG') as mock_log:
            merged = merge_datasets(datasets, configs, self.cfg)
            mock_log.warning.assert_called()
            self.assertEqual(len(merged), 3)  # Should use original dataset

    def test_single_dataset_passthrough(self):
        """Test that single dataset is passed through unchanged"""
        datasets = [self.dataset1]
        configs = [DictDefault({"path": "dataset1"})]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged, self.dataset1)

    def test_weight_one_passthrough(self):
        """Test that datasets with weight=1 are passed through unchanged"""
        datasets = [self.dataset1, self.dataset2]
        configs = [
            DictDefault({"path": "dataset1", "weight": 1.0, "weight_strategy": "upsample"}),
            DictDefault({"path": "dataset2", "weight": 1.0, "weight_strategy": "downsample"})
        ]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        self.assertEqual(len(merged), 5)  # 3 + 2 samples, no change

    def test_downsample_weight_greater_than_one_warning(self):
        """Test that downsampling with weight >= 1 issues a warning and uses original dataset"""
        datasets = [self.dataset1]
        configs = [
            DictDefault({"path": "dataset1", "weight": 1.5, "weight_strategy": "downsample"})
        ]
        
        with patch('axolotl.utils.data.token_weighting.LOG') as mock_log:
            merged = merge_datasets(datasets, configs, self.cfg)
            mock_log.warning.assert_called()
            self.assertEqual(len(merged), 3)  # Should use original dataset

    def test_upsample_fractional_weight(self):
        """Test upsampling with fractional weight > 1"""
        datasets = [self.dataset1]
        configs = [
            DictDefault({"path": "dataset1", "weight": 1.5, "weight_strategy": "upsample"})
        ]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        self.assertGreaterEqual(len(merged), 3)

    def test_merge_datasets_preserves_features(self):
        """Test that merged datasets preserve the original features"""
        datasets = [self.dataset1, self.dataset2]
        configs = [
            DictDefault({"path": "dataset1", "weight": 2.0, "weight_strategy": "upsample"}),
            DictDefault({"path": "dataset2"})
        ]
        
        merged = merge_datasets(datasets, configs, self.cfg)
        
        expected_features = {"input_ids", "attention_mask", "labels"}
        self.assertEqual(set(merged.features.keys()), expected_features)

    def test_shuffle_behavior(self):
        """Test that shuffling works correctly when enabled"""
        datasets = [self.dataset1, self.dataset2]
        configs = [
            DictDefault({"path": "dataset1"}),
            DictDefault({"path": "dataset2"})
        ]
        
        cfg_shuffle = DictDefault({
            "seed": 42,
            "shuffle_merged_datasets": True
        })
        
        merged_shuffled = merge_datasets(datasets, configs, cfg_shuffle)
        
        cfg_no_shuffle = DictDefault({
            "seed": 42,
            "shuffle_merged_datasets": False
        })
        
        merged_no_shuffle = merge_datasets(datasets, configs, cfg_no_shuffle)
        
        self.assertEqual(len(merged_shuffled), len(merged_no_shuffle))
        self.assertEqual(len(merged_shuffled), 5)  # 3 + 2 samples


if __name__ == "__main__":
    unittest.main()

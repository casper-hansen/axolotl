"""
Token-based dataset weighting utilities
"""

import logging
from math import floor
from typing import List

from datasets import Dataset, concatenate_datasets

from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def _count_tokens(ds: Dataset, sample_size: int = 2048) -> int:
    """
    Return the *exact* number of tokens if the dataset is small enough,
    otherwise estimate it from a random sample (saves RAM for huge corpora).
    """
    if len(ds) <= sample_size:
        return sum(len(ids) for ids in ds["input_ids"])

    sample = ds.shuffle(seed=42).select(range(sample_size))
    avg_len = sum(len(ids) for ids in sample["input_ids"]) / sample_size
    return int(avg_len * len(ds))


def merge_datasets(
    datasets: List[Dataset],
    datasets_configs: List,
    cfg: DictDefault,
) -> Dataset:
    """
    Merge several HF datasets into one, honouring per-dataset weights *in tokens*.

    Each entry in datasets_configs may contain:
      weight           (float | int, default = 1)
      weight_strategy  ('upsample' | 'downsample', default = 'upsample')

    The effective target token count for a dataset is:
        target_tokens = original_tokens × weight
    """
    LOG.info(f"merge_datasets called with {len(datasets)} datasets")
    for i, d_cfg in enumerate(datasets_configs):
        LOG.info(f"Dataset {i}: weight={getattr(d_cfg, 'weight', 'NOT_SET')}, strategy={getattr(d_cfg, 'weight_strategy', 'NOT_SET')}")

    has_weighting = any(
        getattr(d_cfg, "weight", None) is not None or 
        getattr(d_cfg, "weight_strategy", None) is not None 
        for d_cfg in datasets_configs
    )
    
    if len(datasets) == 1 and not has_weighting:
        LOG.info("Single dataset with no weighting - returning unchanged")
        return datasets[0]

    LOG.info("Merging datasets with token-based weighting …")

    weighted_parts: list[Dataset] = []

    for ds, d_cfg in zip(datasets, datasets_configs):
        weight = getattr(d_cfg, "weight", None)
        strategy = getattr(d_cfg, "weight_strategy", None)
        
        if weight is None and strategy is None:
            weighted_parts.append(ds)
            continue
            
        weight = float(weight or 1.0)
        if strategy is None:
            strategy = "upsample"
        strategy = strategy.lower()

        LOG.info(f"Processing dataset with weight={weight}, strategy={strategy}, original_size={len(ds)}")

        if weight == 1:
            weighted_parts.append(ds)
            continue

        tok_cnt = _count_tokens(ds)
        target_tok = max(1, int(tok_cnt * weight))
        LOG.info(f"Token count: {tok_cnt}, target tokens: {target_tok}")

        if strategy == "upsample":
            repeats = max(1, floor(target_tok / tok_cnt))
            LOG.info(f"Upsampling: repeats={repeats}")
            weighted_parts.extend([ds] * repeats)

            remaining_tok = target_tok - repeats * tok_cnt
            if remaining_tok:
                avg_len = max(1, tok_cnt // len(ds))
                n_extra = min(len(ds), int(remaining_tok / avg_len) + 1)
                LOG.info(f"Adding {n_extra} extra samples for remaining {remaining_tok} tokens")

                extra = (
                    ds.shuffle(seed=cfg.seed)
                    .select(range(n_extra))
                )
                weighted_parts.append(extra)

        elif strategy == "downsample":
            if weight >= 1:
                LOG.warning(
                    f"Ignoring downsample weight ≥1 for dataset "
                    f"{getattr(d_cfg, 'path', '<unknown>')}."
                )
                weighted_parts.append(ds)
                continue

            target_tok = max(1, int(tok_cnt * weight))
            avg_len = max(1, tok_cnt // len(ds))
            n_keep = max(1, int(target_tok / avg_len))
            LOG.info(f"Downsampling: keeping {n_keep} samples out of {len(ds)}")

            sampled = (
                ds.shuffle(seed=cfg.seed)
                .select(range(n_keep))
            )
            weighted_parts.append(sampled)

        else:
            LOG.warning(
                f"Unknown weight_strategy '{strategy}' "
                f"for dataset {getattr(d_cfg, 'path', '<unknown>')}. "
                "Using dataset without weighting."
            )
            weighted_parts.append(ds)

    merged = concatenate_datasets(weighted_parts)

    if cfg.shuffle_merged_datasets:
        merged = merged.shuffle(seed=cfg.seed)

    return merged

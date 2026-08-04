"""DeepSense 6G V2I streaming beam-tracking data package.

Public surface:
    load_data_config(path)            -> DataConfig
    build_sample_index(cfg, scenario) -> pandas.DataFrame of samples
    build_episodes(cfg, ...)          -> list[Episode]
    resolve_splits(cfg)               -> Splits (frozen, leakage-free)
    SnapshotDataset / StreamingEpisodeDataset (torch; import lazily)

Guardrails live in `splits.py` / `paths.py`: scenario 34 is hard-blocked unless
`allow_heldout=True`, scenario 33 is never in train/val.
"""
from .config import DataConfig, load_data_config

__all__ = ["DataConfig", "load_data_config"]

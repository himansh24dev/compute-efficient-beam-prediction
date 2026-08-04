"""Build a unified, validated sample index from scenario CSVs and group it into
streaming episodes (one vehicle traversal = one episode, keyed by seq_index).

A "sample" is one time-ordered multimodal frame. An "episode" is a contiguous
run of samples with the same seq_index, sorted in time. Episodes are the atomic
unit of the train/val/test split -> no leakage across the split boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DataConfig
from .paths import scenario_csv


@dataclass(frozen=True)
class Episode:
    scenario_id: int
    episode_id: int          # seq_index value
    row_start: int           # inclusive index into the scenario DataFrame
    row_end: int             # exclusive
    length: int

    @property
    def key(self) -> str:
        """Globally unique, stable episode key used for split assignment."""
        return f"s{self.scenario_id:02d}_e{self.episode_id:03d}"


def build_sample_index(
    cfg: DataConfig, scenario_id: int, allow_heldout: bool = False
) -> pd.DataFrame:
    """Load a scenario CSV into a normalized, time-sorted sample table.

    Adds a 0-indexed `beam_label` column ([0, num_beams-1]) and a `scenario_id`
    column, and sorts by (episode, index) so episodes are contiguous & ordered.
    """
    csv_path = scenario_csv(cfg, scenario_id, allow_heldout=allow_heldout)
    cols = cfg.columns
    df = pd.read_csv(csv_path)

    required = [cols["index"], cols["episode"], cols["beam"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"scenario {scenario_id} CSV missing columns {missing}")

    df = df.copy()
    df["scenario_id"] = int(scenario_id)

    # 1-indexed beam label in the CSV -> 0-indexed class id.
    base = int(cfg.beam["label_base"])
    n_beams = int(cfg.beam["num_beams"])
    df["beam_label"] = df[cols["beam"]].astype(int) - base
    bad = df[(df["beam_label"] < 0) | (df["beam_label"] >= n_beams)]
    if len(bad):
        raise ValueError(
            f"scenario {scenario_id}: {len(bad)} beam labels outside [0,{n_beams - 1}] "
            f"after subtracting label_base={base}"
        )

    sort_key = cfg.streaming.get("sort_within_episode_by", cols["index"])
    df = df.sort_values([cols["episode"], sort_key]).reset_index(drop=True)
    return df


def build_episodes(
    cfg: DataConfig, scenario_id: int, allow_heldout: bool = False
) -> tuple[pd.DataFrame, list[Episode]]:
    """Return (sample_index_df, episodes). Episodes shorter than
    streaming.min_episode_len are dropped (degenerate passes)."""
    df = build_sample_index(cfg, scenario_id, allow_heldout=allow_heldout)
    ep_col = cfg.columns["episode"]
    min_len = int(cfg.streaming.get("min_episode_len", 1))

    episodes: list[Episode] = []
    # df is already sorted by (episode, sort_key); contiguous groups.
    start = 0
    n = len(df)
    ep_values = df[ep_col].to_numpy()
    while start < n:
        end = start + 1
        while end < n and ep_values[end] == ep_values[start]:
            end += 1
        length = end - start
        if length >= min_len:
            episodes.append(
                Episode(
                    scenario_id=int(scenario_id),
                    episode_id=int(ep_values[start]),
                    row_start=start,
                    row_end=end,
                    length=length,
                )
            )
        start = end
    return df, episodes


def validate_continuity(cfg: DataConfig, df: pd.DataFrame, episodes: list[Episode]) -> dict:
    """Cheap sanity report used by scripts/validate_data.py.

    Checks (per scenario): no duplicate sample `index`, monotonic time within
    episodes is not enforced (timestamps wrap at midnight), and reports episode
    length distribution.
    """
    idx_col = cfg.columns["index"]
    report = {
        "n_samples": int(len(df)),
        "n_episodes": len(episodes),
        "duplicate_indices": int(df[idx_col].duplicated().sum()),
        "episode_len_min": min((e.length for e in episodes), default=0),
        "episode_len_max": max((e.length for e in episodes), default=0),
        "episode_len_mean": (sum(e.length for e in episodes) / len(episodes)) if episodes else 0.0,
    }
    return report

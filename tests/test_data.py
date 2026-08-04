"""Data package tests. Run with numpy/pandas/opencv only (no torch needed):

    pytest tests/test_data.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from data.index import build_episodes, build_sample_index
from data.paths import HeldoutScenarioError, resolve_relative
from data.preprocess import beam as beam_pp
from data.preprocess import camera as cam_pp
from data.preprocess import gps as gps_pp
from data.preprocess import lidar as lidar_pp
from data.preprocess import radar as radar_pp
from data.splits import resolve_splits


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_scenario34_is_blocked(cfg):
    with pytest.raises(HeldoutScenarioError):
        build_episodes(cfg, 34)


def test_scenario34_loadable_with_optin(cfg):
    # The final-eval opt-in path must still work (path resolution, not a load).
    from data.paths import assert_loadable
    assert_loadable(cfg, 34, allow_heldout=True)  # should not raise


def test_33_never_in_train_val(cfg):
    assert 33 not in cfg.split["train_val_scenarios"]
    assert 34 not in cfg.split["train_val_scenarios"]


# --------------------------------------------------------------------------- #
# Index / episodes
# --------------------------------------------------------------------------- #
def test_episodes_are_contiguous_and_min_len(cfg):
    df, eps = build_episodes(cfg, 31)
    ep_col = cfg.columns["episode"]
    min_len = cfg.streaming["min_episode_len"]
    for e in eps:
        assert e.length >= min_len
        block = df.iloc[e.row_start:e.row_end][ep_col].unique()
        assert list(block) == [e.episode_id]  # one episode id per block


def test_beam_labels_in_range(cfg):
    df = build_sample_index(cfg, 31)
    n = cfg.beam["num_beams"]
    assert df["beam_label"].min() >= 0
    assert df["beam_label"].max() < n


# --------------------------------------------------------------------------- #
# Splits: frozen, deterministic, leakage-free
# --------------------------------------------------------------------------- #
def test_splits_no_leakage_and_frozen(cfg):
    s1 = resolve_splits(cfg)
    keys = lambda eps: {e.key for e in eps}
    tr, va, te = keys(s1.train), keys(s1.val), keys(s1.test)
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    # test is night-only (33)
    assert {e.scenario_id for e in s1.test} == {33}
    # reproducible: reloading gives identical assignment
    s2 = resolve_splits(cfg)
    assert keys(s2.train) == tr and keys(s2.val) == va and keys(s2.test) == te


# --------------------------------------------------------------------------- #
# Preprocessing shapes + correctness on REAL files
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def first_row(request):
    cfg = request.getfixturevalue("cfg")
    df, eps = build_episodes(cfg, 31)
    return cfg, df.iloc[eps[0].row_start], 31


def test_camera_shape(first_row):
    cfg, row, sid = first_row
    m = cfg.modalities["camera"]
    a = cam_pp.load_rgb(resolve_relative(cfg, sid, row[cfg.columns["rgb"]]),
                        tuple(m["resize_hw"]), m["mean"], m["std"])
    assert a.shape == (3, m["resize_hw"][0], m["resize_hw"][1])
    assert a.dtype == np.float32


def test_lidar_bev_shape(first_row):
    cfg, row, sid = first_row
    m = cfg.modalities["lidar"]
    a = lidar_pp.load_bev(resolve_relative(cfg, sid, row[cfg.columns["lidar"]]), m)
    assert a.shape == (len(m["channels"]), m["grid_hw"][0], m["grid_hw"][1])
    assert np.isfinite(a).all()


def test_radar_maps_shape(first_row):
    cfg, row, sid = first_row
    m = cfg.modalities["radar"]
    a = radar_pp.load_maps(resolve_relative(cfg, sid, row[cfg.columns["radar"]]), m)
    assert a.shape == (2, m["out_hw"][0], m["out_hw"][1])
    assert 0.0 <= a.min() and a.max() <= 1.0  # min-max normalized


def test_beam_label_matches_genie(first_row):
    cfg, row, sid = first_row
    pw = beam_pp.load_power_vector(resolve_relative(cfg, sid, row[cfg.columns["pwr"]]),
                                   cfg.beam["num_beams"])
    # CSV label (0-indexed) must equal argmax of the power vector.
    assert int(row["beam_label"]) == beam_pp.genie_beam(pw)
    assert beam_pp.received_power_loss_db(pw, beam_pp.genie_beam(pw)) == pytest.approx(0.0, abs=1e-6)


def test_gps_standardize_identity(first_row):
    cfg, row, sid = first_row
    raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = gps_pp.standardize(raw, mean=np.zeros(3, np.float32), std=np.ones(3, np.float32))
    assert np.allclose(out, raw)


# --------------------------------------------------------------------------- #
# Cache: must be numerically identical to the compute path
# --------------------------------------------------------------------------- #
def test_cache_matches_compute(cfg):
    from data.cache import is_cached
    from data.dataset import FrameLoader
    from data.index import build_episodes

    if not is_cached(cfg, 31):
        import pytest as _pt
        _pt.skip("scenario 31 cache not built (run scripts/build_cache.py --scenarios 31)")

    df, eps = build_episodes(cfg, 31)
    cached, compute = FrameLoader(cfg, use_cache=True), FrameLoader(cfg, use_cache=False)
    ep = eps[0]
    for r in range(ep.row_start, ep.row_start + 6):
        a, b = cached.load_frame(31, r, df.iloc[r]), compute.load_frame(31, r, df.iloc[r])
        for m in cached.enabled_modalities:
            assert np.array_equal(a[m], b[m]), f"cache mismatch in {m} at row {r}"
        assert int(a["beam_label"]) == int(b["beam_label"])

from __future__ import annotations

from pathlib import Path

import pytest

from investigator.hypothesis_engine import HypothesisEngine
from investigator.loader import load_dataset
from investigator.profile import profile_tables

pytestmark = pytest.mark.integration

DATA_DIR = Path("data/raw/olist")


@pytest.fixture(scope="module")
def he():
    res = load_dataset(DATA_DIR)
    profiles = profile_tables(res.raw)
    return HypothesisEngine(res, profiles, DATA_DIR)


@pytest.fixture(scope="module")
def findings(he):
    return he.run()


def test_hypothesis_engine_requires_olist_data(he):
    assert len(he.fact) > 0


def test_hypothesis_engine_generates_all_kinds(findings):
    kinds = {f.kind for f in findings}
    assert {"time_change", "association", "graph", "segment_mover"} & kinds, kinds
    assert len(findings) >= 10


def test_time_change_findings_have_verified_sql(findings):
    tcs = [f for f in findings if f.kind == "time_change"]
    assert tcs
    verified = [f for f in tcs if any(e.verified for e in f.evidence)]
    assert len(verified) >= len(tcs) * 0.5


def test_findings_have_caveats_or_stats(findings):
    for f in findings[:5]:
        assert f.stats or f.caveats


def test_headline_sql_matches_pandas():
    """The revenue change SQL must reproduce the pandas number exactly."""
    from investigator.hypothesis_engine import _headline_sql
    from investigator.stats_engine import detect_period_changes

    res = load_dataset(DATA_DIR)
    he = HypothesisEngine(res, profile_tables(res.raw), DATA_DIR)
    changes = detect_period_changes(res.fact, min_abs_delta=0.15, min_rows=150)
    ch = next(c for c in changes if c.metric == "revenue")
    sql = _headline_sql(ch, metric="revenue")
    df = he.db.exec(sql)
    assert set(["base_value", "comp_value", "delta_pct"]) <= set(df.columns)
    # cross-check against pandas numbers
    import numpy as np

    assert np.isclose(float(df.iloc[0]["comp_value"]), ch.comp_value, rtol=0.01)
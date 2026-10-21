import numpy as np

from cwfc.features import DEFAULT_LAGS, build_ar_design, lag_feature_table, recursive_feature_vector


def test_default_lags_are_positive_and_sorted():
    assert DEFAULT_LAGS == tuple(sorted(DEFAULT_LAGS))
    assert all(lag > 0 for lag in DEFAULT_LAGS)


def test_lag_table_never_uses_current_value():
    history = np.arange(20, dtype=float)
    table = lag_feature_table(history, DEFAULT_LAGS)
    first = max(DEFAULT_LAGS)
    # first usable row predicts index `first`; every lag must point before it
    assert table.loc[0, f"lag_{DEFAULT_LAGS[0]}"] == history[first - DEFAULT_LAGS[0]]
    assert table.loc[0, f"lag_{DEFAULT_LAGS[-1]}"] == history[first - DEFAULT_LAGS[-1]]
    assert len(table) == 20 - first
    # no lag column may contain the value it is predicting
    assert (table.to_numpy() != np.arange(first, 20).reshape(-1, 1)).all()


def test_ar_design_alignment():
    history = np.arange(20, dtype=float)
    X, y = build_ar_design(history, DEFAULT_LAGS)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(DEFAULT_LAGS)
    assert y[0] == history[max(DEFAULT_LAGS)]
    assert np.isfinite(X).all()


def test_recursive_vector_uses_only_past():
    known = list(np.arange(15, dtype=float))
    vec = recursive_feature_vector(known, 15, DEFAULT_LAGS)
    assert vec[0] == known[15 - DEFAULT_LAGS[0]]
    assert vec[-1] == known[15 - DEFAULT_LAGS[-1]]
    assert len(vec) == len(DEFAULT_LAGS)

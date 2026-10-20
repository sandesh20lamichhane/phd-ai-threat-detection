import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean
from src.data.leakage import assert_no_leakage
from src.data.split import split_by_group, split_by_time


def frame():
    return pd.DataFrame({
        "Flow ID": ["1", "2", "3", "3"], " Source IP": ["a", "b", "c", "c"], "Destination Port": [80, 80, 443, 443],
        "Timestamp": ["t1", "t2", "t3", "t3"], "Flow Duration": [1.0, np.inf, 3.0, 3.0],
        "Day": ["Mon", "Tue", "Wed", "Wed"], "Label": ["BENIGN", "DoS", "BENIGN", "BENIGN"]})


def test_clean_drops_identifiers_and_duplicates():
    df, rep = clean(frame(), keep=["Day"])
    assert set(rep["identifier_columns_dropped"]) == {"Flow ID", "Source IP", "Destination Port", "Timestamp"}
    assert rep["duplicate_rows_dropped"] == 1 and rep["inf_values_to_nan"] == 1
    assert list(df.columns) == ["Flow Duration", "Day", "Label"]


def test_splits():
    df, _ = clean(frame(), keep=["Day"])
    tr, va, te = split_by_group(df, "Day", ["Mon"], ["Tue"], ["Wed"])
    assert len(tr) == len(va) == len(te) == 1
    with pytest.raises(ValueError):
        split_by_group(df, "Day", ["Mon"], ["Mon"], ["Wed"])
    a, b = split_by_time(pd.DataFrame({"t": [1, 2, 3]}), "t", 2)
    assert list(a.t) == [1] and list(b.t) == [2, 3]


def test_leakage_guard():
    df, _ = clean(frame(), keep=["Day"])
    with pytest.raises(AssertionError):
        assert_no_leakage(df, df)
    with pytest.raises(AssertionError):
        assert_no_leakage(frame(), frame().iloc[:0])

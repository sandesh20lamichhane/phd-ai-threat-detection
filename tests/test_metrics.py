import numpy as np
import pandas as pd

from src.evaluation.metrics import detection_lead_time, detection_report, fpr_at_detection_rate, macro_f1
from src.evaluation.response_metrics import response_summary
from src.evaluation.stats import friedman_nemenyi, summarise, wilcoxon


def test_macro_f1_punishes_majority_guess():
    y = np.array(["BENIGN"] * 90 + ["DoS"] * 10)
    assert macro_f1(y, np.array(["BENIGN"] * 100)) < 0.5  # accuracy would be 0.90


def test_report_and_fpr():
    y = np.array(["BENIGN", "BENIGN", "DoS", "DoS"])
    rep = detection_report(y, y, attack_scores=[0.1, 0.2, 0.8, 0.9])
    assert rep["macro_f1"] == 1.0 and rep["pr_auc"] == 1.0
    assert fpr_at_detection_rate([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.95], 1.0) == 0.5


def test_lead_time():
    out = detection_lead_time({"a": [5, 50, 200], "b": [300]}, {"a": 100, "b": 100})
    assert out == {"a": 95.0, "b": None}


def test_stats():
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"A": rng.normal(0.9, 0.01, 10), "B": rng.normal(0.8, 0.01, 10), "C": rng.normal(0.7, 0.01, 10)})
    res = friedman_nemenyi(table)
    assert res["p_value"] < 0.01 and ("A", "C", res["avg_rank"]["C"] - res["avg_rank"]["A"]) in res["significant_pairs"]
    assert wilcoxon(table["A"], table["B"])["p_value"] < 0.01
    assert summarise({"A": [1, 2, 3]}).loc["A", "mean"] == 2


def test_response_summary():
    s = response_summary([{"attack_start": 0, "contained_at": 10, "disruption": 1},
                          {"attack_start": 0, "contained_at": None, "disruption": 3}])
    assert s["containment_rate"] == 0.5 and s["mean_time_to_contain"] == 10 and s["mean_disruption_cost"] == 2

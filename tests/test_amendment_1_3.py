"""Amendment 1.3: external S7 against source train + val, and checksums enforced on every S7 artifact."""
import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.utils import paths
from tests.synthetic import write_cic, write_unsw


def _script():
    spec = importlib.util.spec_from_file_location("preprocess_script", paths.REPO / "scripts" / "preprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def built(drive, tmp_path, monkeypatch):
    """Synthetic CICIDS2017 and UNSW-NB15, processed, with S7 records for both (no CSE-CIC-IDS2018)."""
    from src.data import harmonise, near
    from src.data.pipeline import preprocess
    write_cic(drive / "data" / "raw")
    write_unsw(drive / "data" / "interim")
    preprocess("cicids2017_improved", log=lambda *a: None)
    preprocess("unsw_nb15", log=lambda *a: None)
    mdir = tmp_path / "mapping"
    mdir.mkdir()
    shutil.copy(harmonise.MAPPING_DIR / "cic_to_unsw_mapping.yaml", mdir)
    (mdir / "cic_shared_features.yaml").write_text("features: []\n")
    monkeypatch.setattr(harmonise, "MAPPING_DIR", mdir)
    monkeypatch.setattr(near, "WITHIN", ["cicids2017_improved"])
    monkeypatch.setattr(near, "EXTERNAL", {"unsw_nb15": "cicids2017_improved"})
    near._checked.clear()
    _script().cross_overlap()
    near.build("cicids2017_improved", log=lambda *a: None)
    return drive


def test_external_seen_means_source_train_or_val(built):
    from src.data import datasets, harmonise, near
    rec = json.loads((paths.RESULTS / "data_audit" / "cross_dataset_overlap.json").read_text())["unsw_nb15"]
    nd = rec["near_duplicates"]
    assert nd["s7_definition"] == near.S7_DEFINITION and nd["compared_with"] == "cicids2017_improved train + val"
    src, sm = datasets.load("cicids2017_improved", partitions=("train", "val"))
    tgt, tm = datasets.load("unsw_nb15", split="external", partitions=("external",))
    cs, ct = harmonise.to_canonical_unsw(src, "cic"), harmonise.to_canonical_unsw(tgt, "unsw")
    hs, ht = near._hashes(cs)[3], near._hashes(ct)[3]
    side = pd.read_parquet(near.sidecar_path("unsw_nb15", "cicids2017_improved")).set_index("_key")
    expect = np.isin(ht, np.unique(hs))
    assert (side.loc[tm["_key"], "seen_sig3"].to_numpy() == expect).all()
    tr = (sm["_partition"] == "train").to_numpy()
    assert nd["sig3"]["share_seen_in_source_train_only"] == pytest.approx(float(np.isin(ht, np.unique(hs[tr])).mean()))
    assert nd["sig3"]["share_seen_in_source_train_only"] <= 1 - nd["sig3"]["unseen_share"] + 1e-12


def test_unseen_target_row_matching_only_source_val_is_removed(drive, monkeypatch):
    from src.data import near
    cols = ["a", "b"]
    train = pd.DataFrame([[1.0, 10.0]], columns=cols, dtype="float32")
    val = pd.DataFrame([[2.0, 20.0]], columns=cols, dtype="float32")
    tgt = pd.DataFrame([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], columns=cols, dtype="float32")
    meta = pd.DataFrame({"_key": [0, 1, 2], "_y": [0, 1, 1], "_label": ["BENIGN", "DoS", "DoS"]})
    monkeypatch.setattr(near, "provenance", lambda d: {"stub": d})      # no reports in this tiny test
    out = near.external("unsw_nb15", pd.concat([train, val]), train, tgt, meta, np.array([0, 1, 2]))
    side = pd.read_parquet(near.sidecar_path("unsw_nb15", "cicids2017_improved"))
    assert side["seen_sig3"].tolist() == [True, True, False]           # val match removed too
    assert out["sig3"]["share_seen_in_source_train_only"] == pytest.approx(1 / 3)
    assert out["sig3"]["unseen_rows"] == 1 and not out["sig3"]["estimable"]


def test_verify_covers_s7_artifacts(built):
    from src.data.verify import verify_auxiliary
    rows = verify_auxiliary()
    assert len(rows) == 2 and all(r["ok"] for r in rows), rows


def test_changed_sidecar_is_refused_by_loader_and_verify(built):
    from src.data import datasets, near
    from src.data.verify import verify_auxiliary
    p = near.sidecar_path("unsw_nb15", "cicids2017_improved")
    side = pd.read_parquet(p)
    side.loc[side.index[0], "seen_sig3"] = not side.loc[side.index[0], "seen_sig3"]
    side.to_parquet(p, index=False)
    with pytest.raises(RuntimeError, match="SHA-256"):
        datasets.load("unsw_nb15", split="external", partitions=("external",), sensitivity=("S7_unseen_eval",))
    assert not [r for r in verify_auxiliary() if "unsw" in r["artifact"]][0]["ok"]


def test_edited_record_and_old_definition_are_refused(built):
    from src.data import datasets, near
    from src.data.verify import verify_auxiliary
    rp = near.record_path("cicids2017_improved")
    rec = json.loads(rp.read_text())
    rec["s7_definition"] = "1.2"
    rp.write_text(json.dumps(rec))
    near._checked.clear()
    with pytest.raises(RuntimeError, match="definition"):
        datasets.load("cicids2017_improved", partitions=("test",), sensitivity=("S7_unseen_eval",))
    row = [r for r in verify_auxiliary() if r["artifact"] == "S7 cicids2017_improved"][0]
    assert not row["ok"] and any("auxiliary_manifest" in p for p in row["problems"])


def test_rebuilt_dataset_invalidates_its_s7_record(built):
    from src.data import datasets, near
    rep = paths.RESULTS / "data_audit" / "cicids2017_improved" / "report.json"
    r = json.loads(rep.read_text())
    r["splits"]["primary"]["sha256"] = "0" * 64
    rep.write_text(json.dumps(r))
    near._checked.clear()
    with pytest.raises(RuntimeError, match="outputs changed"):
        datasets.load("cicids2017_improved", partitions=("test",), sensitivity=("S7_unseen_eval",))


def test_rebuilding_a_sidecar_reproduces_it(built):
    from src.data import near
    s = near.build("cicids2017_improved", log=lambda *a: None)
    assert s["reproduces_previous_sidecar"] is True

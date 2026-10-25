import pytest

from src.utils import paths


@pytest.fixture
def drive(tmp_path, monkeypatch):
    """A throwaway Drive tree, with results and data cards redirected away from the repo."""
    root = tmp_path / "drive"
    for name, sub in [("RAW", "data/raw"), ("INTERIM", "data/interim"), ("PROCESSED", "data/processed"),
                      ("SPLITS", "data/splits"), ("RESULTS", "results"), ("DATA_CARDS", "data_cards")]:
        d = root / sub
        d.mkdir(parents=True)
        monkeypatch.setattr(paths, name, d)
    monkeypatch.setenv("PHD_SCRATCH", str(tmp_path / "scratch"))
    return root

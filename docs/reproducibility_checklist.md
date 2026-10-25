# Reproducibility checklist

- [x] Every raw file has a SHA-256 in results/data_manifest/
- [x] Protocol frozen before confirmatory runs (docs/protocol/preregistration.md, v1.0, 2026-10-25)
- [x] Preprocessing driven by one versioned config whose SHA-256 is stored in every report
- [ ] Preprocessing reports and data cards generated for every processed dataset (notebooks/01_profile_and_preprocess.ipynb)
- [ ] Split files written and their SHA-256 recorded before any model is trained
- [ ] Shared CIC feature list frozen (config/feature_mapping/cic_shared_features.yaml)
- [ ] Every result records config, git commit, seed and package versions
- [ ] Figures and tables regenerate from results/ with scripts/make_figures.py and make_tables.py
- [ ] Code archived with a DOI (Zenodo) at submission

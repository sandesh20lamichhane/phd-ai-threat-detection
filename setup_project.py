"""Create the complete project structure: the code repo and the Google Drive data tree.

    python setup_project.py                      # repo skeleton + Drive tree (in Colab)
    python setup_project.py --repo-only          # repo skeleton only (e.g. on a laptop)
    python setup_project.py --drive-root /path   # Drive tree somewhere else
    python setup_project.py --tree               # print the planned structure, create nothing

Safe to run any number of times: it only creates what is missing and never overwrites a file.
Code and results live in the repo (GitHub + a clone in Drive). Data, models and run logs live
only in Drive. Dataset folders come from config/datasets.yaml.
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
DATASETS = [d["id"] for d in yaml.safe_load((REPO / "config" / "datasets.yaml").read_text())["datasets"]]

LEVELS = {
    "L1_within": "Level 1: within CICIDS2017, split by day + 5-fold CV",
    "L2_cross": "Level 2: train CICIDS2017, test CSE-CIC-IDS2018 (shared features) and UNSW-NB15 (mapped)",
    "L3_temporal": "Level 3: LANL, train first 40 days, test last 18 days, report lead time",
    "L4_domain": "Level 4: best model on CICIoMT2024, ToN_IoT, CIC-IoT2023 and sector-tagged advisories",
    "response": "Objective 2: RL response agent in CybORG (CAGE 2/4) and CyberBattleSim",
    "data_audit": "Data-quality audit: duplicates, identifiers, label corrections, class balance",
}
THREAT_INTEL = ["cert_in", "cisa", "nvd", "mitre_attack", "microsoft_security"]
SIMULATORS = ["cyborg_cage2", "cyborg_cage4", "cyberbattlesim"]


def py(doc):
    return f'"""{doc}"""\n'


def md(title, body=""):
    return f"# {title}\n\n{body.strip()}\n" if body else f"# {title}\n"


def tex_section(title, label, notes):
    return f"\\section{{{title}}}\\label{{sec:{label}}}\n% {notes}\n"


# ---------------------------------------------------------------- repo
def repo_files():
    f = {}
    # config
    f["config/paths.yaml"] = (
        "# Drive locations (mirrors src/utils/paths.py). Raw data never goes to GitHub.\n"
        "drive_root: /content/drive/MyDrive/PhD-AI-Cyber\n"
        "raw: data/raw\ninterim: data/interim\nprocessed: data/processed\nsplits: data/splits\n"
        "external: data/external\nmodels: models\nruns: runs\nsimulations: simulations\n")
    f["config/seeds.yaml"] = "# Every experiment runs with all of these seeds (protocol: at least 5)\nseeds: [11, 23, 37, 51, 73]\n"
    f["config/feature_mapping/cic_shared_features.yaml"] = (
        "# CICFlowMeter features shared by CICIDS2017 and CSE-CIC-IDS2018 (Level 2, same tool).\n"
        "# Fill from the column audit in notebooks/11_feature_harmonisation.ipynb; publish with the thesis.\n"
        "features: []\nlabel_map: {}\n")
    f["config/feature_mapping/cic_to_unsw_mapping.yaml"] = (
        "# Semantic mapping CICFlowMeter -> Argus/Bro features of UNSW-NB15 (Level 2, different tool).\n"
        "# Each entry: cic_name: {unsw: <name>, transform: identity|seconds_to_ms|..., note: ...}\n"
        "mapping: {}\nlabel_map: {}\n")
    f["config/feature_mapping/iot_shared_features.yaml"] = "# Shared features across CIC-IoT2023, CICIoMT2024 and ToN_IoT (Level 4)\nfeatures: []\nlabel_map: {}\n"
    exps = {
        "L1_within_cicids2017": ("cicids2017_improved", "by_day",
                                 "train_days: [Monday, Tuesday, Wednesday]\n  val_days: [Thursday]\n  test_days: [Friday]\n  cv_folds: 5"),
        "L2_cross_2017_to_2018": ("cicids2017_improved -> csecicids2018_improved", "cross_dataset",
                                  "features: config/feature_mapping/cic_shared_features.yaml"),
        "L2_cross_2017_to_unsw": ("cicids2017_improved -> unsw_nb15", "cross_dataset",
                                  "features: config/feature_mapping/cic_to_unsw_mapping.yaml"),
        "L3_temporal_lanl": ("lanl_cyber1", "by_time", "train_days: [1, 40]\n  test_days: [41, 58]\n  report: lead_time"),
        "L4_domain_transfer": ("best L1-L3 model -> ciciomt2024, ton_iot, ciciot2023", "cross_domain",
                               "features: config/feature_mapping/iot_shared_features.yaml"),
    }
    for name, (data, split, extra) in exps.items():
        f[f"config/experiments/{name}.yaml"] = (
            f"name: {name}\ndata: {data}\nsplit:\n  type: {split}\n  {extra}\n"
            "preprocessing: {drop_identifiers: true, drop_duplicates: true, balance: none}  # balance only on train folds\n"
            "models: [random_forest, xgboost, mlp, cnn_lstm, proposed]\n"
            "metrics: [macro_f1, per_class_recall, pr_auc, fpr_at_tpr_0.95]\n"
            "seeds: config/seeds.yaml\nstats: {two_models: wilcoxon, many_models: friedman_nemenyi}\n")
    for sim in SIMULATORS:
        f[f"config/experiments/response_{sim}.yaml"] = (
            f"name: response_{sim}\nsimulator: {sim}\nagents: [proposed_rl, rule_based_playbook, random]\n"
            "episodes: 1000\nmetrics: [mean_time_to_contain, containment_rate, mean_disruption_cost]\n"
            "seeds: config/seeds.yaml\n")

    # src
    stubs = {
        "src/data/harmonise.py": "Map each dataset onto the shared feature sets in config/feature_mapping/ for cross-dataset tests.",
        "src/data/load.py": "Readers for each raw dataset in Drive; return a DataFrame with a unified 'Label' column.",
        "src/features/build.py": "Feature engineering: scaling (fit on train only), encoding, windowed features for LANL.",
        "src/features/selection.py": "Feature selection fitted on training folds only (mutual information, SHAP-based pruning).",
        "src/models/baselines.py": "Classical baselines: logistic regression, random forest, XGBoost, LightGBM.",
        "src/models/deep.py": "Deep baselines: MLP, 1D-CNN, CNN-LSTM, Transformer encoder for flows.",
        "src/models/proposed.py": "The proposed predictive detection model (thesis contribution, Objective 1).",
        "src/models/predictive.py": "Sequence models over LANL host/user timelines that predict compromise before it happens.",
        "src/models/explain.py": "Explainability: SHAP values and per-attack feature attributions for analysts.",
        "src/response/envs.py": "Wrappers for CybORG (CAGE 2 and 4) and CyberBattleSim with one common interface.",
        "src/response/agents.py": "Reinforcement-learning response agents (PPO / DQN) driven by detector alerts.",
        "src/response/playbook.py": "Rule-based SOAR-style playbook and random agent used as response baselines.",
        "src/threat_intel/collect.py": "Collect advisories from CERT-In, CISA, NVD, MITRE ATT&CK and Microsoft Security Intelligence.",
        "src/threat_intel/tag_sector.py": "Tag each advisory by sector (government, healthcare, finance) for Objective 3.",
        "src/threat_intel/link_attack.py": "Link advisories and detections to MITRE ATT&CK techniques.",
        "src/evaluation/plots.py": "Publication figures: PR curves, confusion matrices, critical-difference diagrams, lead-time plots.",
        "src/evaluation/tables.py": "LaTeX tables (mean +- std over seeds) written to paper/tables/.",
        "src/utils/io.py": "Parquet/CSV helpers that log shapes and hashes of everything read or written.",
        "src/utils/logging.py": "Run logging: config, git commit, seed and package versions saved with every result.",
    }
    for p, doc in stubs.items():
        f[p] = py(doc)
    for pkg in ["src", "src/acquire", "src/data", "src/features", "src/models", "src/response",
                "src/threat_intel", "src/evaluation", "src/utils"]:
        f[f"{pkg}/__init__.py"] = "# package marker\n"

    # scripts
    f["scripts/run_experiment.py"] = py("Run one experiment config over all seeds: python scripts/run_experiment.py config/experiments/<name>.yaml")
    f["scripts/preprocess.py"] = py("Raw -> interim -> processed for one dataset, applying src/data/clean.py and saving the cleaning report.")
    f["scripts/make_tables.py"] = py("Collect results/*/metrics into the paper's LaTeX tables.")
    f["scripts/make_figures.py"] = py("Regenerate every paper figure from results/ (no figure is edited by hand).")
    f["scripts/collect_threat_intel.py"] = py("Download and tag the threat-intelligence corpus into Drive data/external/threat_intel/.")

    # notebooks (placeholders; the setup notebook is written separately)
    nbs = {
        "10_eda_overview": "Exploratory analysis of every dataset: sizes, class balance, feature types",
        "11_feature_harmonisation": "Build and justify the shared feature mappings for cross-dataset tests",
        "12_data_quality_audit": "Duplicates, identifier leakage, label corrections: numbers for the thesis",
        "20_level1_within_dataset": "Level 1 experiments and results",
        "21_level2_cross_dataset": "Level 2 experiments and results",
        "22_level3_temporal_lanl": "Level 3 experiments and lead-time analysis",
        "23_level4_domain_transfer": "Level 4 experiments across domains",
        "30_response_agent": "Objective 2: response agent training and evaluation",
        "40_explainability": "SHAP and attack-level explanations",
        "50_paper_figures": "Final figures and tables for the paper",
    }
    for name, title in nbs.items():
        f[f"notebooks/{name}.ipynb"] = (
            '{"cells":[{"cell_type":"markdown","metadata":{},"source":["# ' + name[:2] + " · " + title +
            '"]}],"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"}},"nbformat":4,"nbformat_minor":0}\n')

    # tests
    f["tests/__init__.py"] = "# package marker\n"

    # docs
    f["docs/protocol/experimental_protocol.md"] = md("Experimental protocol", """
Results are reported at four levels, each harder than the last. Only levels 2 to 4 support claims of real-world use.

| Level | Train on | Test on | Shows |
|---|---|---|---|
| 1 Within dataset | CICIDS2017 by day (70/15/15) + 5-fold CV | held-out days | the model can learn the attacks |
| 2 Across datasets | CICIDS2017 | CSE-CIC-IDS2018 (shared features); UNSW-NB15 (mapped) | works on unseen traffic |
| 3 Across time | LANL days 1-40 | LANL days 41-58 | flags compromise early (lead time) |
| 4 Across domains | best model of 1-3 | CICIoMT2024, ToN_IoT, CIC-IoT2023, advisories | holds in government, healthcare, finance |

**Metrics.** Macro-F1, per-class recall, PR-AUC, FPR at a fixed detection rate, detection lead time. Never accuracy alone.
**Response metrics.** Mean time to contain, containment rate, disruption cost; versus rule-based playbook and random agent.
**Statistics.** At least 5 seeds (config/seeds.yaml), mean and std, Wilcoxon signed-rank or Friedman + Nemenyi.
""")
    f["docs/protocol/data_quality_rules.md"] = md("Data-quality rules", """
Applied to every dataset before training (implemented in src/data/clean.py, checked by src/data/leakage.py).

- Use corrected labels for CICIDS2017 and CSE-CIC-IDS2018 (Engelen et al. 2021; Liu et al. 2022).
- Remove identifier columns: Flow ID, IPs, ports, timestamps.
- Remove exact duplicates before splitting.
- Split before balancing; SMOTE or undersampling on training folds only.
- Harmonise features with one documented mapping (config/feature_mapping/).
- Record why Bot-IoT was dropped (docs/decisions/ADR-0001-drop-bot-iot.md).
""")
    f["docs/protocol/hypotheses.md"] = md("Hypotheses (fixed before running experiments)", """
H1. The proposed detector reaches macro-F1 >= 0.95 at Level 1 and degrades by less than X points at Level 2.
H2. On LANL the predictive model raises an alert before the red-team event for at least Y% of compromises.
H3. The RL response agent contains attacks faster than the rule-based playbook at equal or lower disruption cost.
H4. Performance at Level 4 stays within Z points of Level 2 across healthcare, IoT and government settings.

Set X, Y, Z here and commit before the first Level 2 run; any later change goes in protocol_amendments.md.
""")
    f["docs/protocol/protocol_amendments.md"] = md("Protocol amendments", "Dated log of every change to the protocol after experiments started, with the reason.")
    f["docs/protocol/threats_to_validity.md"] = md("Threats to validity", "Internal, external, construct and conclusion validity; how each is mitigated.")
    f["docs/decisions/ADR-0001-drop-bot-iot.md"] = md("ADR-0001: Replace Bot-IoT with CIC-IoT2023", """
Status: accepted (2026-10-19). Over 99% of Bot-IoT records are attacks, which makes detection unrealistically easy.
CIC-IoT2023 has a more realistic benign/attack mix, 33 attacks and 105 devices.
""")
    f["docs/decisions/ADR-0002-corrected-cic-labels.md"] = md("ADR-0002: Use corrected CIC labels", "Status: accepted. Original CICIDS2017/CSE-CIC-IDS2018 files contain mislabelled and broken flows.")
    f["docs/reproducibility_checklist.md"] = md("Reproducibility checklist", """
- [ ] Every raw file has a SHA-256 in results/data_manifest/
- [ ] Every result records config, git commit, seed and package versions
- [ ] Splits saved as row indices in Drive data/splits/
- [ ] Figures and tables regenerate from results/ with scripts/make_figures.py and make_tables.py
- [ ] Code archived with a DOI (Zenodo) at submission
""")
    f["docs/ethics_and_data_use.md"] = md("Ethics and data use", "Licences and terms of each dataset, and the LANL and UNSW usage conditions.")
    f["docs/literature/README.md"] = md("Literature", "Reading notes and the systematic-review protocol. BibTeX lives in paper/references.bib.")
    cards = yaml.safe_load((REPO / "config" / "datasets.yaml").read_text())["datasets"]
    for d in cards:
        f[f"docs/data_cards/{d['id']}.md"] = md(d["name"], f"""
- **Role / objectives:** {d.get('objectives') or 'reference only'}
- **Source:** see config/datasets.yaml
- **Citation:** {' '.join(str(d.get('cite', 'TODO')).split())}
- **Files and SHA-256:** results/data_manifest/{d['id']}.json
- **Records, features, classes:** TODO after EDA
- **Class balance:** TODO
- **Known issues and what was done about them:** TODO
""")

    # results
    for lvl, desc in LEVELS.items():
        for sub in ("metrics", "tables", "figures"):
            f[f"results/{lvl}/{sub}/.gitkeep"] = ""
        f[f"results/{lvl}/README.md"] = md(lvl, desc)
    f["results/data_manifest/.gitkeep"] = ""

    # paper (Q1 journal layout)
    secs = [
        ("01_introduction", "Introduction", "intro", "problem, gap, contributions (numbered), paper outline"),
        ("02_related_work", "Related Work", "related", "IDS benchmarks and their flaws, predictive detection, automated response, cross-dataset studies"),
        ("03_threat_model", "Threat Model and Problem Formulation", "threat", "attacker capabilities, defender observations, formal task definition"),
        ("04_methodology", "Proposed Framework", "method", "detector, predictive model, response agent, threat-intelligence fusion"),
        ("05_datasets_protocol", "Datasets and Evaluation Protocol", "protocol", "8 datasets, data-quality rules, four-level protocol, metrics, statistics"),
        ("06_results", "Results", "results", "levels 1-4 and response results, mean +- std, significance tests"),
        ("07_discussion", "Discussion", "discussion", "what generalises, failure analysis, explainability findings, deployment cost"),
        ("08_threats_validity", "Threats to Validity and Limitations", "validity", "from docs/protocol/threats_to_validity.md"),
        ("09_conclusion", "Conclusion and Future Work", "conclusion", "answers to research questions"),
    ]
    inputs = "\n".join(f"\\input{{sections/{s}}}" for s, *_ in secs)
    f["paper/main.tex"] = (
        "\\documentclass[journal]{IEEEtran}  % swap for the target journal's class (e.g. elsarticle)\n"
        "\\usepackage{graphicx,booktabs,amsmath,amssymb,hyperref,cleveref,siunitx}\n"
        "\\begin{document}\n\\title{TODO: AI-Driven Predictive Threat Detection and Automated Response}\n"
        "\\author{Sandesh Lamichhane}\n\\maketitle\n\\input{sections/00_abstract}\n"
        f"{inputs}\n\\input{{sections/10_statements}}\n"
        "\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n\\end{document}\n")
    f["paper/sections/00_abstract.tex"] = "\\begin{abstract}\n% 200-250 words: problem, method, protocol, key numbers, significance\n\\end{abstract}\n\\begin{IEEEkeywords}\nintrusion detection, cross-dataset evaluation, reinforcement learning, automated response\n\\end{IEEEkeywords}\n"
    for s, title, label, notes in secs:
        f[f"paper/sections/{s}.tex"] = tex_section(title, label, notes)
    f["paper/sections/10_statements.tex"] = "\\section*{Data and Code Availability}\n% repo URL, Zenodo DOI, dataset sources\n\\section*{Declaration of Competing Interest}\n% none\n"
    f["paper/references.bib"] = "% BibTeX entries: dataset papers, Engelen et al. 2021, Liu et al. 2022, baselines\n"
    for sub in ("figures", "tables", "supplementary"):
        f[f"paper/{sub}/.gitkeep"] = ""
    f["paper/cover_letter.md"] = md("Cover letter", "Target journal, novelty, why it fits the journal's scope.")
    f["paper/response_to_reviewers.md"] = md("Response to reviewers", "Point-by-point responses, filled in at revision.")

    # thesis
    chapters = ["01_introduction", "02_literature_review", "03_research_methodology", "04_predictive_threat_detection",
                "05_automated_response", "06_domain_evaluation", "07_conclusion"]
    f["thesis/main.tex"] = ("\\documentclass[12pt,oneside]{report}\n\\usepackage{graphicx,booktabs,amsmath,hyperref}\n\\begin{document}\n"
                            + "\n".join(f"\\include{{chapters/{c}}}" for c in chapters)
                            + "\n\\bibliographystyle{plain}\n\\bibliography{../paper/references}\n\\appendix\n\\include{appendices/A_reproducibility}\n\\end{document}\n")
    for c in chapters:
        f[f"thesis/chapters/{c}.tex"] = f"\\chapter{{{c[3:].replace('_', ' ').title()}}}\n"
    f["thesis/appendices/A_reproducibility.tex"] = "\\chapter{Reproducibility}\n% dataset manifests, environment, seeds\n"
    f["thesis/progress_reports/README.md"] = md("Progress reports", "Half-yearly reports to the university.")
    f["thesis/figures/.gitkeep"] = ""
    return f


# ---------------------------------------------------------------- Drive
def drive_dirs():
    d = {
        "code": "Git clone of this repo (code + results). Synced to GitHub.",
        "config": "gated_urls.yaml: dataset links you got after registering. Never commit.",
        "data/raw": "Original files exactly as downloaded, with MANIFEST.json (SHA-256). Never edit.",
        "data/interim": "Cleaned: identifiers dropped, duplicates removed, inf -> NaN. One folder per dataset.",
        "data/processed": "Model-ready features (parquet), one folder per dataset and feature set.",
        "data/splits": "Saved train/val/test row indices per experiment and seed.",
        "data/external/threat_intel": "Advisories from CERT-In, CISA, NVD, MITRE ATT&CK, Microsoft Security Intelligence.",
        "models": "Trained models: models/<experiment>/<model>/seed_<n>/",
        "runs": "Training logs, TensorBoard/W&B files, per-run configs.",
        "simulations": "CybORG and CyberBattleSim episodes and agent checkpoints.",
        "backups": "Dated zips of results/ and paper/ before each submission.",
    }
    for ds in DATASETS:
        d[f"data/raw/{ds}"] = None
        d[f"data/interim/{ds}"] = None
        d[f"data/processed/{ds}"] = None
    for exp in ["L1_within_cicids2017", "L2_cross_2017_to_2018", "L2_cross_2017_to_unsw", "L3_temporal_lanl", "L4_domain_transfer"]:
        d[f"data/splits/{exp}"] = None
        d[f"models/{exp}"] = None
    for src in THREAT_INTEL:
        d[f"data/external/threat_intel/{src}"] = None
    for sim in SIMULATORS:
        d[f"simulations/{sim}"] = None
    return d


def create(base, files=None, dirs=None, dry=False):
    made = 0
    for rel, desc in (dirs or {}).items():
        p = base / rel
        if not dry:
            p.mkdir(parents=True, exist_ok=True)
            if desc and not (p / "README.md").exists():
                (p / "README.md").write_text(f"# {rel}\n\n{desc}\n")
                made += 1
    for rel, content in (files or {}).items():
        p = base / rel
        if dry or p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        made += 1
    return made


def tree(paths, title, all_dirs=False):
    """Print paths as an indented tree. Every prefix of a path is a folder."""
    nodes = {}
    for p in paths:
        parts = p.split("/")
        for i in range(1, len(parts)):
            nodes["/".join(parts[:i])] = True     # folder
        nodes.setdefault(p, all_dirs)
    print(f"\n{title}")
    for p in sorted(nodes):
        if p.endswith(".gitkeep"):
            continue
        name = p.rsplit("/", 1)[-1]
        print("  " + "    " * p.count("/") + name + ("/" if nodes[p] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/PhD-AI-Cyber")
    ap.add_argument("--repo-only", action="store_true")
    ap.add_argument("--tree", action="store_true", help="print the structure and exit")
    a = ap.parse_args(argv)

    files, dirs = repo_files(), drive_dirs()
    if a.tree:
        existing = {str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file()
                    and not {".git", "__pycache__", ".pytest_cache"} & set(p.parts)}
        tree(set(files) | existing, "Repo (phd-ai-threat-detection)")
        tree(dirs, f"Drive ({a.drive_root})", all_dirs=True)
        return 0
    n = create(REPO, files=files)
    print(f"Repo: {n} new files ({len(files)} in the plan) at {REPO}")
    if not a.repo_only:
        root = Path(a.drive_root)
        if not root.parent.exists():
            sys.exit(f"{root.parent} not found: mount Google Drive first, or pass --drive-root / --repo-only")
        n = create(root, dirs=dirs)
        print(f"Drive: {len(dirs)} folders ensured, {n} READMEs written at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Confirmatory analysis of Levels 1-3 from the scoring units (preregistration §6, §6b, §7, §8.1; hypotheses.md).

Reads only results/confirmatory/ (written by src/models/score.py), never the data, and writes:

    results/confirmatory/level1_2.json   H1a-H1c, S7, every candidate's figures, the secondary model comparisons
    results/confirmatory/level3.json     H2, its baselines and the boundary sensitivity, every candidate
    results/confirmatory/decisions.json  both of the above and the decisions of the confirmatory family
    results/confirmatory/decisions.md    the same as tables, for the paper and the thesis

Decisions (§6, amendment 1.6). Each claim has two one-sided bootstrap tests at alpha 0.025, "support" (e.g.
F1 >= 0.95) and "against" (F1 < 0.95). Holm's step-down runs over the family {H1a, H1b, H1c, H2, H4 x 3}
separately for each direction. A claim is supported when its support test is rejected after Holm, not supported
when its against test is rejected, and inconclusive otherwise. H4 has not been run: its p-values enter as 1.
Holm-adjusted p-values never increase when another p-value falls, so a supported or not-supported decision made
now stands once H4 exists; only an inconclusive one can still change. The unadjusted 95 % CI outcome of
hypotheses.md is shown beside each decision.

The analysis runs once. Running it again recomputes everything and stops if any number differs from the
decisions already written.
"""
import json

import numpy as np
import pandas as pd

from ..models import develop as D
from ..models import score as S
from ..models import seal
from ..utils.io import sha256_file, write_json
from ..utils.logging import run_info
from . import stats

FAMILY = ("H1a", "H1b", "H1c", "H2", "H4-healthcare", "H4-IoT", "H4-government")
ALPHA = 0.025                     # per direction (§6)
H1A_MIN, X, Y, SEVERE = 0.95, 0.05, 0.50, 0.10       # hypotheses.md, frozen 2026-10-25
COMPARISON_ALPHA = 0.05           # secondary model comparisons, Holm within each table (§6)


# ---- Levels 1-2: counts -------------------------------------------------------------------------------------
def load_groups(eval_id):
    gp = S.groups_path(eval_id)
    return pd.read_parquet(gp), sha256_file(gp)


def load_alerts(eval_id, space, name, gsha, fr, fr_sha):
    """(n_seeds, n_groups) alerted rows per group, after checking every unit against the groups and the freeze."""
    rows, units = [], []
    for s in D.seeds():
        up = S.unit_path(eval_id, space, name, s)
        if not up.exists():
            raise FileNotFoundError(f"unit {eval_id}/{space}/{name}/seed{s} not scored; run scripts/score.py --level12")
        u = json.loads(up.read_text())
        if u["groups_sha256"] != gsha or u["freeze_sha256"] != fr_sha:
            raise AssertionError(f"{up} was scored against other groups or another freeze record")
        if fr["model_files_sha256"].get(u["model_file"]) != u["model_sha256"]:
            raise AssertionError(f"{up}: model file not the frozen one")
        rows.append(np.asarray(u["alerts"], dtype=np.int64))
        units.append(u)
    return np.stack(rows), units


def block_counts(g, alerts, mask):
    """(n_seeds, n_blocks, 2, 3): per 300 s block and class (0 benign, 1 attack), TP, FP and FN."""
    gm, a = g[mask], alerts[:, mask]
    n, att = gm["n"].to_numpy(np.int64), gm["y"].to_numpy() == 1
    _, b = np.unique(gm["block"].to_numpy(), return_inverse=True)
    nb = int(b.max()) + 1 if len(b) else 0
    out = np.zeros((len(a), nb, 2, 3), np.int64)
    for k, ak in enumerate(a):
        tp = np.bincount(b, ak * att, nb)
        fn = np.bincount(b, (n - ak) * att, nb)
        fp = np.bincount(b, ak * ~att, nb)
        tn = np.bincount(b, (n - ak) * ~att, nb)
        out[k, :, 1] = np.stack([tp, fp, fn], axis=1)
        out[k, :, 0] = np.stack([tn, fn, fp], axis=1)
    return out


def metrics(c):
    """Per-seed figures from counts (n_seeds, n_blocks, 2, 3), and their seed means."""
    t = c.sum(axis=1).astype(float)                                     # (n_seeds, 2, 3)
    tp, fp, fn = t[..., 0], t[..., 1], t[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        fpr = np.where(t[:, 0, 0] + t[:, 1, 1] > 0, t[:, 1, 1] / (t[:, 0, 0] + t[:, 1, 1]), 0.0)
    per_seed = {"macro_f1": f1.mean(axis=1), "balanced_accuracy": rec.mean(axis=1), "fpr": fpr,
                "benign_precision": prec[:, 0], "benign_recall": rec[:, 0], "benign_f1": f1[:, 0],
                "attack_precision": prec[:, 1], "attack_recall": rec[:, 1], "attack_f1": f1[:, 1]}
    out = {k: float(v.mean()) for k, v in per_seed.items()}
    out["macro_f1_sd"] = float(per_seed["macro_f1"].std(ddof=1)) if len(c) > 1 else 0.0
    out["macro_f1_per_seed"] = [float(v) for v in per_seed["macro_f1"]]
    return out


def composition(g, mask):
    """Rows and blocks per class, the estimability rule of §8.1 and rows per family, for the rows in mask."""
    gm = g[mask]
    out, est = {"rows": int(gm["n"].sum())}, True
    for cls, name in ((0, "benign"), (1, "attack")):
        k = gm[gm["y"] == cls]
        out[f"{name}_rows"], out[f"{name}_blocks"] = int(k["n"].sum()), int(k["block"].nunique())
        est &= out[f"{name}_rows"] >= 100 and out[f"{name}_blocks"] >= 30    # src/data/near.py, fixed (§8.1)
    out["estimable"] = bool(est)
    out["attack_share"] = out["attack_rows"] / out["rows"] if out["rows"] else None
    out["rows_by_family"] = {str(f): int(v) for f, v in gm.groupby("label")["n"].sum().items()}
    return out


def family_recall(g, alerts, mask):
    gm, a = g[mask], alerts[:, mask]
    out = {}
    for f, idx in gm.groupby("label").indices.items():
        n = gm["n"].to_numpy()[idx].sum()
        out[str(f)] = float(a[:, idx].sum(axis=1).mean() / n) if n else None     # share alerted, seed mean
    return out


def claim(bs, margin, direction):
    """Point estimate, CI, both one-sided p-values and the unadjusted three-outcome verdict of one claim.

    p_support tests the claim (e.g. F1 >= 0.95): share of replicates on its wrong side. p_against tests the
    opposite (F1 < 0.95). Both are the (k + 1) / (B + 1) form of stats.one_sided_p."""
    other = "below" if direction == "at_least" else "at_least"
    reps = bs["replicates"]
    return {"estimate": bs["estimate"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "B": bs["B"],
            "bootstrap_seed": bs["seed"], "margin": margin, "claim": direction,
            "p_support": stats.one_sided_p(reps, margin, direction), "p_against": stats.one_sided_p(reps, margin, other),
            "unadjusted_outcome": stats.three_outcome(bs["ci_low"], bs["ci_high"], margin, direction)}


# ---- Levels 1-2: the analysis -------------------------------------------------------------------------------
H1_SIDES = {"H1a": (("cicids2017_test", "cic"), None),
            "H1b": (("cicids2017_test", "cic"), ("csecicids2018", "cic")),
            "H1c": (("cicids2017_test", "unsw_canonical"), ("unsw_nb15", "unsw_canonical"))}
SUBSETS = {"primary": None, "S7_sig3": "seen_sig3", "S7_sig2": "seen_sig2"}


def level12(log=print):
    fr = seal.require_frozen(track="l12")
    fr_sha = sha256_file(seal.freeze_path("l12"))
    cands = S.frozen_candidates(fr)
    prop = fr["proposed"]
    G, A, U = {}, {}, {}
    for e, es in S.EVAL_SETS.items():
        G[e] = load_groups(e)
        for sp in es["spaces"]:
            for c in cands:
                A[e, sp, c], U[e, sp, c] = load_alerts(e, sp, c, G[e][1], fr, fr_sha)

    def counts(e, sp, c, subset="primary"):
        g = G[e][0]
        col = SUBSETS[subset]
        mask = np.ones(len(g), bool) if col is None else ~g[col].to_numpy()
        return block_counts(g, A[e, sp, c], mask), composition(g, mask)

    res = {"proposed": prop, "freeze_sha256": fr_sha, "claims": {}, "s7": {}, "candidates": {}, "comparisons": {}}
    for h, (src, tgt) in H1_SIDES.items():
        cs, comp_s = counts(*src, prop)
        if tgt is None:
            bs = stats.block_bootstrap(cs)
            entry = claim(bs, H1A_MIN, "at_least")
            entry.update({"source": "/".join(src), "figures": metrics(cs), "rows": comp_s})
        else:
            ct, comp_t = counts(*tgt, prop)
            bs = stats.block_bootstrap(ct, source=cs)
            entry = claim(bs, X, "below")
            entry.update({"source": "/".join(src), "target": "/".join(tgt), "source_figures": metrics(cs),
                          "target_figures": metrics(ct), "source_rows": comp_s, "target_rows": comp_t,
                          "severe_band": {"margin": SEVERE, "share_of_replicates_above": float(np.mean(bs["replicates"] > SEVERE)),
                                          "ci_entirely_above": bool(bs["ci_low"] > SEVERE)}})
        res["claims"][h] = entry
        log(f"{h}: {entry['estimate']:.6f} [{entry['ci_low']:.6f}, {entry['ci_high']:.6f}] "
            f"(unadjusted: {entry['unadjusted_outcome']})")
        res["s7"][h] = {}
        for sub in ("S7_sig3", "S7_sig2"):
            cs7, cps7 = counts(*src, prop, sub)
            r = {"source_rows": cps7, "source_share_removed": 1 - cps7["rows"] / comp_s["rows"]}
            if tgt is None:
                if cps7["estimable"]:
                    r.update(claim(stats.block_bootstrap(cs7), H1A_MIN, "at_least"))
                else:
                    r["status"] = "not estimable"
            else:
                ct7, cpt7 = counts(*tgt, prop, sub)
                r.update({"target_rows": cpt7, "target_share_removed": 1 - cpt7["rows"] / res["claims"][h]["target_rows"]["rows"]})
                if cps7["estimable"] and cpt7["estimable"]:
                    r.update(claim(stats.block_bootstrap(ct7, source=cs7), X, "below"))
                else:
                    r["status"] = "not estimable (both sides needed, §8.1)"
                    if cpt7["estimable"]:                  # descriptive target side only (amendment 1.3)
                        bt = stats.block_bootstrap(ct7)
                        r["target_side_macro_f1"] = {"estimate": bt["estimate"], "ci_low": bt["ci_low"],
                                                     "ci_high": bt["ci_high"]}
            res["s7"][h][sub] = r
    # every candidate on every evaluation set (baselines and the proposed detector)
    for (e, sp, c), al in A.items():
        cc, cp = counts(e, sp, c)
        m = metrics(cc)
        us = U[e, sp, c]
        m["pr_auc"] = float(np.mean([u["pr_auc"] for u in us])) if all(u["pr_auc"] is not None for u in us) else None
        m["fpr_at_tpr_0.95"] = float(np.mean([u["fpr_at_tpr_0.95"] for u in us])) \
            if all(u["fpr_at_tpr_0.95"] is not None for u in us) else None
        m["attack_family_recall"] = family_recall(G[e][0], al, np.ones(len(G[e][0]), bool))
        m["gpu_check"] = [u["gpu_check"] for u in us if u.get("gpu_check")]
        res["candidates"][f"{e}/{sp}/{c}"] = m
    # secondary (§6): proposed vs each baseline, one table per feature space (the models differ between spaces);
    # pairs = seeds x evaluation sets of that space; Holm over the baselines within the table
    for sp in ("cic", "unsw_canonical"):
        cells = [e for e, es in S.EVAL_SETS.items() if sp in es["spaces"]]
        tab = pd.DataFrame({c: np.concatenate([res["candidates"][f"{e}/{sp}/{c}"]["macro_f1_per_seed"] for e in cells])
                            for c in cands}, index=[f"{e}/seed{s}" for e in cells for s in D.seeds()])
        w = {c: _wilcoxon(tab[prop].to_numpy(), tab[c].to_numpy()) for c in cands if c != prop}
        adj = stats.holm({c: v["p_value"] for c, v in w.items()}, alpha=COMPARISON_ALPHA)
        res["comparisons"][sp] = {
            "pairs": len(tab), "evaluation_sets": cells, "alpha": COMPARISON_ALPHA,
            "macro_f1": {c: tab[c].tolist() for c in tab.columns},
            "wilcoxon_vs_proposed": {c: {**w[c], "holm_p": adj[c][0], "rejected": adj[c][1],
                                         "median_difference": float(np.median(tab[prop] - tab[c]))} for c in w},
            "friedman_nemenyi": stats.friedman_nemenyi(tab, alpha=COMPARISON_ALPHA)}
    return res


def _wilcoxon(a, b):
    if np.all(a == b):                       # no difference at all: scipy has no statistic for this
        return {"statistic": 0.0, "p_value": 1.0}
    return stats.wilcoxon(a, b)


# ---- Level 3 --------------------------------------------------------------------------------------------------
def _account_bootstrap(alerted, accounts, B=stats.BOOTSTRAP_B, seed=stats.BOOTSTRAP_SEED, alpha=0.05):
    """alerted: (n_hosts, n_seeds) 0/1. Red-team accounts are resampled with replacement (§7); each replicate is
    the seed mean of (alerted hosts / hosts) over the hosts of the drawn accounts."""
    _, ai = np.unique(np.asarray(accounts), return_inverse=True)
    na = int(ai.max()) + 1
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(B)]).astype(float)[:, ai]
    reps = ((W @ alerted) / W.sum(axis=1, keepdims=True)).mean(axis=1)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(alerted.mean(axis=0).mean()), "ci_low": float(lo), "ci_high": float(hi), "B": B,
            "seed": seed, "accounts": na, "replicates": reps}


def level3(log=print):
    fr = seal.require_frozen(track="lanl")
    fr_sha = sha256_file(seal.freeze_path("lanl"))
    prop = fr["proposed"]
    units = {}
    for c in S.frozen_candidates(fr):
        us = []
        for s in D.seeds():
            up = S.lanl_unit_path(c, s)
            if not up.exists():
                raise FileNotFoundError(f"unit lanl/{c}/seed{s} not scored; run scripts/score.py --lanl")
            u = json.loads(up.read_text())
            if u["freeze_sha256"] != fr_sha or fr["model_files_sha256"].get(u["model_file"]) != u["model_sha256"]:
                raise AssertionError(f"{up} was not scored with the frozen Level 3 model")
            us.append(u)
        units[c] = us
    rb = json.loads((S.out_dir() / "level3" / "random_baseline.json").read_text())
    if rb["freeze_sha256"] != fr_sha:
        raise AssertionError("random baseline computed under another Level 3 freeze")
    res = {"proposed": prop, "rules_version": fr.get("rules_version"), "freeze_sha256": fr_sha,
           "random_baseline": rb, "candidates": {}}
    for c, us in units.items():
        hosts = [h["host"] for h in us[0]["per_host"]]
        if any([h["host"] for h in u["per_host"]] != hosts for u in us):
            raise AssertionError("test hosts differ between seeds")
        acc = [h["account"] for h in us[0]["per_host"]]
        al = np.array([[h["lead_hours"] is not None for h in u["per_host"]] for u in us], float).T
        al_t = np.array([[h["lead_hours_test_days_only"] is not None for h in u["per_host"]] for u in us], float).T
        react = np.array([[h["reactive_alert"] for h in u["per_host"]] for u in us], float).T
        leads = [h["lead_hours"] for u in us for h in u["per_host"] if h["lead_hours"] is not None]
        bs = _account_bootstrap(al, acc)
        m = {"early_alert_rate": bs["estimate"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
             "accounts": bs["accounts"], "hosts": len(hosts),
             "hosts_alerted_per_seed": [u["hosts_alerted"] for u in us],
             "fa_rate_per_seed": [u["fa_rate"] for u in us], "fa_rate": float(np.mean([u["fa_rate"] for u in us])),
             "eligible_host_days": us[0]["eligible_host_days"],
             "median_lead_hours": float(np.median(leads)) if leads else None,
             "pr_auc": float(np.mean([u["pr_auc"] for u in us])) if all(u["pr_auc"] is not None for u in us) else None,
             "reactive_detection_rate": float(react.mean(axis=0).mean()),
             "hosts_without_compromise_hour_row": int(sum(h["compromise_hour_score"] is None for h in us[0]["per_host"])),
             "hosts_with_window_on_previous_day": int(sum(h["window_on_previous_day"] for h in us[0]["per_host"]))}
        bt = _account_bootstrap(al_t, acc)
        m["boundary_sensitivity_test_days_only"] = {"early_alert_rate": bt["estimate"], "ci_low": bt["ci_low"],
                                                    "ci_high": bt["ci_high"]}
        if c == prop:
            entry = claim(bs, Y, "at_least")
            entry["above_random_baseline"] = bool(bs["ci_low"] > rb["early_alert_rate_q975"])
            if entry["unadjusted_outcome"] == "supported" and not entry["above_random_baseline"]:
                entry["unadjusted_outcome"] = "inconclusive"           # hypotheses.md: and above the random baseline
            entry["realised_fa_rate"] = m["fa_rate"]                   # reported beside H2 (§7), not a second test
            entry["figures"] = m
            res["claim"] = entry
            log(f"H2: {entry['estimate']:.6f} [{entry['ci_low']:.6f}, {entry['ci_high']:.6f}] "
                f"(unadjusted: {entry['unadjusted_outcome']}); realised false alarms {m['fa_rate']:.4%}")
        res["candidates"][c] = m
    return res


# ---- decisions ------------------------------------------------------------------------------------------------
def decide(l12, l3):
    claims = {h: l12["claims"][h] for h in ("H1a", "H1b", "H1c")}
    claims["H2"] = l3["claim"]
    pending = [h for h in FAMILY if h not in claims]
    adj_s = stats.holm({h: claims[h]["p_support"] if h in claims else 1.0 for h in FAMILY}, alpha=ALPHA)
    adj_a = stats.holm({h: claims[h]["p_against"] if h in claims else 1.0 for h in FAMILY}, alpha=ALPHA)
    out = {}
    for h in FAMILY:
        if h in pending:
            out[h] = {"decision": "pending (Level 4 design open, preregistration §9)", "p_support": None,
                      "p_against": None}
            continue
        c = claims[h]
        sup, ag = adj_s[h][1], adj_a[h][1]
        if h == "H2":
            sup = sup and c["above_random_baseline"]
        decision = "supported" if sup else "not supported" if ag else "inconclusive"
        out[h] = {"estimate": c["estimate"], "ci_low": c["ci_low"], "ci_high": c["ci_high"], "margin": c["margin"],
                  "claim": c["claim"], "p_support": c["p_support"], "p_against": c["p_against"],
                  "holm_p_support": adj_s[h][0], "holm_p_against": adj_a[h][0],
                  "unadjusted_outcome": c["unadjusted_outcome"], "decision": decision,
                  "final": decision != "inconclusive" or not pending}
    return {"family": list(FAMILY), "alpha_per_direction": ALPHA, "pending": pending, "claims": out,
            "note": "H4 p-values enter Holm as 1 until Level 4 is run; supported / not supported decisions are "
                    "final, inconclusive ones may change once H4 exists (amendment 1.6)."}


def _strip(o):
    """Everything that must reproduce exactly on a second run (no replicates, timestamps or run details)."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ("replicates", "run", "generated_utc")}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return _strip(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def analyse(log=print):
    l12, l3 = level12(log), level3(log)
    dec = decide(l12, l3)
    new = {"level1_2": _strip(l12), "level3": _strip(l3), "decisions": _strip(dec)}
    fp = S.out_dir() / "decisions.json"
    if fp.exists():
        old = json.loads(fp.read_text())
        old = {k: old[k] for k in new}
        if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
            raise AssertionError("the confirmatory analysis already ran and a second run gives different numbers; "
                                 "nothing was overwritten. Send this message to the project thread.")
        log("confirmatory analysis reproduced exactly; nothing rewritten")
        return json.loads(fp.read_text())
    write_json(S.out_dir() / "level1_2.json", new["level1_2"])
    write_json(S.out_dir() / "level3.json", new["level3"])
    out = {**new, "generated_utc": S._now(), "run": run_info()}
    write_json(fp, out)
    (S.out_dir() / "decisions.md").write_text(render(out))
    log(f"decisions written to {fp}")
    return out


def _f(v, d=6):
    return "n/a" if v is None else f"{v:.{d}f}"


def render(out):
    dec, l12, l3 = out["decisions"], out["level1_2"], out["level3"]
    lines = ["# Confirmatory results, Levels 1-3", "",
             f"_Generated by `scripts/score.py --analyse` on {out['generated_utc']} from the frozen models "
             f"(proposed detector `{l12['proposed']}`, Level 3 model `{l3['proposed']}`). Holm over "
             f"{', '.join(dec['family'])} at {dec['alpha_per_direction']} per direction; {dec['note']}_", "",
             "| claim | estimate | 95 % CI | margin | p support | Holm p support | p against | Holm p against | "
             "unadjusted CI outcome | decision |", "|---|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for h in dec["family"]:
        c = dec["claims"][h]
        if c.get("estimate") is None:
            lines.append(f"| {h} | | | | | | | | | {c['decision']} |")
            continue
        lines.append(f"| {h} | {_f(c['estimate'])} | [{_f(c['ci_low'])}, {_f(c['ci_high'])}] | {c['margin']} | "
                     f"{_f(c['p_support'], 4)} | {_f(c['holm_p_support'], 4)} | {_f(c['p_against'], 4)} | "
                     f"{_f(c['holm_p_against'], 4)} | {c['unadjusted_outcome']} | **{c['decision']}** |")
    lines += ["", "## Beside every claim", ""]
    for h in ("H1a", "H1b", "H1c"):
        c = l12["claims"][h]
        if h == "H1a":
            f = c["figures"]
            lines.append(f"- **H1a** ({c['source']}): macro-F1 {_f(f['macro_f1'])} (sd over seeds {_f(f['macro_f1_sd'])}), "
                         f"balanced accuracy {_f(f['balanced_accuracy'])}, benign recall {_f(f['benign_recall'])}, "
                         f"attack recall {_f(f['attack_recall'])}, FPR {_f(f['fpr'])}; {c['rows']['benign_blocks']} benign "
                         f"and {c['rows']['attack_blocks']} attack blocks.")
        else:
            s, t = c["source_figures"], c["target_figures"]
            lines.append(f"- **{h}** ({c['source']} -> {c['target']}): macro-F1 {_f(s['macro_f1'])} -> {_f(t['macro_f1'])}; "
                         f"balanced accuracy {_f(s['balanced_accuracy'])} -> {_f(t['balanced_accuracy'])}; attack recall "
                         f"{_f(s['attack_recall'])} -> {_f(t['attack_recall'])}; benign recall {_f(s['benign_recall'])} -> "
                         f"{_f(t['benign_recall'])}; attack share {_f(c['source_rows']['attack_share'], 4)} -> {_f(c['target_rows']['attack_share'], 4)}; "
                         f"share of replicates with a drop above {c['severe_band']['margin']}: "
                         f"{_f(c['severe_band']['share_of_replicates_above'], 4)}.")
    f = l3["claim"]["figures"]
    rb = l3["random_baseline"]
    lines.append(f"- **H2**: {sum(f['hosts_alerted_per_seed']) / len(f['hosts_alerted_per_seed']):.1f} of {f['hosts']} test "
                 f"hosts alerted early on average ({f['accounts']} red-team accounts); realised false alarms "
                 f"{f['fa_rate']:.4%} of {f['eligible_host_days']:,} host-days; median lead {_f(f['median_lead_hours'], 1)} h; "
                 f"random alerting at the same budget {_f(rb['expected_early_alert_rate'], 4)} (95 % range "
                 f"{_f(rb['early_alert_rate_q025'], 4)}-{_f(rb['early_alert_rate_q975'], 4)}), above it: "
                 f"{l3['claim']['above_random_baseline']}; reactive detection at the compromise hour "
                 f"{_f(f['reactive_detection_rate'], 4)}; alerts on test days only {_f(f['boundary_sensitivity_test_days_only']['early_alert_rate'], 4)} "
                 f"({f['hosts_with_window_on_previous_day']} hosts have part of their window on the day before the test).")
    lines += ["", "## S7 (outside Holm, §8.1)", "", "| claim | subset | status / estimate | 95 % CI | rows removed (source, target) |",
              "|---|---|---|---|---|"]
    for h, subs in l12["s7"].items():
        for sub, r in subs.items():
            removed = f"{_f(r['source_share_removed'], 4)}" + (f", {_f(r['target_share_removed'], 4)}" if "target_share_removed" in r else "")
            if "estimate" in r:
                lines.append(f"| {h} | {sub} | {_f(r['estimate'])} ({r['unadjusted_outcome']}) | "
                             f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {removed} |")
            else:
                ts = r.get("target_side_macro_f1")
                extra = f"; target-side macro-F1 {_f(ts['estimate'])} [{_f(ts['ci_low'])}, {_f(ts['ci_high'])}]" if ts else ""
                lines.append(f"| {h} | {sub} | {r['status']}{extra} | | {removed} |")
    lines += ["", "## Every candidate (seed means)", "", "| evaluation / space / candidate | macro-F1 | sd | balanced acc. | "
              "attack recall | benign recall | FPR | PR-AUC | FPR at 95 % detection |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, m in l12["candidates"].items():
        lines.append(f"| {k} | {_f(m['macro_f1'])} | {_f(m['macro_f1_sd'])} | {_f(m['balanced_accuracy'])} | "
                     f"{_f(m['attack_recall'])} | {_f(m['benign_recall'])} | {_f(m['fpr'])} | {_f(m['pr_auc'])} | "
                     f"{_f(m['fpr_at_tpr_0.95'])} |")
    lines += ["", "Secondary model comparisons (§6), proposed detector vs each baseline, Wilcoxon signed-rank over seeds x "
              "evaluation sets, Holm within each table:", ""]
    for sp, cmp_ in l12["comparisons"].items():
        fn = cmp_["friedman_nemenyi"]
        lines.append(f"- `{sp}` ({cmp_['pairs']} pairs: {', '.join(cmp_['evaluation_sets'])}): " + "; ".join(
            f"{c} median difference {_f(v['median_difference'])}, Holm p {_f(v['holm_p'], 4)}"
            for c, v in cmp_["wilcoxon_vs_proposed"].items()) + f". Friedman p {_f(fn['p_value'], 4)}, Nemenyi critical "
            f"difference {_f(fn['critical_difference'], 3)}, average ranks " + ", ".join(
            f"{c} {r:.2f}" for c, r in fn["avg_rank"].items()) + ".")
    lines += ["",
              "## Level 3, every candidate", "", "| candidate | early-alert rate | 95 % CI (accounts) | false alarms | "
              "median lead (h) | PR-AUC | reactive | test days only |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for c, m in l3["candidates"].items():
        lines.append(f"| {c} | {_f(m['early_alert_rate'])} | [{_f(m['ci_low'])}, {_f(m['ci_high'])}] | {m['fa_rate']:.4%} | "
                     f"{_f(m['median_lead_hours'], 1)} | {_f(m['pr_auc'])} | {_f(m['reactive_detection_rate'], 4)} | "
                     f"{_f(m['boundary_sensitivity_test_days_only']['early_alert_rate'], 4)} |")
    return "\n".join(lines) + "\n"

"""Metrics for the automated-response module (Objective 2).

Each episode is a dict with: attack_start, contained_at (None if never contained), and
disruption, the cost of defensive actions to normal service (simulator units).
Compare every agent against a rule-based playbook and a random agent.
"""
import numpy as np


def response_summary(episodes):
    contained = [e for e in episodes if e.get("contained_at") is not None]
    ttc = [e["contained_at"] - e["attack_start"] for e in contained]
    return {
        "episodes": len(episodes),
        "containment_rate": len(contained) / len(episodes) if episodes else float("nan"),
        "mean_time_to_contain": float(np.mean(ttc)) if ttc else float("nan"),
        "median_time_to_contain": float(np.median(ttc)) if ttc else float("nan"),
        "mean_disruption_cost": float(np.mean([e.get("disruption", 0.0) for e in episodes])) if episodes else float("nan"),
    }

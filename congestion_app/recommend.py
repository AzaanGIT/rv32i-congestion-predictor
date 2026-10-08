"""
recommend.py
-------------
Given a starting set of the 5 design knobs (and the per-layer resource
values the user supplied), searches nearby points in parameter space
(within the ranges seen in the dataset) to find the SMALLEST exact
change to the 5 knobs that brings predicted congestion under a target
threshold while keeping the routed/success prediction positive.

Resource values are NOT part of the search space -- in a real flow
they're a fixed property of the metal stack, not something you'd
change to fix congestion. The search only varies utilization, aspect
ratio, core margin, density, and layer_adj; usage_pct is then derived
from the model's demand prediction and the given resource:
    usage_pct = demand / resource * 100

The search is a coordinate grid search over normalized step sizes per
parameter (fine enough to give concrete, actionable numbers), scored
by a weighted distance so it prefers changing the fewest / smallest
knobs first.
"""
import itertools
import numpy as np
import pandas as pd

# Precision each parameter is reported/rounded to (mirrors the dataset).
ROUND = {
    "utilization": 1,
    "aspect_ratio": 2,
    "core_margin": 2,
    "density": 2,
    "layer_adj": 2,
}

# Step resolution used while searching each parameter's range.
STEPS = 9  # candidate points per parameter (incl. the current value)
MAX_RELATIVE_STEP = 0.35  # search up to +/-35% of the param's full range


def _candidate_grid(current, bounds, features, relative_step=MAX_RELATIVE_STEP):
    """Build the candidate values to try for each feature."""
    grids = {}
    for f in features:
        lo, hi = bounds[f]["min"], bounds[f]["max"]
        span = hi - lo
        window = span * relative_step
        f_lo = max(lo, current[f] - window)
        f_hi = min(hi, current[f] + window)
        vals = np.linspace(f_lo, f_hi, STEPS)
        vals = np.round(vals, ROUND[f])
        vals = sorted(set(vals.tolist() + [round(current[f], ROUND[f])]))
        grids[f] = vals
    return grids


def _normalized_distance(current, candidate, bounds, features):
    dist = 0.0
    for f in features:
        span = bounds[f]["max"] - bounds[f]["min"]
        if span == 0:
            continue
        dist += ((candidate[f] - current[f]) / span) ** 2
    return dist ** 0.5


def usage_pct_from_demand(demand, resource):
    if not resource or resource <= 0:
        return 0.0
    return demand / resource * 100.0


def recommend_fix(current, target_pct, clf, regressors, meta, resource,
                   max_candidates=20000, relative_step=MAX_RELATIVE_STEP, target_layer=None):
    """
    current: dict of the 5 feature values
    target_pct: desired max usage % for the target (a congestion %)
    resource: dict of per-layer resource values, e.g. {"metal2": 27540, ...}
        plus "Total" (sum of all layers). Used to convert the model's
        demand prediction into usage_pct.
    target_layer: which layer to bring under target_pct, e.g. "metal3".
        None (default) targets the overall Total usage_pct.
    relative_step: how far (as a fraction of each parameter's full
        dataset range) the search is allowed to move away from the
        current value. Pass a larger value (e.g. 1.0) for a wider,
        "search the whole space" retry when the narrow search fails.
    Returns: dict describing best achievable fix (or None if nothing in
    the search neighborhood beats the current point).
    """
    features = meta["features"]
    bounds = meta["bounds"]
    layers = meta["layers"]
    grids = _candidate_grid(current, bounds, features, relative_step=relative_step)

    target_res = resource.get("Total") if target_layer is None else resource.get(target_layer)
    target_demand_col = "Total_demand" if target_layer is None else f"{target_layer}_demand"
    target_demand_max = (target_pct / 100.0) * target_res if target_res else float("inf")

    ranked_features = sorted(
        features, key=lambda f: -meta["feature_importance_for_success"].get(f, 0)
    )

    best = None
    best_dist = None
    best_pred = None

    top_sets = [ranked_features[:3], ranked_features[1:4], features]
    for subset in top_sets:
        fixed = [f for f in features if f not in subset]
        value_lists = [grids[f] for f in subset]
        combos = list(itertools.product(*value_lists))[:max_candidates]
        if not combos:
            continue

        candidates = []
        for combo in combos:
            candidate = dict(current)
            for f, v in zip(subset, combo):
                candidate[f] = float(v)
            for f in fixed:
                candidate[f] = current[f]
            candidates.append(candidate)

        X = pd.DataFrame([[c[f] for f in features] for c in candidates], columns=features)
        succ_probs = clf.predict_proba(X)[:, 1]
        target_demand_vals = regressors[target_demand_col].predict(X)
        total_demand_vals = target_demand_vals if target_layer is None else regressors["Total_demand"].predict(X)
        total_res = resource.get("Total") or 1.0
        total_pcts = total_demand_vals / total_res * 100.0

        mask = (succ_probs >= 0.5) & (target_demand_vals <= target_demand_max)
        idxs = np.nonzero(mask)[0]
        for i in idxs:
            candidate = candidates[i]
            dist = _normalized_distance(current, candidate, bounds, features)
            if best is None or dist < best_dist:
                best = candidate
                best_dist = dist
                best_pred = {
                    "success_prob": float(succ_probs[i]),
                    "total_usage_pct": float(total_pcts[i]),
                    "target_metric_pct": float(usage_pct_from_demand(target_demand_vals[i], target_res)),
                }

    if best is None:
        return None

    changes = []
    for f in features:
        old_v = round(current[f], ROUND[f])
        new_v = round(best[f], ROUND[f])
        if abs(new_v - old_v) > (10 ** (-ROUND[f])) / 2:
            changes.append({"parameter": f, "from": old_v, "to": new_v, "delta": round(new_v - old_v, ROUND[f])})

    per_layer_pct = {}
    Xbest = pd.DataFrame([[best[f] for f in features]], columns=features)
    for layer in layers:
        demand_pred = float(regressors[f"{layer}_demand"].predict(Xbest)[0])
        per_layer_pct[f"{layer}_usage_pct"] = usage_pct_from_demand(demand_pred, resource.get(layer))

    return {
        "achieved": True,
        "target_layer": target_layer,
        "changes": changes,
        "new_values": {f: round(best[f], ROUND[f]) for f in features},
        "predicted_success_prob": best_pred["success_prob"],
        "predicted_total_usage_pct": best_pred["total_usage_pct"],
        "predicted_target_metric_pct": best_pred["target_metric_pct"],
        "predicted_layer_usage_pct": per_layer_pct,
    }

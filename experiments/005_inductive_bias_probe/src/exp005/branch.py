"""Spec 4.6: which layers / sub-experiments to run, decided from experiment 4's summary.json. Pure python (no numpy/torch).
Rows are evaluated in table order; when several rows apply, the upper row has priority and the lower rows' sub-experiments are added.
Layer names: enc_k<k> (encoder after k blocks, final LayerNorm applied), pred_h<m> (predictor hidden after m blocks, z-part),
pred_out<m> (predictor early-exit output after m blocks = final norm applied, z-part; m = n_blocks is the model's output)."""
from __future__ import annotations

SUBEXPS = ["5-A", "5-B", "5-C", "5-D"]


def enc(k):
    return {"name": "enc_k%d" % k, "module": "encoder", "depth": int(k), "kind": "hidden"}


def pred_h(m):
    return {"name": "pred_h%d" % m, "module": "predictor", "depth": int(m), "kind": "hidden"}


def pred_out(m):
    return {"name": "pred_out%d" % m, "module": "predictor", "depth": int(m), "kind": "output"}


def _add(plan, layers, subs):
    for l in layers:
        if l["name"] not in [x["name"] for x in plan["layers"]]:
            plan["layers"].append(l)
        cur = plan["layer_subexps"].setdefault(l["name"], [])
        for s in subs:
            if s not in cur:
                cur.append(s)
    for s in subs:
        if s not in plan["subexps"]:
            plan["subexps"].append(s)


def select(summary, exp1_negative=None):
    """summary: dict read from experiments/004_depth_truncation/results/summary.json (or None if missing).
    Returns {rows_matched, layers, subexps, hold, pilot, unresolved, notes}. Never raises on missing fields (they go to `unresolved`)."""
    plan = {"rows_matched": [], "layers": [], "layer_subexps": {}, "subexps": [], "hold": False, "pilot": None, "unresolved": [], "notes": []}
    if summary is None:
        plan["unresolved"].append("exp4_summary_missing")
        return plan
    plan["pilot"] = summary.get("pilot")
    L_enc, L_pred = summary.get("encoder_layers"), summary.get("predictor_layers")
    k_star, m_star = summary.get("k_star"), summary.get("m_star")
    if exp1_negative is True:
        plan["rows_matched"].append("exp1_negative")
        plan["hold"] = True
        if L_enc is None:
            plan["unresolved"].append("encoder_layers_missing")
        else:
            _add(plan, [enc(L_enc)], ["5-C"])
        plan["notes"].append("experiment 5 is on hold (spec 4.6 last row): only the 5-C check on the final encoder layer is prepared; report to the user")
        return plan
    if exp1_negative is None:
        plan["notes"].append("exp1_negative is null in config (not decided): treated as false")
    lb = summary.get("l_best")
    lbm = summary.get("l_best_by_module") or {}
    if k_star is not None:                                                    # row 1
        plan["rows_matched"].append("k_star")
        e_best = (lbm.get("encoder") or {}).get("depth") if lbm.get("encoder") else (lb["depth"] if lb and lb.get("module") == "encoder" else None)
        layers = []
        if e_best is None:
            plan["unresolved"].append("l_best_encoder_missing")
        else:
            layers.append(enc(e_best))
        if L_enc is None:
            plan["unresolved"].append("encoder_layers_missing")
        else:
            layers.append(enc(L_enc))
        _add(plan, layers, ["5-A", "5-B"])
    if k_star is None and m_star is not None:                                 # row 2
        plan["rows_matched"].append("m_star_only")
        layers = []
        if L_enc is None:
            plan["unresolved"].append("encoder_layers_missing")
        else:
            layers.append(enc(L_enc))
        layers += [pred_h(max(m_star - 1, 0)), pred_h(m_star)]
        _add(plan, layers, ["5-A", "5-C"])
    if summary.get("loss_ood_only") is True:                                  # row 3
        plan["rows_matched"].append("loss_ood_only")
        if lb is None:
            plan["unresolved"].append("l_best_missing")
        else:
            _add(plan, [enc(lb["depth"]) if lb["module"] == "encoder" else pred_h(lb["depth"])], ["5-B", "5-D"])
    if k_star is None and m_star is None:                                     # row 4
        plan["rows_matched"].append("no_vanishing")
        if L_pred is None:
            plan["unresolved"].append("predictor_layers_missing")
        else:
            _add(plan, [pred_out(L_pred)], ["5-A", "5-B", "5-C", "5-D"])
    if not plan["rows_matched"]:
        plan["notes"].append("no row of the 4.6 table matched (k_star set to null/non-null combinations all covered; check summary fields)")
    return plan

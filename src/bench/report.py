"""Turn the sample records of a run into the numbers we report."""
from __future__ import annotations

from collections import Counter, defaultdict

from bench import calibration


def _groups(samples: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in samples:
        e = s["evaluation"]
        out[(e["task_id"], e["variant"], e["condition"])].append(s)
    return dict(sorted(out.items()))


def summarise(samples: list[dict]) -> dict:
    """Per (task, variant, condition) and overall. `samples` are the "sample" records of a run log."""
    per_group = {}
    for key, group in _groups(samples).items():
        evs = [s["evaluation"] for s in group]
        entry = {
            "n": len(evs),
            "passed": sum(e["passed"] for e in evs),
            "signatures": dict(Counter(e["signature"] for e in evs if not e["passed"])),
            "unconfirmed": sum(not s["confirmation"]["confirmed"] for s in group),
            "first_reply_differed": sum(s["confirmation"]["first_reply_differs"] for s in group),
        }
        if key[1] == "ambiguous":
            entry["outcomes"] = dict(Counter(e["outcome"] for e in evs))
            entry["choices"] = dict(Counter(e["choice_summary"] for e in evs if e["choice_summary"]))
            entry["per_output_choices"] = {
                out: dict(Counter(e["choices"][out] for e in evs if e["choices"] and out in e["choices"]))
                for out in sorted({o for e in evs for o in (e["choices"] or {})})}
        per_group[key] = entry

    plain = [s["evaluation"] for s in samples if s["evaluation"]["variant"] == "plain"]
    calib = calibration.summarise([(e["confidence"], int(e["passed"])) for e in plain])
    return {"groups": per_group, "n_samples": len(samples), "calibration_plain": calib}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _num(x, digits=3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def format_report(summary: dict) -> str:
    lines = [f"samples: {summary['n_samples']}", ""]
    lines.append(f"{'task':20} {'variant':10} {'cond':7} {'n':>3} {'pass':>5}  what happened")
    for (task, variant, cond), g in summary["groups"].items():
        if variant == "plain":
            what = ", ".join(f"{k} x{v}" for k, v in sorted(g["signatures"].items())) or "all passed"
            passed = f"{g['passed']}/{g['n']}"
        else:
            what = ", ".join(f"{k} x{v}" for k, v in sorted(g["outcomes"].items()))
            if g["choices"]:
                what += "  |  chose: " + ", ".join(f"{k} x{v}" for k, v in sorted(g["choices"].items()))
            passed = f"{g['passed']}/{g['n']}"
        lines.append(f"{task:20} {variant:10} {cond:7} {g['n']:>3} {passed:>5}  {what}")
    flagged = [(k, g) for k, g in summary["groups"].items() if g["unconfirmed"] or g["first_reply_differed"]]
    if flagged:
        lines += ["", "confirmation: first reply differed from the confirmed one / never confirmed"]
        for (task, variant, cond), g in flagged:
            lines.append(f"  {task} {variant} {cond}: {g['first_reply_differed']} differed, {g['unconfirmed']} unconfirmed")
    c = summary["calibration_plain"]
    lines += ["", "calibration (plain specs only)",
              f"  scored {c.n} (no usable confidence: {c.n_missing}), pass rate {_pct(c.pass_rate)}",
              f"  Brier {_num(c.brier)}, baseline {_num(c.baseline)}, skill {_num(c.skill)}",
              f"  mean confidence when passing {_num(c.mean_confidence_when_passing, 1)}, "
              f"when failing {_num(c.mean_confidence_when_failing, 1)}"]
    return "\n".join(lines)

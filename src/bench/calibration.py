"""Does the model's stated confidence match whether its code passed?

Confidence is a percentage; outcomes are 1 (passed every test input) or 0. The Brier score is the mean
squared difference. On its own it is hard to read on a small benchmark, so it is reported next to an
unskilled baseline that always says the overall pass rate (Brier skill score = 1 - score / baseline),
following the baseline used for code models in arXiv 2402.02047.
"""
from __future__ import annotations

from dataclasses import dataclass


def brier(confidences: list[float], outcomes: list[int]) -> float | None:
    """Mean squared error between confidence/100 and the 0/1 outcome. None if there is nothing to score."""
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must be the same length")
    if not confidences:
        return None
    return sum((c / 100.0 - o) ** 2 for c, o in zip(confidences, outcomes)) / len(confidences)


def base_rate_brier(outcomes: list[int]) -> float | None:
    """Brier score of always predicting the overall pass rate."""
    if not outcomes:
        return None
    p = sum(outcomes) / len(outcomes)
    return sum((p - o) ** 2 for o in outcomes) / len(outcomes)


def skill(score: float | None, baseline: float | None) -> float | None:
    """1 means perfect, 0 means no better than the baseline, negative means worse. None if undefined."""
    if score is None or baseline is None or baseline == 0:
        return None
    return 1.0 - score / baseline


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


@dataclass(frozen=True)
class Calibration:
    n: int                       # samples with a usable confidence
    n_missing: int               # samples with no usable confidence (excluded from the scores)
    pass_rate: float | None
    brier: float | None
    baseline: float | None
    skill: float | None
    mean_confidence_when_passing: float | None
    mean_confidence_when_failing: float | None


def summarise(samples: list[tuple[float | None, int]]) -> Calibration:
    """`samples` is (confidence or None, passed 0/1) for each sample."""
    scored = [(c, o) for c, o in samples if c is not None]
    conf = [c for c, _ in scored]
    out = [o for _, o in scored]
    b, ref = brier(conf, out), base_rate_brier(out)
    return Calibration(
        n=len(scored), n_missing=len(samples) - len(scored),
        pass_rate=(sum(out) / len(out)) if out else None,
        brier=b, baseline=ref, skill=skill(b, ref),
        mean_confidence_when_passing=_mean([c for c, o in scored if o == 1]),
        mean_confidence_when_failing=_mean([c for c, o in scored if o == 0]),
    )

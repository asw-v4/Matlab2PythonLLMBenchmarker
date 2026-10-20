import pytest

from bench import calibration as cal


def test_perfect_confidence_gives_zero_brier():
    assert cal.brier([100, 0, 100], [1, 0, 1]) == 0.0


def test_worst_confidence_gives_one():
    assert cal.brier([0, 100], [1, 0]) == 1.0


def test_known_value():
    # (0.9-1)^2 = 0.01, (0.8-0)^2 = 0.64, (0.5-1)^2 = 0.25 -> mean 0.3
    assert cal.brier([90, 80, 50], [1, 0, 1]) == pytest.approx(0.3)


def test_brier_of_nothing_is_none_and_length_mismatch_raises():
    assert cal.brier([], []) is None
    with pytest.raises(ValueError):
        cal.brier([50], [1, 0])


def test_base_rate_brier():
    assert cal.base_rate_brier([1, 1, 0, 0]) == pytest.approx(0.25)     # p = 0.5
    assert cal.base_rate_brier([1, 1, 1]) == 0.0
    assert cal.base_rate_brier([]) is None


def test_skill_scale():
    assert cal.skill(0.0, 0.25) == 1.0
    assert cal.skill(0.25, 0.25) == 0.0
    assert cal.skill(0.5, 0.25) == -1.0


def test_skill_is_undefined_when_the_baseline_is_zero_or_missing():
    assert cal.skill(0.1, 0.0) is None and cal.skill(None, 0.25) is None and cal.skill(0.1, None) is None


def test_always_saying_the_pass_rate_has_zero_skill():
    outcomes = [1, 1, 0, 1, 0, 0, 1, 0]
    conf = [100 * sum(outcomes) / len(outcomes)] * len(outcomes)
    assert cal.skill(cal.brier(conf, outcomes), cal.base_rate_brier(outcomes)) == pytest.approx(0.0)


def test_summarise():
    s = cal.summarise([(90, 1), (80, 0), (50, 1), (None, 0)])
    assert s.n == 3 and s.n_missing == 1
    assert s.pass_rate == pytest.approx(2 / 3)
    assert s.brier == pytest.approx(0.3)
    assert s.mean_confidence_when_passing == pytest.approx(70.0)
    assert s.mean_confidence_when_failing == pytest.approx(80.0)
    assert s.skill is not None


def test_summarise_edge_cases():
    empty = cal.summarise([])
    assert empty.n == 0 and empty.brier is None and empty.skill is None
    all_missing = cal.summarise([(None, 1), (None, 0)])
    assert all_missing.n == 0 and all_missing.n_missing == 2 and all_missing.pass_rate is None
    all_pass = cal.summarise([(90, 1), (80, 1)])
    assert all_pass.mean_confidence_when_failing is None and all_pass.skill is None   # baseline is 0

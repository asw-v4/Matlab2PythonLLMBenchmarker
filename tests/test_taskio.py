from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from bench.taskio import load_task


def make_task(tmp_path: Path, toml: str, variables: dict) -> Path:
    (tmp_path / "task.toml").write_text(toml, encoding="utf-8")
    savemat(tmp_path / "expected.mat", variables)
    return tmp_path


SINGLE = '[task]\nid = "t"\n[tolerance]\nrtol = 1e-9\natol = 1e-9\n[inputs]\nas_int = ["k"]\n'
MULTI = SINGLE + '[outputs]\nnames = ["B", "D"]\n'

R = np.array([[1.0, 2.0]])


def test_single_output_uses_result_key_and_converts_scalars(tmp_path):
    d = make_task(tmp_path, SINGLE, {
        "octave_version": "11.1.0",
        "case01_in_x": R, "case01_in_k": 2.0, "case01_in_scale": 0.5,
        "case01_expected": R, "case01_wrong_a_b": R + 1, "case01_reading_alt": R * 2,
    })
    t = load_task(d)
    c = t.cases[0]
    assert t.output_names == ["result"] and t.octave_version == "11.1.0"
    assert c.inputs["k"] == 2 and isinstance(c.inputs["k"], int)
    assert c.inputs["scale"] == 0.5 and isinstance(c.inputs["scale"], float)
    assert c.inputs["x"].shape == (1, 2)
    assert list(c.expected) == ["result"]
    assert "a_b" in c.wrong and list(c.wrong["a_b"]) == ["result"]  # single underscores stay in the name
    assert list(c.readings["alt"]) == ["result"]


def test_multi_output_splits_on_double_underscore(tmp_path):
    d = make_task(tmp_path, MULTI, {
        "case01_in_A": R,
        "case01_expected__B": R, "case01_expected__D": R,
        "case01_wrong_axis0__B": R + 1, "case01_wrong_axis0__D": R + 2,
        "case01_reading_global__B": R, "case01_reading_global__D": R,
    })
    t = load_task(d)
    c = t.cases[0]
    assert t.output_names == ["B", "D"]
    assert set(c.expected) == {"B", "D"}
    assert set(c.wrong["axis0"]) == {"B", "D"} and c.wrong["axis0"]["D"][0, 0] == 3.0
    assert set(c.readings["global"]) == {"B", "D"}


def test_cases_are_ordered_by_index(tmp_path):
    d = make_task(tmp_path, SINGLE, {"case10_expected": R, "case02_expected": R, "case01_expected": R})
    assert [c.index for c in load_task(d).cases] == [1, 2, 10]


def test_missing_output_in_a_wrong_variant_is_rejected(tmp_path):
    d = make_task(tmp_path, MULTI, {
        "case01_expected__B": R, "case01_expected__D": R, "case01_wrong_x__B": R,
    })
    with pytest.raises(ValueError, match="wrong_x"):
        load_task(d)


def test_expected_missing_an_output_is_rejected(tmp_path):
    d = make_task(tmp_path, MULTI, {"case01_expected__B": R})
    with pytest.raises(ValueError, match="expected"):
        load_task(d)


def test_unknown_output_name_is_rejected(tmp_path):
    d = make_task(tmp_path, SINGLE, {"case01_expected__extra": R})
    with pytest.raises(ValueError):
        load_task(d)


def test_unrelated_variables_are_ignored(tmp_path):
    d = make_task(tmp_path, SINGLE, {"case01_expected": R, "something_else": R, "case01_note": R})
    assert len(load_task(d).cases) == 1


def test_missing_octave_version_is_reported_as_unknown(tmp_path):
    d = make_task(tmp_path, SINGLE, {"case01_expected": R})
    assert load_task(d).octave_version == "unknown"


def test_text_inputs_load_as_python_strings(tmp_path):
    d = make_task(tmp_path, SINGLE, {"case01_expected": R, "case01_in_f": "gauss", "case01_in_n": 4.0})
    c = load_task(d).cases[0]
    assert c.inputs["f"] == "gauss" and isinstance(c.inputs["f"], str)
    assert c.inputs["n"] == 4.0


def test_a_reading_may_cover_a_subset_of_the_outputs(tmp_path):
    d = make_task(tmp_path, MULTI, {
        "case01_expected__B": R, "case01_expected__D": R, "case01_reading_alt__B": R * 2,
    })
    c = load_task(d).cases[0]
    assert set(c.readings["alt"]) == {"B"}


def test_a_reading_with_an_unknown_output_is_rejected(tmp_path):
    d = make_task(tmp_path, MULTI, {
        "case01_expected__B": R, "case01_expected__D": R, "case01_reading_alt__Z": R,
    })
    with pytest.raises(ValueError, match="reading_alt"):
        load_task(d)


def test_exact_outputs_property(tmp_path):
    toml = MULTI.replace('atol = 1e-9\n', 'atol = 1e-9\nexact_outputs = ["D"]\n')
    d = make_task(tmp_path, toml, {"case01_expected__B": R, "case01_expected__D": R})
    assert load_task(d).exact_outputs == ["D"]


def test_exact_outputs_default_to_none(tmp_path):
    d = make_task(tmp_path, MULTI, {"case01_expected__B": R, "case01_expected__D": R})
    assert load_task(d).exact_outputs == []


def test_tolerance_subtables_become_overrides_for_known_outputs_only(tmp_path):
    toml = MULTI.replace('atol = 1e-9\n', 'atol = 1e-9\n[tolerance.D]\nrtol = 1e-4\natol = 1e-4\n[tolerance.zzz]\nrtol = 5\n')
    d = make_task(tmp_path, toml, {"case01_expected__B": R, "case01_expected__D": R})
    t = load_task(d)
    assert t.tolerance_overrides == {"D": {"rtol": 1e-4, "atol": 1e-4}}   # "zzz" is not an output
    assert t.tolerance["rtol"] == 1e-9


def test_compare_kwargs_bundle_matches_the_task(tmp_path):
    d = make_task(tmp_path, MULTI, {"case01_expected__B": R, "case01_expected__D": R})
    kw = load_task(d).compare_kwargs()
    assert kw == dict(names=["B", "D"], rtol=1e-9, atol=1e-9, vector_ok=False, exact=[], overrides={})

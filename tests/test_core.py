import os

import numpy as np
import pandas as pd
import pytest

from app import core

MODEL = core.load_model()
# The contest data is not redistributed; set FACIES_DATA to a local copy of
# facies_vectors.csv to check that the application reproduces the notebook.
CONTEST_DATA = os.environ.get("FACIES_DATA")


def load_demo(name):
    return core.read_csv(core.demo_wells()[name].read_bytes())


@pytest.mark.skipif(not CONTEST_DATA, reason="FACIES_DATA not set")
@pytest.mark.parametrize("name, expected_f1", [("NEWBY", 0.466), ("SHANKLE", 0.404)])
def test_test_wells_match_notebook(name, expected_f1):
    data = pd.read_csv(CONTEST_DATA)
    well = data[data["Well Name"] == name].drop(columns=["Formation"])
    result = core.predict(core.validate(well, MODEL), MODEL)
    assert result["scores"]["macro_f1"] == pytest.approx(expected_f1, abs=0.001)


@pytest.mark.parametrize("name", ["synthetic_a", "synthetic_b"])
def test_demo_wells_are_scored(name):
    result = core.predict(core.validate(load_demo(name), MODEL), MODEL)
    assert 0 < result["scores"]["macro_f1"] <= 1


def test_probabilities_are_valid():
    result = core.predict(core.validate(load_demo("synthetic_a"), MODEL), MODEL)
    proba = np.array(result["probabilities"])
    assert proba.shape == (result["samples"], 9)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-3)
    assert set(result["predicted"]) <= set(range(1, 10))


def test_without_facies_column_has_no_scores():
    data = load_demo("synthetic_a").drop(columns=["Facies", "Well Name"])
    result = core.predict(core.validate(data, MODEL), MODEL)
    assert result["true"] is None and result["scores"] is None and result["well"] is None


def test_row_order_does_not_change_predictions():
    data = load_demo("synthetic_b")
    ordered = core.predict(core.validate(data, MODEL), MODEL)["predicted"]
    shuffled = core.predict(core.validate(data.sample(frac=1, random_state=0), MODEL), MODEL)["predicted"]
    assert ordered == shuffled


def test_macro_f1():
    assert core.macro_f1(np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2])) == 1.0
    assert core.macro_f1(np.array([1, 1, 2, 2]), np.array([1, 2, 2, 2])) == pytest.approx((2 / 3 + 0.8) / 2)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns=["PE"]), "Missing column(s): PE"),
    (lambda d: d.assign(GR=d["GR"].where(d.index != 3)), "GR (1)"),
    (lambda d: d.assign(NM_M=3), "NM_M"),
    (lambda d: d.assign(RELPOS=1.5), "RELPOS"),
    (lambda d: d.assign(Facies=12), "Facies"),
    (lambda d: pd.concat([d, d.assign(**{"Well Name": "OTHER"})]), "several wells"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "Depth values must be unique"),
])
def test_invalid_input_is_reported(change, message):
    with pytest.raises(core.InputError, match=message.replace("(", r"\(").replace(")", r"\)")):
        core.validate(change(load_demo("synthetic_a")), MODEL)


def test_unreadable_file():
    with pytest.raises(core.InputError):
        core.read_csv(b"")


def test_column_names_ignore_case_and_aliases():
    data = load_demo("synthetic_a").rename(columns={"Depth": "DEPT", "GR": "gr", "PE": "PEF", "Well Name": "well"})
    result = core.predict(core.validate(data, MODEL), MODEL)
    reference = core.predict(core.validate(load_demo("synthetic_a"), MODEL), MODEL)
    assert result["well"] == "SYNTHETIC A"
    assert result["predicted"] == reference["predicted"]


def test_missing_column_message_lists_found_columns():
    data = load_demo("synthetic_a").drop(columns=["RELPOS"])
    with pytest.raises(core.InputError, match="Columns found in the file: Well Name, Depth"):
        core.validate(data, MODEL)

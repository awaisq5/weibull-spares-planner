import json
from pathlib import Path

import pytest

from weibull_spares.cli import main
from weibull_spares.data import load_parts
from weibull_spares.weibull import fit_mle

PARTS = str(Path(__file__).resolve().parent.parent / "data" / "parts_register.csv")


def test_fit_reports_both_the_censored_and_the_naive_view(capsys):
    assert main(["fit", PARTS]) == 0
    out = capsys.readouterr().out
    assert "worn out (failures)" in out
    assert "95% bounds" in out
    assert "If every scrapped part counted as a failure" in out


def test_locations_runs_the_likelihood_ratio_test(capsys):
    assert main(["locations", PARTS]) == 0
    assert "Likelihood ratio test" in capsys.readouterr().out


def test_cost_compares_against_the_naive_forecast(capsys):
    assert main(["cost", PARTS, "--unit-cost", "3950", "--horizon", "500"]) == 0
    assert "Counting every scrap as a failure would forecast" in capsys.readouterr().out


def test_spares(capsys):
    assert main(["spares", PARTS, "--lead-time-days", "21"]) == 0
    assert "stock for 95% service level" in capsys.readouterr().out


def test_reports_bad_input_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "parts.csv"
    bad.write_text("part_id,part_number,location,circulating_hours,scrap_reason\nA,PN,X,100,bent shaft\n")
    assert main(["fit", str(bad)]) == 2
    assert "Unknown scrap reason" in capsys.readouterr().err


def _json(capsys, *argv: str) -> dict:
    assert main([*argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_fit_json_matches_the_fitted_model(capsys):
    out = _json(capsys, "fit", PARTS)
    f = fit_mle(load_parts(PARTS).hours, load_parts(PARTS).failed)
    assert out["beta"] == pytest.approx(f.model.beta)
    assert out["eta_hours"] == pytest.approx(f.model.eta)
    assert out["b10_hours"] == pytest.approx(f.model.b_life(0.10))
    lo, hi = out["bounds_95"]["beta"]
    assert lo < out["beta"] < hi
    assert out["failures"] + out["suspensions"] == len(load_parts(PARTS).hours)
    assert out["naive"]["eta_hours"] < out["eta_hours"]


def test_every_command_has_json_output(capsys):
    loc = _json(capsys, "locations", PARTS)
    assert {"locations", "pooled", "p_value", "differs"} <= loc.keys()
    cost = _json(capsys, "cost", PARTS, "--unit-cost", "3950", "--horizon", "500")
    assert cost["expected_cost"] > 0 and len(cost["first_life_events_per_period"]) == 4
    sp = _json(capsys, "spares", PARTS, "--lead-time-days", "21")
    assert isinstance(sp["stock_for_service_level"], int) and sp["stock_for_service_level"] >= 0
    assert 0 < sp["cost_optimal_service_level"] <= 1


def test_json_errors_still_go_to_stderr(tmp_path, capsys):
    bad = tmp_path / "parts.csv"
    bad.write_text("part_id,part_number,location,circulating_hours,scrap_reason\nA,PN,X,100,bent shaft\n")
    assert main(["fit", str(bad), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unknown scrap reason" in captured.err

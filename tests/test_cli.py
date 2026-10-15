from pathlib import Path

from weibull_spares.cli import main

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

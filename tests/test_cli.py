from pathlib import Path

from weibull_spares.cli import main

DATA = Path(__file__).resolve().parent.parent / "data"


def test_plan_runs_on_sample_data(capsys):
    code = main(["plan", str(DATA / "bearing_pack_history.csv"), str(DATA / "fleet.csv"), "--lead-time-days", "21"])
    out = capsys.readouterr().out
    assert code == 0
    assert "112 failures, 108 suspensions" in out
    assert "stock for 95% service level" in out
    assert "Replace at" in out


def test_reports_bad_input_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "history.csv"
    bad.write_text("unit_id,hours,status\nA,100,broken\n")
    assert main(["fit", str(bad)]) == 2
    assert "status should be 'failed' or 'suspended'" in capsys.readouterr().err

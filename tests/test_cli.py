from typer.testing import CliRunner

from lang2action import __version__
from lang2action.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_run_reports_not_implemented():
    result = runner.invoke(app, ["run", "stack the red cube on the blue box"])
    assert result.exit_code == 1
    assert "milestone 3" in result.output

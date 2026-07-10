from typer.testing import CliRunner

from lang2action import __version__
from lang2action.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_run_help_documents_exit_codes():
    # the real agent path (grounding, guard, execution) is covered in
    # test_agent.py / test_agent_sim.py without needing an API key
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "refused" in result.output

from click.testing import CliRunner
from agent import cli

def test_cli_query_smoke():
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "space exploration"])

    assert result.exit_code == 0
    assert "AGENT RESPONSE" in result.output

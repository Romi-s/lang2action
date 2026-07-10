"""Command-line entrypoint: `lang2action run "stack the red cube on the blue box"`."""

import typer

from lang2action import __version__
from lang2action.config import load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"lang2action {__version__}")


@app.command()
def run(instruction: str) -> None:
    """Execute a natural-language tabletop instruction (agent lands in milestone 3)."""
    settings = load_settings()
    typer.echo(f"instruction : {instruction}")
    typer.echo(f"model       : {settings.model}")
    typer.echo(f"perception  : {settings.perception_backend}")
    typer.echo("The planning agent is not implemented yet (milestone 3) - scaffold only.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

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
def scene(
    seed: int = typer.Option(42, help="Random seed for scene generation."),
    n_objects: int = typer.Option(4, help="Number of objects on the table."),
    render_dir: str = typer.Option("", help="If set, save top/side PNG renders here."),
) -> None:
    """Generate a seeded tabletop scene and print its ground-truth scene graph."""
    # sim imports are lazy so the CLI works in environments without pybullet
    from pathlib import Path

    from lang2action.perception.sim_backend import SimGroundTruthBackend
    from lang2action.sim import TabletopWorld, generate_scene
    from lang2action.sim.camera import render, save_png

    with TabletopWorld() as world:
        world.spawn(generate_scene(seed=seed, n_objects=n_objects))
        graph = SimGroundTruthBackend(world).get_scene_graph()
        typer.echo(graph.model_dump_json(indent=2))
        if render_dir:
            out = Path(render_dir)
            out.mkdir(parents=True, exist_ok=True)
            for view in ("top", "side"):
                path = out / f"scene_seed{seed}_{view}.png"
                save_png(render(world, view=view), str(path))
                typer.echo(f"saved {path}", err=True)


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

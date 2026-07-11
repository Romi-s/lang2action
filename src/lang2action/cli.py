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
def run(
    instruction: str,
    seed: int = typer.Option(42, help="Random seed for the tabletop scene."),
    n_objects: int = typer.Option(4, help="Number of objects on the table."),
    render_dir: str = typer.Option("", help="If set, save before/after renders here."),
) -> None:
    """Run a natural-language instruction through the agent on a seeded scene.

    Exit codes: 0 = executed successfully, 2 = refused (hallucination guard /
    infeasible instruction), 1 = execution failed. Needs the API key matching
    LANG2ACTION_MODEL.
    """
    from pathlib import Path

    from lang2action.agent import InProcessRobot, run_agent
    from lang2action.agent.llm import make_llm
    from lang2action.perception.factory import make_perception
    from lang2action.sim import PyBulletExecutor, TabletopWorld, generate_scene
    from lang2action.sim.camera import render, save_png

    settings = load_settings()
    typer.echo(f"model: {settings.model} | perception: {settings.perception_backend}", err=True)

    def snapshot(world, tag: str) -> None:
        if render_dir:
            out = Path(render_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_png(render(world, view="side"), str(out / f"run_{tag}.png"))

    with TabletopWorld() as world:
        world.spawn(generate_scene(seed=seed, n_objects=n_objects))
        robot = InProcessRobot(make_perception(settings, world), PyBulletExecutor(world))
        snapshot(world, "before")
        state = run_agent(robot, make_llm(settings), instruction)
        snapshot(world, "after")

    grounding = state.get("grounding")
    if grounding is not None and grounding.feasible:
        for i, s in enumerate(grounding.steps, start=1):
            typer.echo(f"step {i}: {s.target_id} {s.relation} {s.reference_id}", err=True)
    typer.echo(f"outcome: {state['outcome']} - {state['message']}")
    raise typer.Exit(code={"success": 0, "refused": 2}.get(state["outcome"], 1))


@app.command()
def demo(
    seed: int = typer.Option(42, help="Random seed for the tabletop scene."),
    out: str = typer.Option("outputs/demo.gif", help="Output GIF path."),
) -> None:
    """Record a scripted pick-and-place sequence as a GIF (no API key needed)."""
    from pathlib import Path

    from PIL import Image

    from lang2action.action.base import PickPlace
    from lang2action.sim import PyBulletExecutor, TabletopWorld, generate_scene
    from lang2action.sim.camera import render

    frames: list = []
    with TabletopWorld() as world:
        specs = generate_scene(seed=seed, n_objects=4)
        world.spawn(specs)

        counter = {"steps": 0}

        def capture() -> None:
            counter["steps"] += 1
            if counter["steps"] % 8 == 0:
                frames.append(Image.fromarray(render(world, view="side", width=480, height=360)))

        world.on_step = capture
        executor = PyBulletExecutor(world)
        script = [
            PickPlace(target_id=specs[0].id, relation="on_top_of", reference_id=specs[1].id),
            PickPlace(target_id=specs[2].id, relation="behind", reference_id=specs[1].id),
        ]
        for action in script:
            result = executor.execute(action)
            typer.echo(
                f"{action.target_id} {action.relation} {action.reference_id}: "
                f"{'ok' if result.success else result.message}",
                err=True,
            )

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True
    )
    typer.echo(f"saved {path} ({len(frames)} frames)")


@app.command()
def eval(
    n_cases: int = typer.Option(30, help="Number of auto-generated cases."),
    base_seed: int = typer.Option(0, help="Base seed for the case generator."),
    output: str = typer.Option("", help="If set, write the full JSON report here."),
) -> None:
    """Run the eval set through the agent (needs the API key for LANG2ACTION_MODEL).

    Reports grounding accuracy, task success rate, and hallucination-refusal rate.
    """
    from pathlib import Path

    from lang2action.agent.llm import make_llm
    from lang2action.eval import build_cases, run_eval

    settings = load_settings()
    typer.echo(f"model: {settings.model}", err=True)
    cases = build_cases(n_cases=n_cases, base_seed=base_seed)

    def progress(result) -> None:
        marker = "ok" if result.outcome in ("success", "refused") else "!!"
        typer.echo(
            f"[{marker}] {result.case_id}: {result.instruction!r} -> {result.outcome}", err=True
        )

    report = run_eval(cases, make_llm(settings), on_result=progress)
    typer.echo(report.as_markdown())
    if output:
        Path(output).write_text(report.to_json(), encoding="utf-8")
        typer.echo(f"report written to {output}", err=True)


@app.command("perception-eval")
def perception_eval(
    n_scenes: int = typer.Option(10, help="Number of seeded scenes."),
    base_seed: int = typer.Option(1000, help="First scene seed."),
    output: str = typer.Option("", help="If set, write the JSON report here."),
) -> None:
    """Score the real SGG backend against simulator ground truth (no LLM involved).

    Reports object id-recall/precision and per-predicate relation precision/recall
    on co-detected pairs - the in_front_of/behind rows validate the depth predicates.
    Needs the SGG service running (LANG2ACTION_SGG_URL).
    """
    from pathlib import Path

    from lang2action.eval.perception import run_perception_eval

    settings = load_settings()
    typer.echo(f"sgg service: {settings.sgg_url}", err=True)
    report = run_perception_eval(settings.sgg_url, n_scenes=n_scenes, base_seed=base_seed)
    typer.echo(report.as_markdown())
    if output:
        Path(output).write_text(report.to_json(), encoding="utf-8")
        typer.echo(f"report written to {output}", err=True)


if __name__ == "__main__":
    app()

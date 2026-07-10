# lang2action

**A robot that understands not just what it sees, but what you tell it — and acts on it.**

Give it a natural-language instruction (*"stack the red cube on the blue box"*, *"move the cup
that's to the left of the bottle"*). It perceives the tabletop as a structured **scene graph**,
reasons over that graph with a **LangGraph agent** to resolve the referring expression and plan the
task, and executes **pick-and-place in PyBullet**. Grounded Vision-Language-Action (VLA), with an
eval harness and a hallucination guard: the agent refuses instructions that reference objects not
present in the scene.

This project composes my other work into one system: perception from
[lightweight-scene-graph](https://github.com/Romi-s) (my MSc thesis pipeline, wrapped here as an
**MCP server**), agentic reasoning patterns from my Visual QA Agent, and the eval/LLMOps discipline
from my Compliance RAG Agent.

## Architecture

```
NL command
   |
   v
[LangGraph planning agent] --MCP tool call--> [Scene-Graph MCP server]
   |  (provider-agnostic LLM)                     |  get_scene_graph / find_object
   |  - ground referring expression               |  / spatial_query
   |  - plan pick-and-place                       v
   |  - hallucination guard              [PerceptionBackend]
   v                                      sim ground truth  |  real SGG over HTTP
[ActionExecutor]  -- v1: PyBullet   (v3: ROS2 + MoveIt)
   |
   v
[Eval harness]  grounding accuracy | task success | hallucination-refusal rate
```

Two deliberate seams:

- **`PerceptionBackend`** — the agent queries the scene graph over MCP and never knows whether it
  came from simulator ground truth (`sim`) or the real scene-graph-generation service (`sgg`).
  The eval reports metrics on both, isolating perception error from reasoning error.
- **`ActionExecutor`** — PyBullet now, ROS2 + MoveIt later, without touching the agent.

## Status / roadmap (v1 in progress)

- [x] Scaffold: package layout, scene-graph schema, spatial-relation inference, CI
- [x] PyBullet tabletop world + ground-truth scene graph + pick-and-place executor
- [x] Scene-Graph MCP server: the robot as MCP tools (see below)
- [ ] LangGraph agent: perceive -> ground -> plan -> validate -> execute, with refusal guard
- [ ] Eval harness: ~30 auto-generated (instruction, scene) cases, metrics table below
- [ ] Docker + demo GIF

| Metric | sim backend | sgg backend |
| --- | --- | --- |
| Referring-expression grounding accuracy | – | – |
| Task success rate | – | – |
| Hallucination-refusal rate | – | – |

## The robot as an MCP server

The MCP server owns the simulated world; any MCP client is the robot's brain. It exposes
perception (`get_scene_graph`, `find_object`, `spatial_query`) and action
(`execute_pick_place`, `reset_scene`) tools over stdio:

```bash
python -m lang2action.mcp_server        # or: lang2action-mcp
```

The repo ships a `.mcp.json`, so opening this project in Claude Code offers the
`lang2action-robot` server automatically (adjust the interpreter path to your env) — you can
literally tell Claude Code to "stack the red cube on the blue box" and watch it perceive, plan,
and act through the tools. Scene seed and object count are set via `LANG2ACTION_SCENE_SEED` /
`LANG2ACTION_SCENE_OBJECTS`.

## Quickstart (current state)

```bash
pip install -e ".[dev]"
pytest                     # all tests run against mocks - no API key needed
lang2action scene --seed 42 --render-dir outputs   # seeded scene, ground-truth graph, PNG renders
lang2action run "stack the red cube on the blue box"
```

> **Windows note:** `pybullet` publishes no Windows wheels on PyPI, so `pip install` attempts a
> source build (requires MSVC). Use conda-forge instead:
> `conda create -n lang2action -c conda-forge python=3.12 pybullet=3.25 numpy pillow`,
> then `pip install pydantic typer httpx pytest ruff` inside the env. Linux/CI installs from pip.

Configuration is via environment variables (see `.env.example`). The LLM is a LangChain
`init_chat_model` string, so the agent is provider-agnostic: `openai:gpt-4o-mini` (default),
`anthropic:claude-haiku-4-5`, etc.

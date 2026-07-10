"""The LangGraph planning agent.

    perceive -> ground -> plan -> validate --ok--> execute -> END
                                     |
                                  refuse -> END

The LLM is called in exactly one node (ground), with structured output. The
hallucination guard in validate is deterministic code: it re-checks every id
the LLM produced against the scene graph, so a hallucinated object can never
reach the executor.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from lang2action.action.base import ActionResult, PickPlace
from lang2action.agent.robot import Robot
from lang2action.agent.schema import Grounding
from lang2action.perception.models import SceneGraph

AgentOutcome = Literal["success", "refused", "failed"]

GROUNDING_SYSTEM_PROMPT = """\
You are the grounding module of a tabletop robot. You receive the robot's scene graph
(objects with ids like "red_cube", plus pairwise spatial relations) and a natural-language
instruction. Identify the single pick-and-place step the instruction asks for.

Rules:
- target_id and reference_id MUST be copied verbatim from the scene graph object ids.
  Never invent an id.
- relation is one of: left_of, right_of, in_front_of, behind, on_top_of.
  "on" / "onto" / "on top of" mean on_top_of. "next to" / "beside" mean left_of or
  right_of - pick either. The viewer's perspective matches the graph's relations.
- Use the graph's relations to resolve descriptions like "the cube left of the box".
- If the instruction references an object that is not in the scene, set feasible=false
  and name the missing object in reason.
- If the instruction is ambiguous (matches several objects with no way to choose),
  set feasible=false and ask for clarification in reason.
"""


class AgentState(TypedDict, total=False):
    instruction: str
    scene_graph: SceneGraph
    grounding: Grounding
    actions: list[PickPlace]
    results: list[ActionResult]
    outcome: AgentOutcome
    message: str


def build_agent(robot: Robot, llm):
    """Compile the agent graph around a robot and a (structured-output-capable) LLM."""
    grounder = llm.with_structured_output(Grounding)

    def perceive(state: AgentState) -> AgentState:
        return {"scene_graph": robot.get_scene_graph()}

    def ground(state: AgentState) -> AgentState:
        graph_json = state["scene_graph"].model_dump_json()
        grounding = grounder.invoke(
            [
                ("system", GROUNDING_SYSTEM_PROMPT),
                ("human", f"Scene graph:\n{graph_json}\n\nInstruction: {state['instruction']}"),
            ]
        )
        return {"grounding": grounding}

    def plan(state: AgentState) -> AgentState:
        g = state["grounding"]
        if not g.feasible or g.target_id is None or g.relation is None or g.reference_id is None:
            return {"actions": []}
        return {
            "actions": [
                PickPlace(target_id=g.target_id, relation=g.relation, reference_id=g.reference_id)
            ]
        }

    def validate(state: AgentState) -> AgentState:
        g = state["grounding"]
        if not g.feasible:
            return {"outcome": "refused", "message": g.reason or "instruction is not feasible"}
        if not state["actions"]:
            return {"outcome": "refused", "message": "grounding produced no executable action"}
        graph = state["scene_graph"]
        for action in state["actions"]:
            for object_id in (action.target_id, action.reference_id):
                if not graph.has_object(object_id):
                    # the hallucination guard: the LLM referenced a non-existent object
                    return {
                        "outcome": "refused",
                        "message": f"refusing: '{object_id}' is not in the scene graph",
                    }
            if action.target_id == action.reference_id:
                return {
                    "outcome": "refused",
                    "message": "refusing: target and reference are the same object",
                }
        return {}

    def execute(state: AgentState) -> AgentState:
        results = [robot.execute(action) for action in state["actions"]]
        if all(r.success for r in results):
            return {"results": results, "outcome": "success", "message": "done"}
        failed = next(r for r in results if not r.success)
        return {"results": results, "outcome": "failed", "message": failed.message}

    def after_validate(state: AgentState) -> str:
        return "refuse" if state.get("outcome") == "refused" else "ok"

    graph = StateGraph(AgentState)
    graph.add_node("perceive", perceive)
    graph.add_node("ground", ground)
    graph.add_node("plan", plan)
    graph.add_node("validate", validate)
    graph.add_node("execute", execute)
    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "ground")
    graph.add_edge("ground", "plan")
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges("validate", after_validate, {"ok": "execute", "refuse": END})
    graph.add_edge("execute", END)
    return graph.compile()


def run_agent(robot: Robot, llm, instruction: str) -> AgentState:
    """Convenience wrapper: run one instruction through the compiled agent."""
    agent = build_agent(robot, llm)
    return agent.invoke({"instruction": instruction})

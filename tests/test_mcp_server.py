"""Integration test: spawn the MCP server as a real subprocess and drive it
over stdio with the official MCP client - the same protocol path Claude Code
or the LangGraph agent uses."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pybullet")
pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")

EXPECTED_TOOLS = {
    "get_scene_graph",
    "find_object",
    "spatial_query",
    "execute_pick_place",
    "reset_scene",
}


def _payload(result) -> dict:
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


def test_mcp_roundtrip():
    async def scenario():
        env = dict(
            os.environ,
            PYTHONPATH=SRC,
            LANG2ACTION_SCENE_SEED="7",
            LANG2ACTION_SCENE_OBJECTS="4",
            LANG2ACTION_PERCEPTION="sim",
        )
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "lang2action.mcp_server"], env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                assert EXPECTED_TOOLS <= {t.name for t in tools.tools}

                graph = _payload(await session.call_tool("get_scene_graph", {}))
                assert len(graph["objects"]) == 4
                ids = [o["id"] for o in graph["objects"]]

                found = _payload(
                    await session.call_tool("find_object", {"description": "a nonexistent gizmo"})
                )
                assert found["matches"] == []

                target, reference = ids[0], ids[1]
                result = _payload(
                    await session.call_tool(
                        "execute_pick_place",
                        {
                            "target_id": target,
                            "relation": "on_top_of",
                            "reference_id": reference,
                        },
                    )
                )
                assert result["success"], result["message"]

                stacked = _payload(
                    await session.call_tool(
                        "spatial_query", {"reference_id": reference, "relation": "on_top_of"}
                    )
                )
                assert target in stacked["matches"]

                assert _payload(await session.call_tool("reset_scene", {})) == {"ok": True}

    asyncio.run(scenario())

"""LangGraph pipeline: wires all agent nodes into a fan-out / fan-in graph."""

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.state import ReviewState
from src.agents.orchestrator import orchestrator_node
from src.agents.bug_detector import bug_detector_node
from src.agents.security import security_node
from src.agents.code_quality import code_quality_node
from src.agents.test_coverage import test_coverage_node
from src.agents.summarizer import summarizer_node

_SPECIALIST_AGENTS = {"bug_detector", "security", "code_quality", "test_coverage"}


def _route_to_specialists(state: ReviewState) -> list[str]:
    """Return the subset of specialist agents selected by the orchestrator."""
    return [a for a in state["active_agents"] if a in _SPECIALIST_AGENTS]


def build_graph():
    """Build and compile the multi-agent review graph.

    Topology:
        orchestrator → [specialist agents in parallel] → summarizer
    """
    graph = StateGraph(ReviewState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("bug_detector", bug_detector_node)
    graph.add_node("security", security_node)
    graph.add_node("code_quality", code_quality_node)
    graph.add_node("test_coverage", test_coverage_node)
    graph.add_node("summarizer", summarizer_node)

    graph.add_edge(START, "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route_to_specialists,
        {agent: agent for agent in _SPECIALIST_AGENTS},
    )

    for agent in _SPECIALIST_AGENTS:
        graph.add_edge(agent, "summarizer")

    graph.add_edge("summarizer", END)

    return graph.compile(checkpointer=MemorySaver())

import operator
from typing import Annotated, TypedDict


class ReviewState(TypedDict):
    # Input
    code_diff: str
    file_paths: list[str]

    # Orchestrator routing decisions
    active_agents: list[str]  # e.g. ["bug_detector", "security", "code_quality", "test_coverage"]

    # Specialist outputs (accumulated via operator.add)
    bug_report: Annotated[list[str], operator.add]
    security_report: Annotated[list[str], operator.add]
    quality_report: Annotated[list[str], operator.add]
    test_report: Annotated[list[str], operator.add]

    # Final output
    final_review: str

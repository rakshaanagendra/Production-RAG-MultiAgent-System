import operator
from typing import Annotated
from typing_extensions import TypedDict


class MultiAgentState(TypedDict):
    # Input
    query: str

    # Research Agent output
    research_context: str
    sources: list[str]
    confidence: str        # "high", "medium", "low"
    action: str            # "generate", "generate_cautiously", "retry_or_abstain"
    answerable: bool
    retrieval_strategy: str

    # Writer/Critic Agent output
    critique: str
    final_answer: str
    answer_grounded: bool

    # Observability — appends across agents, never overwrites
    agent_log: Annotated[list[str], operator.add]
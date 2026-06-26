import json
import re
import sys
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.state import MultiAgentState

# -----------------------------------------------------------------------
# LLM — llama3.1:8b for better instruction following
# No tools bound — pure evaluation task
# -----------------------------------------------------------------------
llm = ChatOllama(model="llama3.1:8b")

# -----------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """You are a strict answer grounding critic.

Your job is to check whether a given answer is grounded in the provided research context.

Rules:
1. Read the query, the final answer, and the research context carefully
2. Check every claim in the final answer against the research context
3. If ALL claims are directly supported by the context, set answer_grounded to true
4. If ANY claim cannot be traced back to the context — even minor connecting language or assumptions — set answer_grounded to false
5. Write a critique explaining your decision — be specific about which claims are grounded or not

Respond ONLY with this exact JSON format, no other text, no markdown:
{"answer_grounded": true or false, "critique": "your explanation here"}"""

# -----------------------------------------------------------------------
# Critic Node
# -----------------------------------------------------------------------
def critic_node(state: MultiAgentState) -> dict:
    """
    Validates whether final_answer is grounded in research_context.
    Hard gate on empty final_answer.
    LLM call compares answer against context for grounding.
    """
    query = state["query"]
    final_answer = state.get("final_answer", "")
    research_context = state.get("research_context", "")

    # -----------------------------------------------------------------------
    # Hard gate — nothing to critique if writer produced no answer
    # -----------------------------------------------------------------------
    if not final_answer:
        log_entry = "[CriticAgent] Skipped — final_answer is empty"
        return {
            "answer_grounded": False,
            "critique": "No answer was generated to critique.",
            "agent_log": [log_entry]
        }

    # -----------------------------------------------------------------------
    # Build human message — query + final_answer + research_context
    # -----------------------------------------------------------------------
    human_message = f"""Query: {query}

Final Answer to critique:
{final_answer}

Research Context to compare against:
{research_context}

Now evaluate whether the final answer is grounded in the research context."""

    # -----------------------------------------------------------------------
    # LLM call
    # -----------------------------------------------------------------------
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=human_message)
    ]

    response = llm.invoke(messages)
    raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()

    # Strip markdown code fences
    clean = raw.replace("```json", "").replace("```", "").strip()

    # Replace literal newlines and tabs inside JSON string values
    clean = re.sub(r'[\n\r\t]', ' ', clean)

    # -----------------------------------------------------------------------
    # Parse JSON response
    # -----------------------------------------------------------------------
    try:
        parsed = json.loads(clean)
        answer_grounded = bool(parsed.get("answer_grounded", False))
        critique = parsed.get("critique", "")
    except (json.JSONDecodeError, KeyError) as e:
        # If model ignored JSON instructions, default to not grounded
        answer_grounded = False
        critique = f"Critic parse failed — raw output: {raw}"

    log_entry = (
        f"[CriticAgent] Query: '{query}' | "
        f"Grounded: {answer_grounded} | "
        f"Critique length: {len(critique)} chars"
    )

    return {
        "answer_grounded": answer_grounded,
        "critique": critique,
        "agent_log": [log_entry]
    }


# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_state = MultiAgentState({
        "query": "What is ReAct prompting framework for LLM agents?",
        "final_answer": (
            "ReAct is a prompting framework that enhances LLM agents by interleaving "
            "reasoning and acting. It allows agents to take actions, observe results, "
            "and incorporate observations into future reasoning steps."
        ),
        "research_context": (
            "ReAct interleaves reasoning and acting to create synergy between the two. "
            "It was shown to improve performance on language and decision-making tasks. "
            "The framework generates reasoning traces and action plans simultaneously."
        ),
        "sources": ["REACT.pdf"],
        "confidence": "medium",
        "action": "generate_cautiously",
        "answerable": True,
        "retrieval_strategy": "dense",
        "critique": "",
        "answer_grounded": False,
        "agent_log": []
    })

    result = critic_node(test_state)
    print("=== Critic Node Test Result ===")
    print("Answer Grounded:", result["answer_grounded"])
    print("Critique:", result["critique"])
    print("Agent Log:", result["agent_log"])
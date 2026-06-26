import json
import sys
import re
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
# LLM — no tools bound, plain generation only
# -----------------------------------------------------------------------
llm = ChatOllama(model="qwen2.5:7b")

# -----------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------
WRITER_SYSTEM_PROMPT = """You are a research writer. Your job is to write a clear, 
accurate answer to a query using only the provided research context.

Rules:
1. Only use information from the provided context — do not add outside knowledge
2. Cite sources by filename where relevant
3. If the context is insufficient, say so clearly
4. Respond ONLY in this exact JSON format, no other text:
{
    "final_answer": "your complete answer here"
}"""

CAUTIOUS_WRITER_SYSTEM_PROMPT = """You are a research writer. Your job is to write a careful, 
hedged answer to a query using the provided research context.

Rules:
1. Only use information from the provided context — do not add outside knowledge
2. Cite sources by filename where relevant  
3. Add a disclaimer that the retrieved context had medium/low confidence
4. Respond ONLY with this exact JSON, no extra fields, no markdown, no explanation:
{
    "final_answer": "your answer here. End with: Note: Retrieved context had medium/low confidence."
}"""

# -----------------------------------------------------------------------
# Writer Node
# -----------------------------------------------------------------------
def writer_node(state: MultiAgentState) -> dict:
    """
    Generates a final answer from research context.
    Hard gates on answerable, context, and action.
    Tone determined by action field.
    """
    query = state["query"]
    research_context = state.get("research_context", "")
    answerable = state.get("answerable", False)
    action = state.get("action", "retry_or_abstain")
    sources = state.get("sources", [])

    # -----------------------------------------------------------------------
    # Hard gates — do not generate if any of these fail
    # -----------------------------------------------------------------------
    if not answerable:
        log_entry = "[WriterAgent] Skipped — answerable: False"
        return {
            "final_answer": "I cannot answer this query — the retrieved context was not relevant.",
            "agent_log": [log_entry]
        }

    if not research_context:
        log_entry = "[WriterAgent] Skipped — research_context is empty"
        return {
            "final_answer": "I cannot answer this query — no context was retrieved.",
            "agent_log": [log_entry]
        }

    if action == "retry_or_abstain":
        log_entry = "[WriterAgent] Skipped — action is retry_or_abstain"
        return {
            "final_answer": "I cannot answer this query — retrieval confidence was too low.",
            "agent_log": [log_entry]
        }

    # -----------------------------------------------------------------------
    # Select prompt based on action
    # -----------------------------------------------------------------------
    system_prompt = (
        CAUTIOUS_WRITER_SYSTEM_PROMPT
        if action == "generate_cautiously"
        else WRITER_SYSTEM_PROMPT
    )

    # -----------------------------------------------------------------------
    # Build the human message — full context with metadata for traceability
    # -----------------------------------------------------------------------
    human_message = f"""Query: {query}

Research Context:
{research_context}

Sources available: {sources}

Write your answer now."""

    # -----------------------------------------------------------------------
    # LLM call
    # -----------------------------------------------------------------------
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ]

    response = llm.invoke(messages)
    content = response.content
    raw = (content if isinstance(content, str) else str(content)).strip()

    # Strip markdown code fences if model wraps JSON in them
    clean = raw.replace("```json", "").replace("```", "").strip()

    # Replace literal newlines and tabs inside JSON string values only
    # by normalizing the entire thing through a regex
    clean = re.sub(r'[\n\r\t]', ' ', clean)

    # -----------------------------------------------------------------------
    # Parse JSON response
    # -----------------------------------------------------------------------
    try:
        parsed = json.loads(clean)
        final_answer = parsed.get("final_answer", "")
    except (json.JSONDecodeError, KeyError) as e:
        # If model ignored JSON instructions, use raw output as fallback
        print(f"JSON PARSE FAILED: {e}")
        print(f"RAW OUTPUT: {repr(raw)}")
        final_answer = raw



    log_entry = (
        f"[WriterAgent] Query: '{query}' | "
        f"Action: {action} | "
        f"Answer length: {len(final_answer)} chars | "
        f"Sources used: {len(sources)}"
    )

    return {
        "final_answer": final_answer,
        "agent_log": [log_entry]
    }


# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_state: MultiAgentState = {
        "query": "What is ReAct prompting framework for LLM agents?",
        "research_context": "Sample context about ReAct framework...",
        "answerable": True,
        "action": "generate_cautiously",
        "sources": ["react_paper.pdf"],
        "confidence": "medium",
        "retrieval_strategy": "dense",
        "critique": "",
        "final_answer": "",
        "answer_grounded": False,
        "agent_log": []
    }

    result = writer_node(test_state)
    print("=== Writer Node Test Result ===")
    print("Final Answer:", result["final_answer"])
    print("Agent Log:", result["agent_log"])
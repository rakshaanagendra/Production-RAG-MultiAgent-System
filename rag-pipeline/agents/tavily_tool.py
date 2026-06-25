from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

# TavilySearch is a prebuilt LangChain tool
# max_results controls how many search results to return
_tavily = TavilySearch(max_results=3)

@tool
def web_search(query: str) -> str:
    """Use this tool to answer general knowledge questions about people,
    places, history, science, geography, and topics NOT related to
    AI research or LLM engineering. Input must be a search query string."""

    result = _tavily.invoke({"query": query})
    return str(result)


if __name__ == "__main__":

    print("=== TEST 1: GENERAL KNOWLEDGE ===")
    result = web_search.invoke({"query": "What is the capital of France?"})
    print(result)

    print("\n=== TEST 2: SHOULD USE RAG NOT THIS ===")
    result2 = web_search.invoke({"query": "What is ReAct in LLMs?"})
    print(result2)
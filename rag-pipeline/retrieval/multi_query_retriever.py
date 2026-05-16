import re
import json
import ollama


class MultiQueryRetriever:
    def __init__(self, model_name="qwen2.5:3b", client=None):
        self.model_name = model_name
        self.client = client

    def _clean_text(self, text):
        text = text.strip()
        text = text.strip('"').strip("'")
        text = re.sub(r"\s+", " ", text)
        return text

    def _deduplicate_queries(self, queries):
        seen = set()
        unique = []

        for q in queries:
            cleaned = self._clean_text(q)

            if not cleaned:
                continue

            lowered = cleaned.lower()

            if lowered not in seen:
                unique.append(cleaned)
                seen.add(lowered)

        return unique

    def generate_queries(self, query, num_queries=4):
        query = self._clean_text(query)

        prompt = f"""
You are an expert retrieval query expansion system for RAG applications.

Generate diverse search queries that improve document retrieval quality.

Goals:
- Cover semantic variations
- Cover keyword variations
- Cover technical terminology
- Cover abbreviations if relevant
- Cover alternative phrasings

Rules:
- Keep queries short and retrieval-oriented
- Do not answer the question
- Do not explain anything
- Return ONLY a valid JSON list of strings
- Do not use markdown

Example:
[
  "dense retrieval semantic search",
  "sparse retrieval bm25 lexical search"
]

User Query:
{query}
""".strip()

        try:
            if self.client is None:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You generate multiple retrieval-focused search queries.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    options={
                        "temperature": 0.3,
                    },
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You generate multiple retrieval-focused search queries.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

            content = response["message"]["content"]
            content = self._clean_text(content)

            generated_queries = json.loads(content)

            if not isinstance(generated_queries, list):
                return [query]

            generated_queries = [
                self._clean_text(q)
                for q in generated_queries
                if isinstance(q, str)
            ]

            generated_queries.insert(0, query)

            generated_queries = self._deduplicate_queries(generated_queries)

            return generated_queries[:num_queries]

        except Exception:
            return [query]


if __name__ == "__main__":
    retriever = MultiQueryRetriever()

    test_queries = [
        "What is dense and sparse retrieval?",
        "that thing where llms forget the middle part",
        "how do vector databases work",
        "bm25 vs embeddings",
    ]

    for query in test_queries:
        print("\n==================================================")
        print("ORIGINAL QUERY:")
        print(query)

        expanded_queries = retriever.generate_queries(query)

        print("\nGENERATED QUERIES:\n")

        for i, q in enumerate(expanded_queries, start=1):
            print(f"[{i}] {q}")
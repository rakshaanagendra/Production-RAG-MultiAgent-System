import re
import ollama


class QueryRewriter:
    def __init__(self, model_name="qwen2.5:7b", client=None):
        self.model_name = model_name
        self.client = client

    def _clean_text(self, text):
        text = text.strip()
        text = text.strip('"').strip("'")
        text = re.sub(r"\s+", " ", text)
        return text

    def _fallback_rewrite(self, query):
        return self._clean_text(query)

    def rewrite(self, query):
        query = self._clean_text(query)

        prompt = f"""
You are an expert retrieval query optimizer for RAG systems.

Rewrite the user's query to maximize retrieval quality from:
- technical documentation like text files and PDFs
- research papers
- AI documentation
- engineering notes

Instructions:
- Preserve original meaning
- Add important technical keywords if useful
- Expand abbreviations when useful
- Make the query retrieval-oriented
- Improve semantic and lexical matching
- Do NOT answer the question
- Return ONLY the rewritten query

User Query:
{query}
""".strip()

        try:
            if self.client is None:
                import ollama

                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You rewrite search queries for retrieval.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You rewrite search queries for retrieval.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

            rewritten_query = response["message"]["content"]
            rewritten_query = self._clean_text(rewritten_query)

            if not rewritten_query:
                return self._fallback_rewrite(query)

            return rewritten_query

        except Exception:
            return self._fallback_rewrite(query)
        

if __name__ == "__main__":
    rewriter = QueryRewriter()

    query = "How does retrieval augmented generation reduce hallucinations in LLM systems?"

    rewritten_query = rewriter.rewrite(query)

    print("\nOriginal Query:")
    print(query)

    print("\nRewritten Query:")
    print(rewritten_query)
import ollama
import re

class Generator:
    """
    Simple confidence-aware generator.

    Responsibilities:
    - Build prompts
    - Call LLM
    - Apply confidence-aware generation behavior

    NOT responsible for:
    - Retrieval
    - Reranking
    - Validation
    - Retry logic
    """

    def __init__(self, model_name="qwen2.5:3b"):
        self.model = model_name

    def _extract_terms(self, text):
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
            "was", "were", "what", "when", "where", "which", "who", "why",
            "with", "some", "all", "name", "list", "tell", "do", "does", "did",
            "this", "these", "those", "if", "then", "than", "into", "about"
        }
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {t for t in tokens if len(t) >= 3 and t not in stopwords}

    def _split_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    # Without this, compressor shrunk one query by 25% and the other one by 4% which means behaviour was not consistent enough
    # Introducing scoring sentences against the query to keep the best ones and never drop every sentence from a chunk
    # i.e. if no sentence has any query term, we keep the first one to preserve at least some context and not drop the chunk entirely
    def _score_sentence(self, sentence, query_terms):
        """
        Score a sentence by how much it supports the query.
        Higher score = more likely to keep.
        """
        sent_terms = self._extract_terms(sentence)
        normalized_query_terms = {
            t[:-1] if t.endswith("s") else t
            for t in query_terms
        }

        normalized_sent_terms = {
            t[:-1] if t.endswith("s") else t
            for t in sent_terms
        }

        overlap_terms = normalized_query_terms & normalized_sent_terms
        overlap = len(overlap_terms)

        word_count = len(sentence.split())

        score = 0.0
        score += 2.0 * overlap

        # Prefer compact and information-dense sentences
        if word_count <= 20:
            score += 0.4
        elif word_count <= 35:
            score += 0.2
        else:
            score -= 0.2

        # Slight boost for definition-like / summary-like lines
        if ":" in sentence or sentence.startswith(("1.", "2.", "-", "*")):
            score += 0.2

        return score


    def _compress_context(self, query, context, confidence_level):
        """
        Keep only the most query-relevant sentences from each retrieved chunk,
        then enforce a global compression budget.
        """
        if not context or not context.strip():
            return context

        query_terms = self._extract_terms(query)
        if not query_terms:
            return context

        # Compression budget: stricter when confidence is lower
        # Original context = 1000 then 0.70 which means keep 700 chars but remove 300 chars --> compression =30%
        target_ratio_map = {
            "high": 0.70,
            "medium": 0.55,
            "low": 0.40,
        }
        target_ratio = target_ratio_map.get(confidence_level, 0.45)

        # How many sentences per chunk we allow to survive initially
        max_sentences_per_chunk_map = {
            "high": 2,
            "medium": 2,
            "low": 1,
        }
        max_sentences_per_chunk = max_sentences_per_chunk_map.get(confidence_level, 2)

        # Split into chunk blocks
        blocks = re.split(r"\n-{40,}\n", context.strip())
        parsed_blocks = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if "Text:\n" not in block:
                parsed_blocks.append({
                    "header": block,
                    "sentences": [],
                })
                continue

            header, text_part = block.split("Text:\n", 1)

            # Remove formatting artifacts
            text_part = re.sub(r"[-=]{5,}", " ", text_part)

            # Collapse excessive blank lines
            text_part = re.sub(r"\n\s*\n+", "\n", text_part)

            sentences = self._split_sentences(text_part)

            scored_sentences = []
            for idx, sentence in enumerate(sentences):
                if sentence.isupper():
                    continue
                if len(sentence.split()) <= 4 and sentence.isupper():
                    continue
                score = self._score_sentence(sentence, query_terms)
                scored_sentences.append({
                    "idx": idx,
                    "sentence": sentence,
                    "score": score,
                })

            if not scored_sentences:
                parsed_blocks.append({
                    "header": header,
                    "sentences": [],
                })
                continue

            # Keep only positive-scoring sentences first
            positive = [s for s in scored_sentences if s["score"] > 0.0]

            # If nothing matches, keep the first sentence so the chunk is not lost entirely
            if not positive:
                kept = [scored_sentences[0]]
            else:
                # Keep the best few, then restore original order
                best = sorted(
                    positive,
                    key=lambda x: (-x["score"], x["idx"])
                )[:max_sentences_per_chunk]

                best_idx = {item["idx"] for item in best}
                kept = [s for s in scored_sentences if s["idx"] in best_idx]

                # If we still kept nothing for some reason, fall back safely
                if not kept:
                    kept = [scored_sentences[0]]

            parsed_blocks.append({
                "header": header,
                "sentences": kept,
            })

        def build_text(block_list):
            parts = []
            for item in block_list:
                header = item["header"].strip()
                sentences = item["sentences"]

                if not sentences:
                    parts.append(header)
                    continue

                sentence_text = " ".join(s["sentence"] for s in sentences)
                parts.append(f"{header}\nText:\n{sentence_text}")

            return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(parts) if parts else ""

        compressed = build_text(parsed_blocks)

        # Global budget enforcement
        original_len = len(context)
        target_len = max(int(original_len * target_ratio), 300) # Never go below 300 chars to avoid over-compression on already small contexts
        # E.g. original len = 400 and target ratio = 0.45 -> target_len = 180 chars, but we set a floor of 300 chars to avoid over-compressing already small contexts and losing all useful info
        # But if original len = 2000 and target ratio = 0.45 -> target_len = 900 chars, which is fine to enforce compression

        if len(compressed) <= target_len:
            return compressed

        # Flatten all candidate sentences so we can prune low-value ones globally
        flat_items = []
        for block_idx, item in enumerate(parsed_blocks):
            for sent in item["sentences"]:
                flat_items.append({
                    "block_idx": block_idx,
                    "idx": sent["idx"],
                    "sentence": sent["sentence"],
                    "score": sent["score"],
                })

        # Nothing to prune
        if len(flat_items) <= len(parsed_blocks):
            return compressed

        # Ensure at least one sentence per block survives
        while len(compressed) > target_len:
            removable = []

            for block_idx, item in enumerate(parsed_blocks):
                if len(item["sentences"]) <= 1:
                    continue

                for s in item["sentences"]:
                    removable.append((s["score"], block_idx, s["idx"]))

            if not removable:
                break

            # Remove the weakest sentence available
            removable.sort(key=lambda x: (x[0], x[1], x[2]))
            _, b_idx, s_idx = removable[0]

            parsed_blocks[b_idx]["sentences"] = [
                s for s in parsed_blocks[b_idx]["sentences"]
                if s["idx"] != s_idx
            ]

            compressed = build_text(parsed_blocks)

        return compressed


    def _build_prompt(self, query, context, confidence_level):
        """
        Build a prompt based on retrieval confidence.
        """

        if confidence_level == "high":

            instructions = """
                You are a factual RAG assistant.

                Rules:
                - Use ONLY the provided context.
                - Answer the user's question directly.
                - Do not invent information.
                - If the answer is not supported by the context, say:
                "I don't know."
                - Keep the answer concise and informative.
                """

        elif confidence_level == "medium":

            instructions = """
                You are a cautious factual RAG assistant.

                Rules:
                - Use ONLY the provided context.
                - Answer ONLY information explicitly stated in the context.
                - Do not infer, speculate, or fill gaps.
                - Author lists, researcher names, paper metadata,
                  publication details, and titles are valid evidence.
                - If the answer appears anywhere in the context, extract it and answer.
                - If any part of the answer is unsupported, say:
                "I don't know."
                - Be conservative.
                """

        else:
            instructions = """
                You are a factual assistant.

                 Rules:
                - Use ONLY the provided context.
                - If the answer appears in the context,
                answer it.
                - Author lists, metadata, paper titles,
                and publication information count as evidence.
                - Only say "I don't know" when the answer
                is genuinely absent from the context.
                """

        prompt = f"""
{instructions}

CONTEXT:
{context}

USER QUESTION:
{query}

Answer directly and do not repeat the question.

ANSWER:
"""

        return prompt

    def _generate(self, prompt):
        """
        Single LLM call.
        """

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.0,
                "num_predict": 300
            }
        )

        return response["message"]["content"].strip()

    # Clean public method to call from pipeline, which handles context compression based on confidence route
    def compress_context(self, query, context, confidence_route):
        confidence = confidence_route.get("confidence", "low") if confidence_route else "low"
        return self._compress_context(query, context, confidence)


    def generate(self, query, context, confidence_route, already_compressed=False):
        """
        Main generation entrypoint.
        """

        confidence = confidence_route.get("confidence", "low")
        action = confidence_route.get("action", "retry_or_abstain")

        print("\n==================================================")
        print("GENERATION ROUTING")
        print("==================================================")
        print(f"Confidence: {confidence}")
        print(f"Action: {action}")

        # LOW confidence -- abstain i.e. adaptive retrieval has already been done based on health check
        # but the confidence is still low by the time we reach generator.generate() so we abstain
        #Note that no retries happen in generation
        self.debug_mode = True
        if action == "retry_failed_abstain":

            if not self.debug_mode:
                return (
                "I do not have enough evidence in the knowledge base "
                "to answer this question."
            )
            print(
                "WARNING: Low confidence route triggered."
            )

        prompt_context = context if already_compressed else self.compress_context(
            query=query,
            context=context,
            confidence_route=confidence_route,
        )

        prompt = self._build_prompt(
            query=query,
            context=prompt_context,
            confidence_level=confidence
        )

        # print("\nCOMPRESSED CONTEXT")
        # print("=" * 50)
        # print(prompt_context[:1500])

        print("\n" + "=" * 80)
        print("FINAL PROMPT")
        print("=" * 80)
        print(prompt)

        return self._generate(prompt)
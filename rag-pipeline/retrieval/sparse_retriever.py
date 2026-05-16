from rank_bm25 import BM25Okapi
import numpy as np
import re

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class SparseRetriever:
    def __init__(self, chunks):
        self.texts = [chunk["text"] for chunk in chunks]
        self.tokenized_corpus = [_tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunks = chunks

    def search(self, query, top_k=12, min_score=0.0):
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] < min_score:
                continue
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results
                
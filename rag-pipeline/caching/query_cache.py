from pathlib import Path

from caching.cache_manager import CacheManager


class QueryCache:

    def __init__(self):

        cache_file = (
            Path(__file__).resolve().parent
            / "query_cache.json"
        )

        self.cache = CacheManager(cache_file)

    def get(self, query):

        result = self.cache.get(
            query.lower().strip()
        )

        return result if result is not None else []

    def set(
        self,
        query,
        expanded_queries
    ):

        self.cache.set(
            query.lower().strip(),
            expanded_queries
        )

    def exists(
        self,
        query
    ):

        return self.cache.exists(
            query.lower().strip()
        )
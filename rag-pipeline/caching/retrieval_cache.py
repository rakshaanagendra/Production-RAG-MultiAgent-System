from caching.cache_manager import CacheManager
from pathlib import Path


class RetrievalCache:

    def __init__(self):

        cache_file = (
            Path(__file__).resolve().parent
            / "retrieval_cache.json"
        )

        self.cache = CacheManager(cache_file)
        # print("\nRETRIEVAL CACHE FILE")
        # print(cache_file)

    def _key(
        self,
        query,
        strategy,
    ):
        return f"{query.strip().lower()}::{strategy}"

    def exists(
        self,
        query,
        strategy,
    ):
        return self.cache.exists(
            self._key(
                query,
                strategy,
            )
        )

    def get(
        self,
        query,
        strategy,
    ):
        return self.cache.get(
            self._key(
                query,
                strategy,
            )
        )

    def set(
        self,
        query,
        strategy,
        data,
    ):
        self.cache.set(
            self._key(
                query,
                strategy,
            ),
            data,
        )
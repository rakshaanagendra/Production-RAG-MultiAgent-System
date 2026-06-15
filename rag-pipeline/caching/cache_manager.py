from pathlib import Path
import json


class CacheManager:

    def __init__(self, cache_file):

        self.cache_file = Path(cache_file)

        if self.cache_file.exists():

            with open(
                self.cache_file,
                "r",
                encoding="utf-8"
            ) as f:

                self.cache = json.load(f)

        else:
            self.cache = {}

        # print("\nCACHE MANAGER INIT")
        # print(f"file={self.cache_file}")

        # if self.cache_file.exists():
        #     print("file exists=True")

        #     with open(
        #         self.cache_file,
        #         "r",
        #         encoding="utf-8"
        #     ) as f:
        #         self.cache = json.load(f)

        #     print(f"loaded keys={len(self.cache)}")

        #     sample_keys = list(self.cache.keys())[:5]
        #     print(sample_keys)

        # else:
        #     print("file exists=False")
        #     self.cache = {}

            
    def get(self, key):

        return self.cache.get(key)

    def set(self, key, value):

        self.cache[key] = value

        with open(
            self.cache_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.cache,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def exists(self, key):

        # print(f"lookup={key}")
        # print(f"cache_size={len(self.cache)}")
        # print(f"contains={key in self.cache}")

        return key in self.cache
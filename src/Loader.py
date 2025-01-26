# Loader.py

from ftfy import fix_encoding
from abc import ABC, abstractmethod
import os
import json
from Fetcher import Fetcher


class Loader(ABC):
    def __init__(self, path: str):
        self.data = self.load_data(path)

    @abstractmethod
    def load_data(self, path: str):
        pass

    @abstractmethod
    def get_entry(self, query: str):
        pass


class WebLoader(Loader):
    def load_data(self, path: str):
        if not path:
            path = "https://www.jeuxdemots.org/rezo-dump.php"

        return Fetcher(path)

    def get_entry(self, query: str):
        word, r_id, r_type = query.split(":", 2)

        if not r_type:
            r_type = "relin=norelin"
        else:
            r_type = "relout=norelout"

        if os.path.exists(f"data/cache/{word}@{r_id}@{r_type}.json"):
            try:
                with open(
                    f"data/cache/{word}@{r_id}@{r_type}.json", "r", encoding="utf-8"
                ) as json_file:
                    return json.load(json_file)
            except Exception as e:
                print(f"An error occurred while loading the parsed file: {e}")

        data = {
            "e": [],
            "r": [],
        }

        try:
            page = self.data.fetch(
                f"?gotermsubmit=Chercher&gotermrel={word}&rel={r_id}&{r_type}"
            )

            if not page:
                return None

            for line in page.split("\n"):
                line = line.strip()
                split_line = line.split(";")

                if not line.startswith("e;") and not line.startswith("r;"):
                    continue

                if line.startswith("e;"):
                    # e;eid;'name';type;w;'formated name'
                    while len(split_line) < 6:
                        split_line.append("")

                    split_line = [part.strip().strip("'") for part in split_line]

                    data["e"].append(split_line[1:])
                else:
                    # r;rid;node1;node2;type;w;w_normed;rank
                    while len(split_line) < 8:
                        split_line.append("")

                    split_line = [part.strip().strip("'") for part in split_line]

                    data["r"].append(split_line[1:])

            with open(
                f"data/cache/{word}@{r_id}@{r_type}.json", "w", encoding="utf-8"
            ) as json_file:
                json.dump(data, json_file)
        except Exception as e:
            print(f"An error occurred while fetching the data: {e}")

        return data


if __name__ == "__main__":
    web_loader = WebLoader("")
    print(web_loader.get_entry("chat:19:"))
    print(web_loader.get_entry("chat:4:"))


class MWELoader(Loader):
    def load_data(self, path):
        base_path, ext = os.path.splitext(path)
        cleaned_path = f"{base_path}.cleaned{ext}.json"

        if os.path.exists(cleaned_path):
            try:
                with open(cleaned_path, "r", encoding="utf-8") as cleaned_file:
                    return json.load(cleaned_file)
            except Exception as e:
                print(f"An error occurred while loading the cleaned file: {e}")

        data = {}

        try:
            with open(path, "r", encoding="latin-1") as file:
                for line in file:
                    line = fix_encoding(line).strip()

                    if not line or line[-1] != ";":
                        continue

                    split_line = line.split(";")

                    if len(split_line) != 3:
                        continue

                    id, mwe, _ = [part.strip().strip('"') for part in split_line]
                    data[mwe] = id

            with open(cleaned_path, "w", encoding="utf-8") as cleaned_file:
                json.dump(data, cleaned_file)
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

        return data

    def get_entry(self, query: str):
        return self.data.get(query, None)


if __name__ == "__main__":
    mwe = MWELoader("data/mwe.txt")
    print(mwe.get_entry("à la queue leu leu"))
    print(mwe.get_entry("vivre d'amour et d'eau fraîche"))

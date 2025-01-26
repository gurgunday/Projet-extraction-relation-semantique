# Fetcher.py

import requests


class Fetcher:
    def __init__(self, url: str):
        self.url = url

    def fetch(self, query: str):
        response = requests.get(self.url + query)

        if response.status_code != 200:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

            raise requests.RequestException

        return response.text

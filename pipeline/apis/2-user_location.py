#!/usr/bin/env python3
"""Script that fetches and displays location data from a given API URL."""

import sys
import time

import requests


if __name__ == "__main__":
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data.get("location"))
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-Ratelimit-Reset"))
        current_time = time.time()
        X = (reset_time - current_time) // 60
        print(f"Reset in {int(X)} min")

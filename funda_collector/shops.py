#!/usr/bin/env python3

import requests


query = """
[out:json][timeout:25];
node["shop"="supermarket"](52.30,4.70,52.45,5.05);
out;
"""

url = "https://overpass-api.de/api/interpreter"

r = requests.post(
    url,
    data={"data": query},
    headers={"User-Agent": "housing-neural-network"}
)

print(r.json)
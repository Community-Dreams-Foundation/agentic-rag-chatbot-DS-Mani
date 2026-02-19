from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode_city(name: str, country: str | None = None) -> dict:
    query = name.strip()
    if not query:
        raise ValueError("City name is required.")

    params = {
        "name": query,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    if country:
        params["country"] = country.strip()

    req = Request(f"{BASE_URL}?{urlencode(params)}", headers={"User-Agent": "agentic-rag-demo/1.0"})
    with urlopen(req, timeout=10) as resp:
        payload = resp.read().decode("utf-8")

    data = json.loads(payload)
    results = data.get("results") or []
    if not results:
        raise ValueError("City not found.")

    result = results[0]
    return {
        "name": result.get("name"),
        "country_code": result.get("country_code"),
        "admin1": result.get("admin1"),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
    }

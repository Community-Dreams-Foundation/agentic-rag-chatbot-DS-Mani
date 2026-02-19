from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode_search(name: str, country: str | None = None, limit: int = 5) -> list[dict]:
    query = name.strip()
    if not query:
        raise ValueError("City name is required.")
    if limit < 1 or limit > 20:
        raise ValueError("limit must be between 1 and 20")

    params = {
        "name": query,
        "count": limit,
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

    cleaned: list[dict] = []
    for result in results:
        cleaned.append(
            {
                "name": result.get("name"),
                "country_code": result.get("country_code"),
                "admin1": result.get("admin1"),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
            }
        )
    return cleaned


def geocode_city(name: str, country: str | None = None) -> dict:
    results = geocode_search(name, country=country, limit=1)
    return results[0]

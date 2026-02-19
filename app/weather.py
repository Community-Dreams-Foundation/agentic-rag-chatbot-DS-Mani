from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BASE_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _safe_date_range(start: str, end: str, max_days: int = 31) -> tuple[str, str]:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt < start_dt:
        raise ValueError("end date must be after start date")
    if (end_dt - start_dt).days > max_days:
        raise ValueError(f"date range too large (max {max_days} days)")
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _fetch(url: str, params: dict) -> dict:
    req = Request(url, headers={"User-Agent": "agentic-rag-demo/1.0"})
    with urlopen(f"{url}?{urlencode(params)}", timeout=10) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def _merge_hourly_series(parts: list[dict]) -> dict:
    if not parts:
        return {}
    merged: dict[str, list] = {}
    for part in parts:
        hourly = part.get("hourly", {})
        for key, values in hourly.items():
            if key not in merged:
                merged[key] = []
            merged[key].extend(values or [])
    return {"hourly": merged}


def fetch_hourly_series(lat: float, lon: float, start: str, end: str, variable: str = "temperature_2m") -> dict:
    start_date, end_date = _safe_date_range(start, end)
    start_dt = _parse_date(start_date).date()
    end_dt = _parse_date(end_date).date()
    today = datetime.utcnow().date()

    base_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": variable,
        "timezone": "UTC",
    }

    if end_dt < today:
        params = {**base_params, "start_date": start_date, "end_date": end_date}
        return _fetch(BASE_ARCHIVE_URL, params)

    if start_dt >= today:
        params = {**base_params, "start_date": start_date, "end_date": end_date}
        return _fetch(BASE_FORECAST_URL, params)

    past_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    past_params = {**base_params, "start_date": start_date, "end_date": past_end}
    future_params = {**base_params, "start_date": today.strftime("%Y-%m-%d"), "end_date": end_date}

    past = _fetch(BASE_ARCHIVE_URL, past_params)
    future = _fetch(BASE_FORECAST_URL, future_params)
    return _merge_hourly_series([past, future])


def analyze_series(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "missing": 0, "mean": None, "std": None, "min": None, "max": None, "anomalies": 0}

    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return {"count": len(values), "missing": len(values), "mean": None, "std": None, "min": None, "max": None, "anomalies": 0}

    mean = sum(cleaned) / len(cleaned)
    variance = sum((v - mean) ** 2 for v in cleaned) / len(cleaned)
    std = variance ** 0.5
    anomalies = sum(1 for v in cleaned if abs(v - mean) > 2 * std) if std > 0 else 0

    return {
        "count": len(values),
        "missing": len(values) - len(cleaned),
        "mean": mean,
        "std": std,
        "min": min(cleaned),
        "max": max(cleaned),
        "anomalies": anomalies,
    }


def run_weather_analysis(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    data = fetch_hourly_series(lat, lon, start, end)
    hourly = data.get("hourly", {})
    values = hourly.get("temperature_2m", []) if hourly else []
    stats = analyze_series(values)
    return {
        "location": {"lat": lat, "lon": lon},
        "start_date": start,
        "end_date": end,
        "metric": "temperature_2m",
        "stats": stats,
    }

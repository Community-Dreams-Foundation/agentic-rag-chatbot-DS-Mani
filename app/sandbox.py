from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


DOCKER_IMAGE = "python:3.11-slim"


class SandboxError(RuntimeError):
    pass


@dataclass
class SandboxResult:
    payload: dict
    stdout: str
    stderr: str


_JOB_SCRIPT = """\
import json
from datetime import datetime, timedelta
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
    req = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "agentic-rag-sandbox/1.0"})
    with urlopen(req, timeout=10) as resp:
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


def analyze_series(values: list[float]) -> dict:
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


def main() -> None:
    with open("input.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    lat = float(payload["lat"])
    lon = float(payload["lon"])
    start = str(payload["start"])
    end = str(payload["end"])
    variable = payload.get("variable", "temperature_2m")

    data = fetch_hourly_series(lat, lon, start, end, variable=variable)
    hourly = data.get("hourly", {})
    values = hourly.get(variable, []) if hourly else []
    stats = analyze_series(values)

    output = {
        "location": {"lat": lat, "lon": lon},
        "start_date": start,
        "end_date": end,
        "metric": variable,
        "stats": stats,
    }

    with open("output.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh)


if __name__ == "__main__":
    main()
"""


def _ensure_docker_available() -> None:
    if shutil.which("docker") is None:
        raise SandboxError("Docker not found. Install Docker Desktop to use the sandbox.")
    try:
        subprocess.run(
            ["docker", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=5,
        )
    except Exception as exc:
        raise SandboxError("Docker is not running or not reachable.") from exc


def run_weather_in_docker(lat: float, lon: float, start: str, end: str) -> SandboxResult:
    _ensure_docker_available()

    payload = {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "variable": "temperature_2m",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        workdir = Path(tmp_dir)
        input_path = workdir / "input.json"
        output_path = workdir / "output.json"
        script_path = workdir / "job.py"

        input_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path.write_text(_JOB_SCRIPT, encoding="utf-8")

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "bridge",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            f"{workdir}:/work:rw",
            "-w",
            "/work",
            DOCKER_IMAGE,
            "python",
            "/work/job.py",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise SandboxError((result.stderr or result.stdout).strip() or "Sandbox failed.")

        if not output_path.exists():
            raise SandboxError("Sandbox did not produce output.")

        output = json.loads(output_path.read_text(encoding="utf-8"))
        return SandboxResult(payload=output, stdout=result.stdout, stderr=result.stderr)

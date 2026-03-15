from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from detection.detector import AnomalyResult
from generator.data_generator import DataPoint
from models.user_profile import UserProfile
from processing.insight_engine import InsightEngine

_DATA_FORMAT = (
    "[DATA]  {ts} | {user} | "
    "HR={hr:.1f} bpm | SpO2={spo2:.1f}% | Temp={temp:.2f}C | Activity={activity}"
)

_ALERT_FORMAT = (
    "[ALERT] {ts} | {user} | "
    "HR={hr:.1f} bpm | SpO2={spo2:.1f}% | Temp={temp:.2f}C | Activity={activity} | "
    "Score={score:.5f} | Drivers: {drivers} | \033[93mInsight: {insight}\033[0m"
)

_WARMUP_FORMAT = "[WARMUP] Gathering physiological baseline for {user}: {current}/{total} observations..."
_READY_FORMAT = "\033[92m[READY]\033[0m  IsolationForest Model fitted and monitoring {user} live stream."


def _driver_string(contributors: list[tuple[str, float]]) -> str:
    return ", ".join(f"{name}({val:+.3f})" for name, val in contributors)


class AsyncNotifier:
    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

        logger = logging.getLogger("wearable")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            formatter = logging.Formatter("%(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.addHandler(file_handler)

        self._logger = logger

    def _base_fields(self, point: DataPoint, profile: UserProfile) -> dict:
        return {
            "ts": point.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "user": profile.describe(),
            "hr": point.heart_rate,
            "spo2": point.spo2,
            "temp": point.body_temperature,
            "activity": point.activity.value,
        }

    async def log_normal(self, point: DataPoint, profile: UserProfile) -> None:
        async with self._lock:
            msg = _DATA_FORMAT.format(**self._base_fields(point, profile))
            self._logger.info(msg)

    async def log_warmup_progress(self, profile: UserProfile, current: int, total: int) -> None:
        # Only print progress periodically so we don't spam the console too heavily
        if current % 30 == 0 or current == total:
            async with self._lock:
                msg = _WARMUP_FORMAT.format(user=profile.name, current=current, total=total)
                self._logger.info(msg)

    async def log_model_ready(self, profile: UserProfile) -> None:
        async with self._lock:
            msg = _READY_FORMAT.format(user=profile.name)
            self._logger.info(msg)

    async def log_anomaly(
        self, point: DataPoint, profile: UserProfile, result: AnomalyResult
    ) -> None:
        async with self._lock:
            fields = self._base_fields(point, profile)
            insight = InsightEngine.generate_insight(point, profile, result)
            msg = _ALERT_FORMAT.format(
                **fields,
                score=result.score,
                drivers=_driver_string(result.top_contributors),
                insight=insight,
            )
            self._logger.warning(msg)

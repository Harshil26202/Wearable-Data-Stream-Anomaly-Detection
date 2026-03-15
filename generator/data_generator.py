from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Iterator

import numpy as np

from models.user_profile import Gender, Region, UserProfile


class ActivityLevel(str, Enum):
    REST = "rest"
    WALK = "walk"
    RUN = "run"


# Adjusted for Indian lifestyle context: higher sedentary tendency, reduced
# probability of sustained running especially in hot-climate regions.
_ACTIVITY_TRANSITIONS: dict[ActivityLevel, list[tuple[ActivityLevel, float]]] = {
    ActivityLevel.REST: [
        (ActivityLevel.REST, 0.93),
        (ActivityLevel.WALK, 0.06),
        (ActivityLevel.RUN, 0.01),
    ],
    ActivityLevel.WALK: [
        (ActivityLevel.REST, 0.20),
        (ActivityLevel.WALK, 0.70),
        (ActivityLevel.RUN, 0.10),
    ],
    ActivityLevel.RUN: [
        (ActivityLevel.REST, 0.05),
        (ActivityLevel.WALK, 0.25),
        (ActivityLevel.RUN, 0.70),
    ],
}

# (mean_offset_above_resting_hr, std_dev) per activity state
_ACTIVITY_HR_PARAMS: dict[ActivityLevel, tuple[float, float]] = {
    ActivityLevel.REST: (0.0, 4.0),
    ActivityLevel.WALK: (25.0, 7.0),
    ActivityLevel.RUN: (60.0, 12.0),
}

# SpO2 decreases slightly under exertion; more pronounced in highland regions.
_ACTIVITY_SPO2_OFFSET: dict[ActivityLevel, float] = {
    ActivityLevel.REST: 0.0,
    ActivityLevel.WALK: -0.3,
    ActivityLevel.RUN: -0.8,
}


@dataclass
class DataPoint:
    timestamp: datetime
    user_id: str
    heart_rate: float
    spo2: float
    body_temperature: float
    activity: ActivityLevel
    age: int
    gender: Gender
    region: Region
    injected_anomaly: str | None


class HealthDataGenerator:
    def __init__(
        self,
        profile: UserProfile,
        anomaly_rate: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self._profile = profile
        self._anomaly_rate = anomaly_rate
        self._rng = np.random.default_rng(seed)
        self._activity = ActivityLevel.REST

    def _transition_activity(self) -> ActivityLevel:
        transitions = _ACTIVITY_TRANSITIONS[self._activity]
        states = [s for s, _ in transitions]
        probs = [p for _, p in transitions]
        idx = self._rng.choice(len(states), p=probs)
        return states[idx]

    def _normal_point(self) -> DataPoint:
        p = self._profile
        hr_offset, hr_std = _ACTIVITY_HR_PARAMS[self._activity]
        heart_rate = float(self._rng.normal(p.resting_hr + hr_offset, hr_std))
        heart_rate = float(np.clip(heart_rate, 35.0, p.max_hr))

        spo2 = float(
            self._rng.normal(
                p.baseline_spo2 + _ACTIVITY_SPO2_OFFSET[self._activity], 0.4
            )
        )
        spo2 = float(np.clip(spo2, 82.0, 100.0))

        body_temp = float(self._rng.normal(p.baseline_temp, 0.1))
        body_temp = float(np.clip(body_temp, 35.0, 41.0))

        return DataPoint(
            timestamp=datetime.now(tz=timezone.utc),
            user_id=p.user_id,
            heart_rate=round(heart_rate, 1),
            spo2=round(spo2, 1),
            body_temperature=round(body_temp, 2),
            activity=self._activity,
            age=p.age,
            gender=p.gender,
            region=p.region,
            injected_anomaly=None,
        )

    def _inject_anomaly(self, point: DataPoint, force_type: str | None = None) -> DataPoint:
        anomaly_types = [
            "hr_spike_at_rest",
            "spo2_drop",
            "fever",
            "bradycardia",
            "combined_stress",
            "heat_stroke",
        ]
        anomaly_type: str = force_type if force_type else self._rng.choice(anomaly_types)  # type: ignore[assignment]
        hr = point.heart_rate
        spo2 = point.spo2
        temp = point.body_temperature

        if anomaly_type == "hr_spike_at_rest":
            hr = float(self._rng.uniform(135.0, 165.0))
        elif anomaly_type == "spo2_drop":
            spo2 = float(self._rng.uniform(82.0, 91.0))
        elif anomaly_type == "fever":
            temp = float(self._rng.uniform(38.6, 40.2))
        elif anomaly_type == "bradycardia":
            hr = float(self._rng.uniform(28.0, 44.0))
        elif anomaly_type == "combined_stress":
            hr = float(self._rng.uniform(130.0, 158.0))
            spo2 = float(self._rng.uniform(88.0, 93.0))
        elif anomaly_type == "heat_stroke":
            # Heat stroke: body fails to regulate temperature. Core temp rises
            # sharply, HR compensates, and the person is typically at rest or
            # has just stopped activity — common in Indian summer months.
            hr = float(self._rng.uniform(120.0, 145.0))
            temp = float(self._rng.uniform(39.5, 41.0))
            spo2 = float(self._rng.uniform(92.0, 96.0))

        return replace(
            point,
            heart_rate=round(hr, 1),
            spo2=round(spo2, 1),
            body_temperature=round(temp, 2),
            injected_anomaly=anomaly_type,
        )

    def warmup_stream(self) -> Iterator[DataPoint]:
        """Yields clean observations with no anomaly injection.
        Used exclusively during the model training (warm-up) phase to ensure
        the IsolationForest learns only from normal physiological data.
        """
        while True:
            self._activity = self._transition_activity()
            yield self._normal_point()

    def stream(self) -> Iterator[DataPoint]:
        """Yields live observations with configured anomaly injection rate."""
        while True:
            self._activity = self._transition_activity()
            point = self._normal_point()
            if random.random() < self._anomaly_rate:
                point = self._inject_anomaly(point)
            yield point

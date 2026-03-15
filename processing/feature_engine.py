from __future__ import annotations

from collections import deque

import numpy as np

from config import ROLLING_WINDOW
from generator.data_generator import ActivityLevel, DataPoint
from models.user_profile import Gender, Region, UserProfile

_ACTIVITY_IDX: dict[ActivityLevel, int] = {
    ActivityLevel.REST: 0,
    ActivityLevel.WALK: 1,
    ActivityLevel.RUN: 2,
}

_GENDER_IDX: dict[Gender, int] = {
    Gender.MALE: 0,
    Gender.FEMALE: 1,
}

_REGION_IDX: dict[Region, int] = {
    Region.NORTH_PLAINS: 0,
    Region.COASTAL_SOUTH: 1,
    Region.ARID_WEST: 2,
    Region.HIGHLAND: 3,
    Region.NORTHEAST: 4,
}

FEATURE_NAMES: list[str] = [
    "heart_rate",
    "spo2",
    "body_temperature",
    "activity_encoded",
    "age",
    "gender_encoded",
    "region_encoded",
    "humidity_index",
    "hr_reserve_pct",
    "rolling_hr_mean",
    "rolling_hr_std",
    "spo2_drop",
    "temp_deviation",
]


class FeatureEngine:
    def __init__(self, profile: UserProfile, window_size: int = ROLLING_WINDOW) -> None:
        self._profile = profile
        self._hr_window: deque[float] = deque(maxlen=window_size)
        self._spo2_window: deque[float] = deque(maxlen=window_size)

    def transform(self, point: DataPoint) -> np.ndarray:
        self._hr_window.append(point.heart_rate)
        self._spo2_window.append(point.spo2)

        hr_arr = np.asarray(self._hr_window, dtype=np.float64)
        spo2_arr = np.asarray(self._spo2_window, dtype=np.float64)

        rolling_hr_mean = float(hr_arr.mean())
        rolling_hr_std = float(hr_arr.std()) if len(hr_arr) > 1 else 0.0
        spo2_baseline = float(spo2_arr.mean())

        # Positive value indicates current SpO2 is below its rolling baseline.
        spo2_drop = spo2_baseline - point.spo2

        # Fraction of usable HR range currently consumed (0 = resting, 1 = max).
        hr_range = self._profile.max_hr - self._profile.resting_hr
        hr_reserve_pct = (point.heart_rate - self._profile.resting_hr) / hr_range

        # Deviation from the user's region-adjusted expected temperature.
        temp_deviation = point.body_temperature - self._profile.baseline_temp

        return np.array(
            [
                point.heart_rate,
                point.spo2,
                point.body_temperature,
                _ACTIVITY_IDX[point.activity],
                point.age,
                _GENDER_IDX[point.gender],
                _REGION_IDX[point.region],
                self._profile.humidity_index,
                hr_reserve_pct,
                rolling_hr_mean,
                rolling_hr_std,
                spo2_drop,
                temp_deviation,
            ],
            dtype=np.float64,
        )

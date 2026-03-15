from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"


class Region(str, Enum):
    NORTH_PLAINS = "north_plains"      # Delhi, UP, Punjab, Haryana, Bihar
    COASTAL_SOUTH = "coastal_south"    # Kerala, Tamil Nadu coast, Andhra coast
    ARID_WEST = "arid_west"            # Rajasthan, Gujarat
    HIGHLAND = "highland"              # Himachal, Uttarakhand, J&K, Sikkim
    NORTHEAST = "northeast"            # Assam, West Bengal, Manipur


# Indian-specific resting HR bases. Indian adults average slightly higher than
# global norms due to higher prevalence of heat stress and metabolic conditions.
_RESTING_HR_BASE: dict[Gender, float] = {
    Gender.MALE: 70.0,
    Gender.FEMALE: 76.0,
}

# Ambient heat offset on body temperature by region.
# Persistent heat-humidity raises the thermoregulatory set-point slightly.
_REGION_TEMP_OFFSET: dict[Region, float] = {
    Region.NORTH_PLAINS: 0.15,    # Hot summers, cold winters; averaged impact
    Region.COASTAL_SOUTH: 0.25,   # Consistently hot-humid year-round
    Region.ARID_WEST: 0.20,       # Very hot but dry; high heat load
    Region.HIGHLAND: -0.15,       # Cooler ambient; sub-alpine zones
    Region.NORTHEAST: 0.20,       # Tropical and persistently humid
}

# SpO2 baseline offset. Altitude in highland zones (avg 1,500-3,000m) lowers
# partial pressure of oxygen, reducing the equilibrium SpO2 reading slightly.
_REGION_SPO2_OFFSET: dict[Region, float] = {
    Region.NORTH_PLAINS: 0.0,
    Region.COASTAL_SOUTH: 0.0,
    Region.ARID_WEST: 0.0,
    Region.HIGHLAND: -0.8,
    Region.NORTHEAST: 0.0,
}

# Chronic heat stress in hot regions adds a small persistent resting HR offset.
_REGION_HR_OFFSET: dict[Region, float] = {
    Region.NORTH_PLAINS: 1.0,
    Region.COASTAL_SOUTH: 2.0,
    Region.ARID_WEST: 1.5,
    Region.HIGHLAND: 0.5,
    Region.NORTHEAST: 1.5,
}

# Relative humidity index (0.0–1.0) by region; used as a direct feature in
# the model to capture heat-humidity interaction on perceived exertion.
REGION_HUMIDITY: dict[Region, float] = {
    Region.NORTH_PLAINS: 0.50,
    Region.COASTAL_SOUTH: 0.80,
    Region.ARID_WEST: 0.20,
    Region.HIGHLAND: 0.40,
    Region.NORTHEAST: 0.85,
}


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    name: str
    age: int
    gender: Gender
    region: Region
    resting_hr: float
    max_hr: float
    baseline_spo2: float
    baseline_temp: float
    humidity_index: float

    @classmethod
    def from_defaults(
        cls,
        user_id: str,
        name: str,
        age: int,
        gender: Gender,
        region: Region,
    ) -> UserProfile:
        # Resting HR: age-adjusted, gender-adjusted, and region heat-stress adjusted.
        resting_hr = (
            _RESTING_HR_BASE[gender]
            + max(0.0, (age - 30) * 0.1)
            + _REGION_HR_OFFSET[region]
        )
        # Haskell formula; widely validated on South Asian populations.
        max_hr = 220.0 - age
        baseline_spo2 = 98.0 + _REGION_SPO2_OFFSET[region]
        baseline_temp = 36.8 + _REGION_TEMP_OFFSET[region]
        return cls(
            user_id=user_id,
            name=name,
            age=age,
            gender=gender,
            region=region,
            resting_hr=round(resting_hr, 1),
            max_hr=float(max_hr),
            baseline_spo2=round(baseline_spo2, 1),
            baseline_temp=round(baseline_temp, 2),
            humidity_index=REGION_HUMIDITY[region],
        )

    def describe(self) -> str:
        return (
            f"{self.name} "
            f"({self.gender.value}, age={self.age}, region={self.region.value})"
        )

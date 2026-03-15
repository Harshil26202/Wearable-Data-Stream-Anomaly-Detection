from __future__ import annotations

from detection.detector import AnomalyResult
from generator.data_generator import ActivityLevel, DataPoint
from models.user_profile import UserProfile


class InsightEngine:
    """Translates model SHAP drivers and raw physiological context into
    human-readable, actionable clinical insights.
    """

    @staticmethod
    def generate_insight(point: DataPoint, profile: UserProfile, result: AnomalyResult) -> str:
        if not result.is_anomaly or not result.top_contributors:
            return "All vitals are within your normal baseline."

        # Extract the names of the top 3 contributing features
        drivers = [name for name, _ in result.top_contributors]
        primary = drivers[0]
        secondary = drivers[1] if len(drivers) > 1 else ""

        act = point.activity
        temp = point.body_temperature
        hr = point.heart_rate
        spo2 = point.spo2

        # 1. Heat Stroke / Hyperthermia (Very specific to Indian summer context)
        if "body_temperature" in drivers or "temp_deviation" in drivers:
            if temp > 39.0:
                if act == ActivityLevel.REST and hr > 110:
                    return (
                        "CRITICAL: High fever and rapid heart rate detected while resting. "
                        "Given your region's climate, this may indicate severe heat stress or infection. "
                        "Cool down immediately, hydrate, and seek medical attention."
                    )
                return "Elevated body temperature detected. Please monitor for fever and stay hydrated."

        # 2. Hypoxia / Low Blood Oxygen
        if "spo2" in drivers or "spo2_drop" in drivers:
            if spo2 < 92.0:
                base_msg = "Significant drop in blood oxygen detected."
                if act in (ActivityLevel.RUN, ActivityLevel.WALK):
                    return f"{base_msg} Please stop your current activity, sit down, and take deep breaths."
                return f"{base_msg} While resting, this is unusual. If you feel short of breath, seek medical advice."

        # 3. Tachycardia (High HR when it shouldn't be)
        if "heart_rate" in drivers or "hr_reserve_pct" in drivers:
            if hr > 110 and act == ActivityLevel.REST:
                return (
                    "Unusually high heart rate detected while resting. "
                    "This could be caused by dehydration, stress, or caffeine. "
                    "Please sit quietly and drink water."
                )
            
            # 4. Bradycardia (Low HR)
            if hr < 50 and act == ActivityLevel.REST:
                return "Unusually low heart rate detected. If you feel lightheaded or dizzy, please sit down."

        # 5. Composite / Stress
        if ("heart_rate" in drivers or "hr_reserve_pct" in drivers) and ("spo2" in drivers or "spo2_drop" in drivers):
            return "Combined physiological stress detected (elevated heart rate and dipping oxygen). Please pause and rest."

        # 6. Activity Mismatches & Erratic HR
        if "rolling_hr_mean" in drivers or "activity_encoded" in drivers:
            if act in (ActivityLevel.WALK, ActivityLevel.RUN):
                return "Your heart rate is responding unusually to your current activity level. Pace yourself."
            return "Erratic heart rate pattern detected. Take a moment to relax and breathe steadily."

        # Fallback for complex multivariate anomalies
        return (
            "Unusual physiological pattern detected. "
            "Your body is behaving slightly differently than your normal baseline."
        )

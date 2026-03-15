from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from config import ANOMALY_RATE, LOG_FILE, OUTPUT_DIR, PLOT_OUTPUT, STREAM_DELAY_RANGE, WARMUP_SIZE
from detection.detector import AnomalyDetector
from generator.data_generator import HealthDataGenerator
from models.user_profile import Gender, Region, UserProfile
from notification.notifier import AsyncNotifier
from processing.feature_engine import FeatureEngine
from stream.streamer import stream

PROFILES: list[UserProfile] = [
    UserProfile.from_defaults("u001", "Aarav", age=34, gender=Gender.MALE, region=Region.NORTH_PLAINS),
    UserProfile.from_defaults("u002", "Priya", age=52, gender=Gender.FEMALE, region=Region.COASTAL_SOUTH),
    UserProfile.from_defaults("u003", "Rohan", age=22, gender=Gender.MALE, region=Region.HIGHLAND),
]


async def run_pipeline(
    profile: UserProfile,
    notifier: AsyncNotifier,
    shutdown: asyncio.Event,
) -> None:
    generator = HealthDataGenerator(profile, anomaly_rate=ANOMALY_RATE)
    engine = FeatureEngine(profile)
    detector = AnomalyDetector()

    print(f"[START] Pipeline initialised for {profile.describe()}")

    model_was_ready = False

    async for point in stream(generator, warmup_size=WARMUP_SIZE, delay_range=STREAM_DELAY_RANGE):
        if shutdown.is_set():
            break

        feature_vector = engine.transform(point)
        result = detector.update(feature_vector)

        if result is None:
            current, total = detector.warmup_progress
            await notifier.log_warmup_progress(profile, current, total)
            continue
            
        if not model_was_ready:
            await notifier.log_model_ready(profile)
            model_was_ready = True

        if result.is_anomaly:
            await notifier.log_anomaly(point, profile, result)
        else:
            await notifier.log_normal(point, profile)


async def main() -> None:
    print("=======================================================")
    print(" Wearable Data Stream Anomaly Detection System         ")
    print("=======================================================")
    print(f" Output log: {LOG_FILE}")
    print(f" Summary chart: {PLOT_OUTPUT}")
    print(" Press Ctrl+C at any time to stop and generate chart.")
    print("-------------------------------------------------------\n")
    
    shutdown = asyncio.Event()

    def _handle_sigint(*_: object) -> None:
        print("\nShutdown signal received. Flushing and generating report...")
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _handle_sigint)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notifier = AsyncNotifier(LOG_FILE)

    tasks = [
        asyncio.create_task(run_pipeline(p, notifier, shutdown))
        for p in PROFILES
    ]

    await shutdown.wait()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    from visualization.plotter import plot_report
    plot_report(LOG_FILE, PLOT_OUTPUT)


if __name__ == "__main__":
    asyncio.run(main())

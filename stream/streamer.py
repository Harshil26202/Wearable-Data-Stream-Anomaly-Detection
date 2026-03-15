from __future__ import annotations

import asyncio
import itertools
import random
from collections.abc import AsyncIterator

from generator.data_generator import DataPoint, HealthDataGenerator


async def stream(
    generator: HealthDataGenerator,
    warmup_size: int,
    delay_range: tuple[float, float] = (1.0, 2.0),
) -> AsyncIterator[DataPoint]:
    # Pull exactly enough clean points for the detector's warmup phase
    warmup_iter = itertools.islice(generator.warmup_stream(), warmup_size)
    # Then indefinitely yield from the stream with anomaly injection
    main_iter = generator.stream()

    for point in itertools.chain(warmup_iter, main_iter):
        delay = random.uniform(*delay_range)
        await asyncio.sleep(delay)
        yield point

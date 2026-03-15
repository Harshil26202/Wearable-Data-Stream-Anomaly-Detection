from pathlib import Path

# Anomaly detector
WARMUP_SIZE: int = 150
CONTAMINATION: float = 0.06
SHAP_TOP_N: int = 3

# Stream
STREAM_DELAY_RANGE: tuple[float, float] = (1.0, 2.0)
ANOMALY_RATE: float = 0.08

# Feature engine
ROLLING_WINDOW: int = 30

# Output
OUTPUT_DIR: Path = Path("output")
LOG_FILE: Path = OUTPUT_DIR / "stream.log"
PLOT_OUTPUT: Path = OUTPUT_DIR / "system_report.png"

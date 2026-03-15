# Wearable Data Stream Anomaly Detection

A Python 3.12 system that simulates a continuous wearable health data stream, applies
context-aware ML-based anomaly detection, and logs real-time structured alerts.

---

## System Architecture

```
main.py
  └── asyncio.gather()  ─── run_pipeline() × N users (concurrent)
                               │
                               ├── HealthDataGenerator   (generator/)
                               │     Markov chain activity transitions
                               │     Gaussian noise around personalised baselines
                               │     Five anomaly injection types
                               │
                               ├── async stream()        (stream/)
                               │     Async generator with configurable delay
                               │
                               ├── FeatureEngine         (processing/)
                               │     13-dimensional feature vector per observation
                               │     Rolling statistics, HR reserve %, SpO2 drop, temp deviation
                               │
                               ├── AnomalyDetector       (detection/)
                               │     IsolationForest trained on warm-up observations
                               │     SHAP TreeExplainer attributes anomaly score to features
                               │
                               ├── InsightEngine         (processing/)
                               │     Translates SHAP drivers + context into human-readable clinical warnings
                               │
                               └── AsyncNotifier         (notification/)
                                     Console + file logging with model-derived driver strings & insights
```

---

## Data Generation

Each observation is produced by `HealthDataGenerator` for a given `UserProfile`.

**UserProfile** captures demographic and environmental context:

| Field | Effect on baseline |
|---|---|
| `age` | Resting HR shifts by `(age - 30) * 0.1`; max HR = `220 - age` |
| `gender` | Male baseline resting HR 70 bpm, female 76 bpm (Indian norms) |
| `region` | `north_plains`, `coastal_south`, `arid_west`, `highland`, `northeast` |
| `region_temp_offset` | +0.15°C to +0.25°C in hotter regions, -0.15°C in sub-alpine highlands |
| `humidity_index` | 0.20 (arid_west) to 0.85 (northeast) directly fed to the model |

**Activity** transitions follow a Markov chain (`rest → walk → run → rest`). Each state
shifts the HR and SpO2 baselines before Gaussian noise is added, ensuring the model sees
realistic physiological variation rather than fixed averages.

**Injected anomaly types:**

| Type | Description |
|---|---|
| `hr_spike_at_rest` | HR 135–165 bpm while resting |
| `spo2_drop` | SpO2 84–91% |
| `fever` | Body temperature 38.6–40.2°C |
| `bradycardia` | HR 28–44 bpm |
| `combined_stress` | Elevated HR + depressed SpO2 simultaneously |
| `heat_stroke` | Indian summer context: HR spike + high fever while sedentary |

---

## Anomaly Detection

### Model Rationale: Why Isolation Forest?
We used an **Isolation Forest** rather than simple thresholds or a supervised classifier because:
1. **Unsupervised:** In real-world health streams, we rarely have perfectly labelled anomaly data. Isolation Forests learn what "normal" looks like and isolate outliers.
2. **Context-Aware:** By feeding it 13 dimensions (not just HR/SpO2, but age, region, activity), the model learns complex multivariate relationships (e.g., *a high HR is normal during a run, but anomalous while resting in a hot climate*).
3. **Explainable:** Isolation Forests are tree-based, making them natively compatible with `SHAP TreeExplainer` for exact, high-performance feature attribution.

### Train / Test Split & Input Dimensions
Because this is a streaming architecture, we use an **Online Setup** rather than a traditional static split:
- **Training (Warm-up phase):** The first **150** observations per user are strictly normal (no anomalies injected). This forms a purely clean `(150, 13)` training matrix. The model is fitted on this baseline.
- **Testing (Inference phase):** Every observation from point 151 onwards is streamed live. The model evaluates each incoming `(1, 13)` vector against its fitted trees, returning a continuous anomaly score.

**CRITICAL DATA QUALITY FIX**: The `generator.warmup_stream()` explicitly segregates a purely normal dataset for the warm-up phase. If test anomalies leak into the training buffer, the model learns them as "normal," degrading inference accuracy.
No hand-coded threshold rules are applied for detection. The fitted model produces a continuous
anomaly score for each incoming observation. A SHAP `TreeExplainer` then decomposes that
score into per-feature contributions. The top-3 features by absolute SHAP magnitude are
reported as the alert's driver string.

Finally, the `InsightEngine` takes those SHAP drivers, aligns them with the user's current context (activity level, environmental heat), and synthesizes a **plain-English clinical insight**.

**Example**: a heart rate of 110 bpm during a run produces near-zero SHAP contribution
from `heart_rate`. The same value during rest produces a high positive SHAP for `heart_rate`, which the `InsightEngine` translates to: *"Unusually high heart rate detected while resting. This could be caused by dehydration, stress, or caffeine."*

### Feature vector (13 dimensions)

| Feature | Description |
|---|---|
| `heart_rate` | Raw HR in bpm |
| `spo2` | Blood oxygen saturation % |
| `body_temperature` | Body temperature in °C |
| `activity_encoded` | Ordinal: 0=rest, 1=walk, 2=run |
| `age` | User age |
| `gender_encoded` | 0=male, 1=female |
| `region_encoded` | 0=north_plains, 1=coastal_south, 2=arid_west, 3=highland, 4=northeast |
| `humidity_index` | 0.20 to 0.85 depending on region |
| `hr_reserve_pct` | `(HR - resting_HR) / (max_HR - resting_HR)` |
| `rolling_hr_mean` | 30-sample rolling mean of HR |
| `rolling_hr_std` | 30-sample rolling std of HR |
| `spo2_drop` | Rolling SpO2 baseline minus current SpO2 |
| `temp_deviation` | Current temperature minus region-adjusted baseline |

---

## Assumptions

- Physiological baselines follow published norms (Haskell max HR formula; AHA resting HR ranges).
- The warm-up phase is assumed to be anomaly-free at the configured `anomaly_rate` (8%).
  In production this phase would be seeded with labelled clean data.
- Region encodes ambient environmental pressure on vitals; it does not model altitude.
- The Markov activity chain approximates typical daily activity distributions.

---

## How to Run

**Requirements:** Python 3.12, pip.

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

Press `Ctrl+C` to stop. On shutdown the system:
1. Flushes all pending log writes.
2. Generates `output/system_report.png` — a per-user, per-metric time-series chart
   with detected anomalies marked.

**To regenerate the report from an existing log without re-running the stream:**

```bash
python - <<'EOF'
from pathlib import Path
from visualization.plotter import plot_report
plot_report(Path("output/stream.log"), Path("output/system_report.png"))
EOF
```

---

## Output

### 2. Check the Outputs

The pipeline streams formatted alerts to `stderr` and logs them systematically in `example_output/stream.log`.

Upon termination (e.g., via `Ctrl+C`), a comprehensive visual report is generated at `example_output/system_report.png`.

### Sample log entries

```text
[DATA]  2026-03-15T12:50:16Z | Aarav (M, age=34, region=north_plains) | HR=95.3 bpm | SpO2=98.3% | Temp=37.01C | Activity=walk
[DATA]  2026-03-15T12:50:16Z | Priya (F, age=52, region=coastal_south) | HR=85.9 bpm | SpO2=97.8% | Temp=37.15C | Activity=rest
[DATA]  2026-03-15T12:50:16Z | Rohan (M, age=22, region=highland) | HR=68.0 bpm | SpO2=97.7% | Temp=36.71C | Activity=rest
[ALERT] 2026-03-15T12:50:16Z | Aarav (M, age=34, region=north_plains) | HR=165.0 bpm | SpO2=98.1% | Temp=37.10C | Activity=rest | Score=-0.62332 | Drivers: hr_reserve_pct(-1.387), heart_rate(-1.313), spo2_drop(-0.573) | Insight: Unusually high heart rate detected while resting. This could be caused by dehydration, stress, or caffeine. Please sit quietly and drink water.
[ALERT] 2026-03-15T12:50:16Z | Rohan (M, age=22, region=highland) | HR=140.2 bpm | SpO2=92.2% | Temp=39.61C | Activity=rest | Score=-0.65734 | Drivers: heart_rate(-0.973), hr_reserve_pct(-0.837), body_temperature(-0.794) | Insight: CRITICAL: High fever and rapid heart rate detected while resting. Given your region's climate, this may indicate severe heat stress or infection. Cool down immediately, hydrate, and seek medical attention.
```

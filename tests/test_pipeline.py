import asyncio
import os
import sys

# Ensure the root directory is in sys.path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from detection.detector import AnomalyDetector, AnomalyResult
from generator.data_generator import ActivityLevel, HealthDataGenerator
from models.user_profile import Gender, Region, UserProfile
from processing.feature_engine import FeatureEngine
from processing.insight_engine import InsightEngine


@pytest.fixture
def sample_profile() -> UserProfile:
    return UserProfile.from_defaults(
        user_id="test1",
        name="TestUser",
        age=30,
        gender=Gender.MALE,
        region=Region.COASTAL_SOUTH,
    )


def test_user_profile_baselines(sample_profile: UserProfile):
    """Test that Indian demographic baselines calculate correctly."""
    # Age 30 Male, Coastal South (+0.15 temp offset)
    assert sample_profile.resting_hr > 60.0
    assert sample_profile.max_hr == 190.0  # 220 - 30
    assert sample_profile.baseline_temp > 36.5
    assert sample_profile.baseline_spo2 >= 97.0
    assert sample_profile.humidity_index == 0.80  # COASTAL_SOUTH


def test_generator_warmup_stream(sample_profile: UserProfile):
    """Ensure warmup stream yields clean points regardless of anomaly rate."""
    generator = HealthDataGenerator(sample_profile, anomaly_rate=1.0) # Force 100% anomalies
    
    stream_gen = generator.warmup_stream()
    for _ in range(50):
        point = next(stream_gen)
        # It must ignore the 100% anomaly rate and yield clean data
        assert point.injected_anomaly is None
        # Physiological sanity checks
        assert 40 <= point.heart_rate <= 190
        assert 92 <= point.spo2 <= 100


def test_feature_engine_dimensions(sample_profile: UserProfile):
    """Test the FeatureEngine outputs the exact expected 13-dimensional vector."""
    engine = FeatureEngine(sample_profile)
    generator = HealthDataGenerator(sample_profile)
    
    # Generate one standard point
    point = next(generator.stream())
    fv = engine.transform(point)
    
    assert isinstance(fv, np.ndarray)
    assert fv.shape == (13,)
    
    # Test caching (rolling means don't explode)
    for _ in range(50):
        p = next(generator.stream())
        fv = engine.transform(p)
    assert fv.shape == (13,)


def test_anomaly_detector_integration(sample_profile: UserProfile):
    """Test that the IsolationForest fits properly on clean data, then flags outliers."""
    detector = AnomalyDetector(warmup_size=50, top_n=3)
    engine = FeatureEngine(sample_profile)
    generator = HealthDataGenerator(sample_profile, anomaly_rate=0.0)
    
    # 1. Warmup phase
    for _ in range(50):
        point = next(generator.stream())
        fv = engine.transform(point)
        res = detector.update(fv)
        assert res is None  # Still warming up
        
    assert detector.is_ready is True
    
    # 2. Live normal phase
    point = next(generator.stream())
    fv = engine.transform(point)
    res = detector.update(fv)
    assert res is not None
    assert len(res.top_contributors) == 3
    
    # 3. Inject explicit anomaly (Heat stroke)
    generator._current_activity = ActivityLevel.REST
    point = generator._inject_anomaly(generator._normal_point(), force_type="heat_stroke")
    fv = engine.transform(point)
    res = detector.update(fv)
    
    assert res is not None
    # Model should flag this as an outlier given it was trained on clean data
    assert res.is_anomaly is True


def test_insight_engine_translation(sample_profile: UserProfile):
    """Test that the InsightEngine converts SHAP targets accurately."""
    detector = AnomalyDetector(warmup_size=50)
    engine = FeatureEngine(sample_profile)
    generator = HealthDataGenerator(sample_profile, anomaly_rate=0.0)
    
    # Fit the detector
    for _ in range(50):
        detector.update(engine.transform(next(generator.stream())))
        
    # Generate Heat Stroke
    generator._current_activity = ActivityLevel.REST
    anomaly = generator._inject_anomaly(generator._normal_point(), force_type="heat_stroke")
    
    # Mock the anomaly result so the test is deterministic and independent of SHAP calculation
    res = AnomalyResult(
        is_anomaly=True,
        score=-0.8,
        top_contributors=[("body_temperature", 0.6), ("heart_rate", 0.3)]
    )

    insight = InsightEngine.generate_insight(anomaly, sample_profile, res)
    assert "CRITICAL" in insight
    assert "fever" in insight.lower()

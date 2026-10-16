"""
Twin State Engine
=================

Reusable module for the SHARE Digital Twin prototype.

Provides:
- Patient-specific static profile lookup
- Median-based physiological baseline calculation
- Per-window robust deviation (MAD-based z-scores)
- Overall Twin State Deviation (engineering measure)
- State classification (Near Baseline / Moderate / High)
- 24-hour trajectory extraction

Methodology matches Experiments 03 and 08 exactly.

IMPORTANT:
    All deviations and state categories are engineering
    representations for visualization. They are NOT
    clinical severity scores or diagnoses.
"""

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

HRV_FEATURES = [
    "HR_bpm",
    "Mean_RR_ms",
    "SDNN_ms",
    "RMSSD_ms",
    "pNN50_percent",
]

HRV_LABELS = {
    "HR_bpm": "Heart Rate (bpm)",
    "Mean_RR_ms": "Mean RR Interval (ms)",
    "SDNN_ms": "SDNN (ms)",
    "RMSSD_ms": "RMSSD (ms)",
    "pNN50_percent": "pNN50 (%)",
}

STATIC_FIELDS = [
    "sex",
    "age",
    "weight_kg",
    "height_cm",
    "BSA_m2",
    "BMI",
    "smoker",
    "SBP_mmHg",
    "DBP_mmHg",
    "IMT_MAX_mm",
    "LVMi_g_m2",
    "EF_percent",
]

STATIC_LABELS = {
    "sex": "Sex",
    "age": "Age (years)",
    "weight_kg": "Weight (kg)",
    "height_cm": "Height (cm)",
    "BSA_m2": "BSA (m²)",
    "BMI": "BMI (kg/m²)",
    "smoker": "Smoker",
    "SBP_mmHg": "Systolic BP (mmHg)",
    "DBP_mmHg": "Diastolic BP (mmHg)",
    "IMT_MAX_mm": "IMT Max (mm)",
    "LVMi_g_m2": "LVMi (g/m²)",
    "EF_percent": "Ejection Fraction (%)",
}

STATIC_UNITS = {
    "sex": "",
    "age": "years",
    "weight_kg": "kg",
    "height_cm": "cm",
    "BSA_m2": "m²",
    "BMI": "kg/m²",
    "smoker": "",
    "SBP_mmHg": "mmHg",
    "DBP_mmHg": "mmHg",
    "IMT_MAX_mm": "mm",
    "LVMi_g_m2": "g/m²",
    "EF_percent": "%",
}

# Engineering deviation thresholds (NOT clinical thresholds)
DEVIATION_THRESHOLDS = {
    "near_baseline": 1.0,
    "moderate": 2.0,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_cohort_states(filepath):
    """Load the full cohort 15-minute twin state data."""
    df = pd.read_csv(filepath, dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(5)
    return df


def load_static_cohort(filepath):
    """Load the static clinical cohort data."""
    df = pd.read_csv(filepath, dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(5)
    return df


def load_patient_level(filepath):
    """Load the patient-level dataset with dynamic summary stats."""
    df = pd.read_csv(filepath, dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(5)
    return df


def get_patient_ids(static_df):
    """Return sorted list of patient IDs."""
    return sorted(static_df["patient_id"].unique().tolist())


# ============================================================
# STATIC PROFILE
# ============================================================

def get_static_profile(static_df, patient_id):
    """
    Extract the static clinical profile for a patient.

    Returns a dict of field -> value.
    """
    patient_id = str(patient_id).zfill(5)
    row = static_df[static_df["patient_id"] == patient_id]

    if row.empty:
        return None

    row = row.iloc[0]
    profile = {}

    for field in STATIC_FIELDS:
        if field in row.index:
            val = row[field]
            if pd.isna(val):
                profile[field] = None
            else:
                profile[field] = val
        else:
            profile[field] = None

    # Include vascular event outcome if available
    if "vascular_event_raw" in row.index:
        profile["vascular_event_raw"] = row["vascular_event_raw"]
    if "vascular_event" in row.index:
        profile["vascular_event"] = int(row["vascular_event"])

    return profile


# ============================================================
# BASELINE CALCULATION
# ============================================================

def compute_baseline(patient_states):
    """
    Compute patient-specific median baseline.

    Uses median (robust to outlier windows).
    Matches Experiment 03 methodology.

    Parameters
    ----------
    patient_states : DataFrame
        15-minute HRV state windows for one patient.

    Returns
    -------
    dict
        Feature -> baseline (median) value.
    """
    baseline = {}
    for feature in HRV_FEATURES:
        if feature in patient_states.columns:
            baseline[feature] = patient_states[feature].median()
        else:
            baseline[feature] = np.nan
    return baseline


def compute_mad_scale(patient_states):
    """
    Compute MAD (Median Absolute Deviation) scale per feature.

    MAD is multiplied by 1.4826 to approximate standard deviation
    under normality. This matches Experiment 03/08 methodology.

    Parameters
    ----------
    patient_states : DataFrame
        15-minute HRV state windows for one patient.

    Returns
    -------
    dict
        Feature -> robust scale value.
    """
    scales = {}
    for feature in HRV_FEATURES:
        if feature in patient_states.columns:
            series = patient_states[feature].dropna()
            median = series.median()
            mad = np.median(np.abs(series - median))
            robust_sd = 1.4826 * mad

            # Avoid zero scale
            if robust_sd == 0 or np.isnan(robust_sd):
                robust_sd = series.std()
                if robust_sd == 0 or np.isnan(robust_sd):
                    robust_sd = 1.0

            scales[feature] = robust_sd
        else:
            scales[feature] = 1.0
    return scales


# ============================================================
# DEVIATION CALCULATION
# ============================================================

def compute_deviations(patient_states, baseline, scales):
    """
    Compute per-window deviations from baseline.

    For each HRV feature and each window, computes:
    - absolute deviation
    - robust z-score

    Then computes overall Twin State Deviation as mean
    of absolute robust z-scores across features.

    Matches Experiment 03 methodology.

    Parameters
    ----------
    patient_states : DataFrame
        15-minute windows (will be copied, not modified).
    baseline : dict
        Feature -> baseline value.
    scales : dict
        Feature -> robust scale value.

    Returns
    -------
    DataFrame
        Enhanced with deviation columns and state classification.
    """
    df = patient_states.copy()

    z_columns = []

    for feature in HRV_FEATURES:
        if feature not in df.columns:
            continue

        # Absolute deviation
        df[feature + "_deviation"] = (
            df[feature] - baseline[feature]
        )

        # Robust z-score
        z_col = feature + "_robust_z"
        df[z_col] = (
            df[feature] - baseline[feature]
        ) / scales[feature]

        z_columns.append(z_col)

    # Overall Twin State Deviation
    if z_columns:
        df["Twin_State_Deviation"] = (
            df[z_columns].abs().mean(axis=1)
        )
    else:
        df["Twin_State_Deviation"] = 0.0

    # Classify state
    df["State_Deviation_Level"] = df["Twin_State_Deviation"].apply(
        classify_deviation
    )

    return df


def classify_deviation(score):
    """
    Classify Twin State Deviation into engineering categories.

    These are visualization labels, NOT clinical severity.

    Categories:
        Near Baseline:      deviation < 1.0
        Moderate Deviation:  1.0 <= deviation < 2.0
        High Deviation:      deviation >= 2.0
    """
    if score < DEVIATION_THRESHOLDS["near_baseline"]:
        return "Near Baseline"
    elif score < DEVIATION_THRESHOLDS["moderate"]:
        return "Moderate Deviation"
    else:
        return "High Deviation"


# ============================================================
# TWIN STATE SUMMARY
# ============================================================

def get_twin_state_summary(enhanced_states, baseline):
    """
    Build per-window state summary for display.

    Returns a list of dicts, one per window.
    """
    summaries = []

    for _, row in enhanced_states.iterrows():
        window = {
            "window_id": int(row.get("window_id", 0)),
            "start_hour": round(float(row.get("start_hour", 0)), 2),
            "end_hour": round(float(row.get("end_hour", 0)), 2),
            "deviation_score": round(
                float(row.get("Twin_State_Deviation", 0)), 3
            ),
            "deviation_level": row.get(
                "State_Deviation_Level", "Unknown"
            ),
            "features": {},
        }

        for feature in HRV_FEATURES:
            current = row.get(feature, np.nan)
            base_val = baseline.get(feature, np.nan)
            deviation = row.get(feature + "_deviation", np.nan)
            robust_z = row.get(feature + "_robust_z", np.nan)

            window["features"][feature] = {
                "current": _safe_round(current, 2),
                "baseline": _safe_round(base_val, 2),
                "deviation": _safe_round(deviation, 2),
                "robust_z": _safe_round(robust_z, 3),
            }

        summaries.append(window)

    return summaries


# ============================================================
# TRAJECTORY EXTRACTION
# ============================================================

def extract_trajectory(enhanced_states):
    """
    Extract trajectory arrays for charting.

    Returns a dict with:
        time_hours: list of floats
        For each HRV feature:
            {feature}_values: list of floats
            {feature}_baseline: float
        deviation_values: list of floats
        deviation_levels: list of strings
    """
    df = enhanced_states.sort_values("window_id").reset_index(drop=True)

    trajectory = {
        "time_hours": df["start_hour"].round(2).tolist(),
        "deviation_values": df["Twin_State_Deviation"].round(3).tolist(),
        "deviation_levels": df["State_Deviation_Level"].tolist(),
    }

    for feature in HRV_FEATURES:
        trajectory[feature + "_values"] = (
            df[feature].round(3).tolist()
        )
        # Baseline is a single value per patient
        if feature + "_deviation" in df.columns:
            # Reconstruct baseline from current - deviation
            first_current = df[feature].iloc[0]
            first_dev = df[feature + "_deviation"].iloc[0]
            trajectory[feature + "_baseline"] = round(
                float(first_current - first_dev), 3
            )

    return trajectory


# ============================================================
# FULL PATIENT TWIN STATE
# ============================================================

def build_patient_twin(
    cohort_states_df,
    static_df,
    patient_id
):
    """
    Build the complete Digital Twin state for a patient.

    Parameters
    ----------
    cohort_states_df : DataFrame
        Full cohort 15-minute states (share_cohort_twin_states.csv).
    static_df : DataFrame
        Static clinical data (share_static_cohort.csv or patient-level).
    patient_id : str
        Five-digit patient ID (e.g. "01911").

    Returns
    -------
    dict with keys:
        patient_id : str
        static_profile : dict or None
        baseline : dict
        scales : dict
        num_windows : int
        recording_hours : float
        enhanced_states : DataFrame
        current_state : dict (last window summary)
        trajectory : dict
        deviation_distribution : dict
    """
    patient_id = str(patient_id).zfill(5)

    # Get patient states
    patient_states = cohort_states_df[
        cohort_states_df["patient_id"] == patient_id
    ].copy()

    if patient_states.empty:
        return None

    # Sort chronologically
    if "window_id" in patient_states.columns:
        patient_states = patient_states.sort_values("window_id")

    # Static profile
    static_profile = get_static_profile(static_df, patient_id)

    # Baseline
    baseline = compute_baseline(patient_states)

    # MAD scale
    scales = compute_mad_scale(patient_states)

    # Deviations
    enhanced = compute_deviations(patient_states, baseline, scales)

    # State summaries
    state_summaries = get_twin_state_summary(enhanced, baseline)

    # Current (latest) state
    current_state = state_summaries[-1] if state_summaries else None

    # Trajectory
    trajectory = extract_trajectory(enhanced)

    # Deviation distribution
    dist = enhanced["State_Deviation_Level"].value_counts().to_dict()
    total = len(enhanced)
    deviation_distribution = {
        "Near Baseline": dist.get("Near Baseline", 0),
        "Moderate Deviation": dist.get("Moderate Deviation", 0),
        "High Deviation": dist.get("High Deviation", 0),
        "total_windows": total,
    }

    # Recording duration
    max_hour = enhanced["end_hour"].max() if "end_hour" in enhanced.columns else 0

    return {
        "patient_id": patient_id,
        "static_profile": static_profile,
        "baseline": {
            k: _safe_round(v, 3) for k, v in baseline.items()
        },
        "scales": {
            k: _safe_round(v, 3) for k, v in scales.items()
        },
        "num_windows": total,
        "recording_hours": round(float(max_hour), 2),
        "enhanced_states": enhanced,
        "state_summaries": state_summaries,
        "current_state": current_state,
        "trajectory": trajectory,
        "deviation_distribution": deviation_distribution,
    }


# ============================================================
# HELPERS
# ============================================================

def _safe_round(val, decimals=3):
    """Safely round a value, handling NaN and None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return val

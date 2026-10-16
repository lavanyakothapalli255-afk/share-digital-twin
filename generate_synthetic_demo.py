"""
Generate Synthetic Demo Datasets
=================================

Standalone script that creates the two CSV files required by
the Digital Twin Dashboard when running in --demo mode.

Output
------
    demo_data/synthetic_cohort_twin_states.csv
        288 rows (96 x 15-min windows x 3 synthetic patients).
        Columns match share_cohort_twin_states.csv exactly.

    demo_data/synthetic_patient_level_dataset.csv
        3 rows - one per synthetic patient.
        Columns match share_patient_level_dataset.csv exactly
        (static profile + 42 dynamic HRV summary features).

    demo_data/demo_feature_importance.json
        Top-15 feature importances copied from the real
        experiment_11_model/feature_importance.csv (if available)
        or a static documented fallback derived from published
        Experiment 11 results.

Safety Guarantees
-----------------
- Fixed numpy random seeds ensure byte-for-byte reproducible output.
- Patient IDs are "DEMO-001", "DEMO-002", "DEMO-003" - clearly
  fictional, zero overlap with PhysioNet SHARE five-digit IDs.
- vascular_event_raw is "None (Synthetic)", vascular_event is 0
  for all patients.
- The script validates actual engine deviation distributions before
  writing any output and will abort if the distributions fall outside
  the expected ranges.  Adjust expected ranges here if seeds change.
- No real patient data is read or written.

IMPORTANT
---------
This is a research prototype demonstration only.
Synthetic values are not based on any real individual.
Do not present demo outputs as clinical risk estimates.

Usage
-----
    python generate_synthetic_demo.py

Run this once before launching the dashboard in --demo mode.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_state_engine as engine  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(SCRIPT_DIR, "demo_data")
STATES_OUT = os.path.join(DEMO_DIR, "synthetic_cohort_twin_states.csv")
PATIENT_OUT = os.path.join(DEMO_DIR, "synthetic_patient_level_dataset.csv")
IMPORTANCE_OUT = os.path.join(DEMO_DIR, "demo_feature_importance.json")
REAL_IMPORTANCE = os.path.join(
    SCRIPT_DIR, "experiment_11_model", "feature_importance.csv"
)

EXPECTED_BOUNDS = {
    "DEMO-001": {
        "Near Baseline_min": 85,
        "High Deviation_max": 3,
        "mean_dev_max": 0.90,
    },
    "DEMO-002": {
        "High Deviation_min": 12,
        "mean_dev_min": 0.70,
    },
    "DEMO-003": {
        "High Deviation_min": 8,
        "mean_dev_min": 0.60,
    },
}

N_WINDOWS = 96


def build_states(patient_id, hr, sdnn, rmssd, pnn50, t_hours):
    n = len(t_hours)
    idx = np.arange(n)
    mean_rr = 60_000.0 / hr
    return pd.DataFrame({
        "patient_id":    patient_id,
        "window_id":     idx,
        "start_minute":  t_hours * 60,
        "end_minute":    t_hours * 60 + 15,
        "start_hour":    t_hours,
        "end_hour":      t_hours + 0.25,
        "HR_bpm":        hr,
        "Mean_RR_ms":    mean_rr,
        "SDNN_ms":       sdnn,
        "RMSSD_ms":      rmssd,
        "pNN50_percent": pnn50,
        "RR_count":      np.full(n, 450),
    })


def compute_dynamic_summary(df):
    row = {
        "num_valid_windows": len(df),
        "recording_hours":   round(float(df["end_hour"].max()), 4),
    }
    for feat in engine.HRV_FEATURES:
        s = df[feat]
        row[f"{feat}_mean"]   = round(float(s.mean()),    4)
        row[f"{feat}_std"]    = round(float(s.std()),     4)
        row[f"{feat}_min"]    = round(float(s.min()),     4)
        row[f"{feat}_max"]    = round(float(s.max()),     4)
        row[f"{feat}_median"] = round(float(s.median()),  4)
        row[f"{feat}_first"]  = round(float(s.iloc[0]),  4)
        row[f"{feat}_last"]   = round(float(s.iloc[-1]), 4)
        row[f"{feat}_change"] = round(float(s.iloc[-1] - s.iloc[0]), 4)
    return row


def validate_distributions(pid, enhanced):
    dist = enhanced["State_Deviation_Level"].value_counts().to_dict()
    mean_dev = float(enhanced["Twin_State_Deviation"].mean())
    bounds = EXPECTED_BOUNDS.get(pid, {})
    errors = []
    nb = dist.get("Near Baseline", 0)
    hd = dist.get("High Deviation", 0)
    md = dist.get("Moderate Deviation", 0)
    if "Near Baseline_min" in bounds and nb < bounds["Near Baseline_min"]:
        errors.append(f"Near Baseline windows ({nb}) below minimum ({bounds['Near Baseline_min']})")
    if "High Deviation_max" in bounds and hd > bounds["High Deviation_max"]:
        errors.append(f"High Deviation windows ({hd}) above maximum ({bounds['High Deviation_max']})")
    if "High Deviation_min" in bounds and hd < bounds["High Deviation_min"]:
        errors.append(f"High Deviation windows ({hd}) below minimum ({bounds['High Deviation_min']})")
    if "mean_dev_max" in bounds and mean_dev > bounds["mean_dev_max"]:
        errors.append(f"Mean deviation ({mean_dev:.3f}) above maximum ({bounds['mean_dev_max']})")
    if "mean_dev_min" in bounds and mean_dev < bounds["mean_dev_min"]:
        errors.append(f"Mean deviation ({mean_dev:.3f}) below minimum ({bounds['mean_dev_min']})")
    if errors:
        msg = (f"\n[ABORT] {pid} engine validation failed:\n"
               + "\n".join(f"  - {e}" for e in errors)
               + "\nAdjust EXPECTED_BOUNDS in generate_synthetic_demo.py if seeds changed.")
        print(msg); sys.exit(1)
    print(f"  {pid}: Near={nb} Moderate={md} High={hd} "
          f"mean_dev={mean_dev:.3f} max_dev={enhanced['Twin_State_Deviation'].max():.3f}  OK")


STATIC_PROFILES = {
    "DEMO-001": {
        "sex": "F", "age": 52.0, "weight_kg": 63.0, "height_cm": 166.0,
        "BSA_m2": 1.72, "BMI": 22.9, "smoker": "no",
        "SBP_mmHg": 122.0, "DBP_mmHg": 78.0,
        "IMT_MAX_mm": 0.68, "LVMi_g_m2": 78.0, "EF_percent": 64.0,
        "vascular_event_raw": "None (Synthetic)", "vascular_event": 0,
    },
    "DEMO-002": {
        "sex": "M", "age": 68.0, "weight_kg": 91.0, "height_cm": 175.0,
        "BSA_m2": 2.09, "BMI": 29.7, "smoker": "yes",
        "SBP_mmHg": 154.0, "DBP_mmHg": 96.0,
        "IMT_MAX_mm": 1.22, "LVMi_g_m2": 128.0, "EF_percent": 50.0,
        "vascular_event_raw": "None (Synthetic)", "vascular_event": 0,
    },
    "DEMO-003": {
        "sex": "M", "age": 61.0, "weight_kg": 80.0, "height_cm": 174.0,
        "BSA_m2": 1.97, "BMI": 26.4, "smoker": "no",
        "SBP_mmHg": 138.0, "DBP_mmHg": 86.0,
        "IMT_MAX_mm": 0.88, "LVMi_g_m2": 95.0, "EF_percent": 58.0,
        "vascular_event_raw": "None (Synthetic)", "vascular_event": 0,
    },
}

DOCUMENTED_IMPORTANCE_FALLBACK = [
    {"feature": "SDNN_ms_first",        "importance": 0.0698},
    {"feature": "pNN50_percent_min",    "importance": 0.0517},
    {"feature": "pNN50_percent_change", "importance": 0.0469},
    {"feature": "pNN50_percent_max",    "importance": 0.0468},
    {"feature": "pNN50_percent_first",  "importance": 0.0407},
    {"feature": "HR_bpm_std",           "importance": 0.0392},
    {"feature": "Mean_RR_ms_change",    "importance": 0.0355},
    {"feature": "RMSSD_ms_min",         "importance": 0.0353},
    {"feature": "RMSSD_ms_last",        "importance": 0.0351},
    {"feature": "HR_bpm_max",           "importance": 0.0296},
    {"feature": "Mean_RR_ms_min",       "importance": 0.0279},
    {"feature": "pNN50_percent_last",   "importance": 0.0275},
    {"feature": "SDNN_ms_std",          "importance": 0.0256},
    {"feature": "HR_bpm_change",        "importance": 0.0226},
    {"feature": "SDNN_ms_last",         "importance": 0.0225},
]


def write_feature_importance():
    if os.path.exists(REAL_IMPORTANCE):
        df = pd.read_csv(REAL_IMPORTANCE)
        top15 = df.head(15)[["feature", "importance"]].copy()
        top15["importance"] = top15["importance"].round(4)
        data = top15.to_dict(orient="records")
        source = "real experiment_11_model/feature_importance.csv"
    else:
        data = DOCUMENTED_IMPORTANCE_FALLBACK
        source = "static documented fallback"
    with open(IMPORTANCE_OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  demo_feature_importance.json: {len(data)} features (source: {source})")


def main():
    print("=" * 62)
    print("SHARE Digital Twin - Synthetic Demo Data Generator")
    print("=" * 62)
    os.makedirs(DEMO_DIR, exist_ok=True)
    t   = np.linspace(0, 24, N_WINDOWS, endpoint=False)
    idx = np.arange(N_WINDOWS)

    def demo001(t):
        np.random.seed(42); n=len(t); circ=2*np.pi*(t-6)/24
        hr  = 65 + 6*np.sin(circ)+np.random.normal(0,1.2,n)
        sd  = np.clip(52-8*np.sin(circ)+np.random.normal(0,2.0,n),20,100)
        rm  = np.clip(42-7*np.sin(circ)+np.random.normal(0,2.0,n),12,90)
        pn  = np.clip(22-8*np.sin(circ)+np.random.normal(0,1.5,n),0,60)
        return hr, sd, rm, pn

    def demo002(t, idx):
        np.random.seed(99); n=len(t); circ=2*np.pi*(t-6)/24
        st = (idx>=36)&(idx<=55)
        hr = 88+5*np.sin(circ)+np.where(st,20,0)+np.random.normal(0,1.5,n)
        sd = np.clip(20-3*np.sin(circ)+np.where(st,-5,0)+np.random.normal(0,1.5,n),5,60)
        rm = np.clip(14-2*np.sin(circ)+np.where(st,-4,0)+np.random.normal(0,1.0,n),3,40)
        pn = np.clip(2.5-1*np.sin(circ)+np.where(st,-1,0)+np.random.normal(0,0.3,n),0,15)
        return hr, sd, rm, pn

    def demo003(t, idx):
        np.random.seed(77); n=len(t); circ=2*np.pi*(t-6)/24
        mid=(idx>=36)&(idx<=60)
        hr = 72+7*np.sin(circ)+np.random.normal(0,1.2,n)
        sd = np.clip(40-6*np.sin(circ)+np.random.normal(0,1.5,n),8,90)
        rm = np.clip(32-5*np.sin(circ)+np.random.normal(0,1.5,n),5,70)
        pn = np.clip(14-5*np.sin(circ)+np.random.normal(0,0.8,n),0,40)
        hr[mid]+=14; sd[mid]-=10; rm[mid]-=10; pn[mid]-=6
        sd=np.clip(sd,5,90); rm=np.clip(rm,3,70); pn=np.clip(pn,0,40)
        return hr, sd, rm, pn

    specs = [
        ("DEMO-001", demo001(t)),
        ("DEMO-002", demo002(t, idx)),
        ("DEMO-003", demo003(t, idx)),
    ]

    print("\nBuilding and validating synthetic trajectories...")
    all_states = []; patient_rows = []
    for pid, (hr, sd, rm, pn) in specs:
        df_s = build_states(pid, hr, sd, rm, pn, t)
        b = engine.compute_baseline(df_s)
        sc = engine.compute_mad_scale(df_s)
        en = engine.compute_deviations(df_s, b, sc)
        validate_distributions(pid, en)
        all_states.append(df_s)
        row = {"patient_id": pid}
        row.update(STATIC_PROFILES[pid])
        row.update(compute_dynamic_summary(df_s))
        patient_rows.append(row)

    print("\nWriting output files...")
    cohort_df = pd.concat(all_states, ignore_index=True)
    cohort_df.to_csv(STATES_OUT, index=False)
    print(f"  synthetic_cohort_twin_states.csv: {len(cohort_df)} rows")

    static_cols = [
        "patient_id","sex","age","weight_kg","height_cm",
        "BSA_m2","BMI","smoker","SBP_mmHg","DBP_mmHg",
        "IMT_MAX_mm","LVMi_g_m2","EF_percent",
        "vascular_event_raw","vascular_event",
        "num_valid_windows","recording_hours",
    ]
    hrv_cols = [f"{feat}_{stat}"
                for feat in engine.HRV_FEATURES
                for stat in ["mean","std","min","max","median","first","last","change"]]
    all_cols = static_cols + hrv_cols
    patient_df = pd.DataFrame(patient_rows)
    patient_df = patient_df[[c for c in all_cols if c in patient_df.columns]]
    patient_df.to_csv(PATIENT_OUT, index=False)
    print(f"  synthetic_patient_level_dataset.csv: {len(patient_df)} rows, {len(patient_df.columns)} cols")
    write_feature_importance()

    print("\nDone. Run demo with:  python experiment_11_dashboard.py --demo")
    print("IMPORTANT: Synthetic data only. Not for clinical use.")


if __name__ == "__main__":
    main()

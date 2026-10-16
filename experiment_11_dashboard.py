
"""
Experiment 11 — Digital Twin Dashboard
=======================================

Flask-based clinician-facing prototype dashboard for the
SHARE Digital Twin.

Displays:
- Patient selector
- Static clinical profile
- Current Twin State with baseline comparison
- 24-hour physiological trajectory
- Exploratory model output with disclaimer
- Model explanation (feature importance)
- What-if simulation placeholder

IMPORTANT:
    This is a research prototype only.
    All model outputs are exploratory.
    Not intended for clinical decision-making.

Usage:
    python experiment_11_dashboard.py          # synthetic demo (default, no real data needed)
    python experiment_11_dashboard.py --demo   # synthetic demo (explicit flag)
    python experiment_11_dashboard.py --real   # real research data (requires SHARE CSVs)

    Then open http://localhost:5000 in a browser.

Operational Modes
-----------------
    By default (without arguments), the dashboard launches in synthetic demo mode
    using demo_data/synthetic_cohort_twin_states.csv and
    demo_data/synthetic_patient_level_dataset.csv (fictional DEMO-* patients).
    Real patient data is never loaded by default.

    To enable real-data mode for authorized local research, pass --real or --real-data.
    Real and synthetic datasets are NEVER mixed.
    If --real is supplied and real data files are absent, the server refuses to start
    with a clear error message.
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, jsonify, request

import twin_state_engine as engine
import what_if_simulator as simulator

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------------
# Operational Mode Configuration
# ------------------------------------------------------------------
# By default, the dashboard launches in synthetic demo mode using
# fictional patient data (DEMO-001) for safe public evaluation without
# exposing real clinical data.
# Real-data mode for authorized local research requires an explicit,
# deliberate flag: --real or --real-data (or SHARE_REAL_DATA=1 env var).
# ------------------------------------------------------------------
REAL_MODE: bool = (
    ("--real" in sys.argv or "--real-data" in sys.argv or os.environ.get("SHARE_REAL_DATA") == "1")
    and ("--demo" not in sys.argv)
)
DEMO_MODE: bool = not REAL_MODE

DATA_DIR = "."
DEMO_DIR = "demo_data"
MODEL_DIR = "experiment_11_model"
STATIC_MODEL_DIR = "experiment_13_model"

if DEMO_MODE:
    COHORT_STATES_FILE = os.path.join(DEMO_DIR, "synthetic_cohort_twin_states.csv")
    STATIC_FILE        = os.path.join(DEMO_DIR, "synthetic_patient_level_dataset.csv")
else:
    COHORT_STATES_FILE = os.path.join(DATA_DIR, "share_cohort_twin_states.csv")
    STATIC_FILE        = os.path.join(DATA_DIR, "share_patient_level_dataset.csv")

MODEL_FILE = os.path.join(
    MODEL_DIR, "exploratory_rf_pipeline.joblib"
)
FEATURES_FILE = os.path.join(
    MODEL_DIR, "dynamic_features.json"
)
META_FILE = os.path.join(
    MODEL_DIR, "model_metadata.json"
)
IMPORTANCE_FILE = os.path.join(
    MODEL_DIR, "feature_importance.csv"
)
DEMO_IMPORTANCE_FILE = os.path.join(
    DEMO_DIR, "demo_feature_importance.json"
)

# Optional Experiment 13 Static Model Artifacts (Dual-Stream Mode)
STATIC_MODEL_FILE = os.path.join(
    STATIC_MODEL_DIR, "static_rf_pipeline.joblib"
)
STATIC_FEATURES_FILE = os.path.join(
    STATIC_MODEL_DIR, "static_features.json"
)
STATIC_META_FILE = os.path.join(
    STATIC_MODEL_DIR, "static_model_metadata.json"
)

# Fixed illustrative mock scores for demo mode when no model artifacts
# are available.  These are constants, NOT derived from any trained model.
# The values are intentionally arbitrary and clearly labeled.
_DEMO_MOCK_DYN  = {"DEMO-001": 0.115, "DEMO-002": 0.485, "DEMO-003": 0.260}
_DEMO_MOCK_STAT = {"DEMO-001": 0.100, "DEMO-002": 0.410, "DEMO-003": 0.190}

DEFAULT_PATIENT = "DEMO-001" if DEMO_MODE else "01911"


# ============================================================
# LOAD DATA
# ============================================================

if DEMO_MODE:
    print("[DEMO MODE] Default public demo mode active (fictional synthetic patient data).")
    print(f"  Default patient: {DEFAULT_PATIENT}")
    print("  To enable real-data mode for local research: python experiment_11_dashboard.py --real")
    missing_demo = [f for f in (COHORT_STATES_FILE, STATIC_FILE) if not os.path.exists(f)]
    if missing_demo:
        print("\n[ERROR] Required synthetic demo data file(s) not found:")
        for m in missing_demo:
            print(f"  {m}")
        print("\nGenerate synthetic demo files with: python generate_synthetic_demo.py\n")
        sys.exit(1)
else:
    print("\n" + "=" * 60)
    print("[WARNING] REAL RESEARCH DATA MODE ACTIVE")
    print("Loading PhysioNet SHARE dataset records.")
    print("This mode processes real patient data and is intended exclusively")
    print("for authorized local research under the PhysioNet DUA.")
    print("Not for clinical diagnostic, treatment, or interventional use.")
    print("=" * 60 + "\n")

    missing = [f for f in (COHORT_STATES_FILE, STATIC_FILE)
               if not os.path.exists(f)]
    if missing:
        print("\n[ERROR] Required real data file(s) not found:")
        for m in missing:
            print(f"  {m}")
        print(
            "\nTo run the dashboard in real-data mode you need the SHARE dataset CSV files.\n"
            "If you want to explore the dashboard using synthetic demo data (no real data needed),\n"
            "run:  python experiment_11_dashboard.py\n"
            "or:   python experiment_11_dashboard.py --demo\n"
        )
        sys.exit(1)

cohort_states = engine.load_cohort_states(COHORT_STATES_FILE)
static_data = engine.load_patient_level(STATIC_FILE)
patient_ids = engine.get_patient_ids(static_data)

print(f"  Cohort states: {len(cohort_states)} rows")
print(f"  Patients: {len(patient_ids)}")
if DEMO_MODE:
    print("  [DEMO] Synthetic data loaded. Real patient data not used.")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading exploratory model...")

model_pipeline = None
model_features = None
model_metadata = None
feature_importance = None

if os.path.exists(MODEL_FILE):
    model_pipeline = joblib.load(MODEL_FILE)
    print(f"  Model loaded: {MODEL_FILE}")
else:
    print(f"  WARNING: Model not found at {MODEL_FILE}")
    print("  Run experiment_11_train_model.py first.")

if os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE, "r") as f:
        model_features = json.load(f)
    print(f"  Features: {len(model_features)}")

if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        model_metadata = json.load(f)

if os.path.exists(IMPORTANCE_FILE):
    feature_importance = pd.read_csv(IMPORTANCE_FILE)

# Optional Static Model for Dual-Stream Mode (Experiment 13)
static_model_pipeline = None
static_model_features = None
static_model_metadata = None

if os.path.exists(STATIC_MODEL_FILE):
    static_model_pipeline = joblib.load(STATIC_MODEL_FILE)
    print(f"  Static model loaded: {STATIC_MODEL_FILE}")
else:
    print(f"  Note: Static model not found at {STATIC_MODEL_FILE} (Dual-stream mode disabled)")

if os.path.exists(STATIC_FEATURES_FILE):
    with open(STATIC_FEATURES_FILE, "r") as f:
        static_model_features = json.load(f)

if os.path.exists(STATIC_META_FILE):
    with open(STATIC_META_FILE, "r") as f:
        static_model_metadata = json.load(f)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def get_static_prediction(patient_id):
    """
    Generate exploratory static model-derived probability (Experiment 13).
    Uses the trained Random Forest pipeline with 12 static clinical features.

    In demo mode with no model artifacts, returns an illustrative mock score
    clearly labeled as a placeholder (not a trained model output).
    """
    # --- Demo mode: trained model absent — return illustrative mock score ---
    if DEMO_MODE and (static_model_pipeline is None or static_model_features is None):
        mock_prob = _DEMO_MOCK_STAT.get(str(patient_id), 0.200)
        return {
            "probability": round(mock_prob, 4),
            "is_synthetic": True,
            "is_illustrative": True,
            "model_status": "Demo Mode — Illustrative Placeholder Score (no trained model)",
            "model_type": "Mock Score (Static stream placeholder)",
            "disclaimer": (
                "DEMO ONLY: Fictional patient data. This is a fixed illustrative "
                "placeholder score, NOT generated by a trained statistical model. "
                "Do not interpret as clinical risk or validated prediction."
            ),
        }

    if static_model_pipeline is None or static_model_features is None:
        return {
            "probability": None,
            "model_status": "Static model not available",
            "disclaimer": "Static model artifact not found in experiment_13_model/.",
        }

    patient_id = str(patient_id).zfill(5)
    patient_row = static_data[static_data["patient_id"] == patient_id]

    if patient_row.empty:
        return {
            "probability": None,
            "model_status": "Patient not found",
            "disclaimer": "",
        }

    all_cols = static_model_features.get("all_static_features", [])
    cat_cols = static_model_features.get("categorical_features", [])

    row_data = {}
    for col in all_cols:
        if col in patient_row.columns:
            val = patient_row[col].iloc[0]
            if col in cat_cols:
                row_data[col] = str(val).strip().upper() if col == "sex" else str(val).strip().lower()
            else:
                row_data[col] = pd.to_numeric(val, errors="coerce")
        else:
            row_data[col] = np.nan

    X_pred = pd.DataFrame([row_data])[all_cols]

    try:
        probability = float(static_model_pipeline.predict_proba(X_pred)[0, 1])
    except Exception as e:
        return {
            "probability": None,
            "model_status": f"Static prediction error: {str(e)}",
            "disclaimer": "",
        }

    res = {
        "probability": round(probability, 4),
        "model_status": "Exploratory static model active",
        "model_type": "Random Forest (12 static clinical features)",
        "oof_roc_auc": (
            static_model_metadata.get("oof_roc_auc_static_only", None)
            if static_model_metadata else None
        ),
        "disclaimer": (
            "Research prototype only. Static clinical features carry low standalone "
            "discrimination (OOF ROC-AUC 0.526). Not for clinical diagnosis or treatment."
        ),
    }

    if DEMO_MODE:
        res["is_synthetic"] = True
        res["is_illustrative"] = False
        res["model_status"] = "Exploratory static model active (trained model, synthetic demo input)"
        res["disclaimer"] = (
            "DEMO ONLY: Score produced by the trained static model running on "
            "purely fictional synthetic patient data. Not for clinical use."
        )

    return res


def get_dual_stream_prediction(patient_id):
    """
    Generate exploratory late-fusion dual-stream prediction (Experiment 13).
    Combines 80% dynamic model probability and 20% static model probability.

    In demo mode:
    - If both trained pipelines are present: runs real inference on synthetic
      inputs, labels result as synthetic-demo.
    - If pipelines are absent: uses fixed illustrative mock component scores
      and applies the exact 80/20 formula, labeled as illustrative placeholder.
    """
    dyn_res = get_exploratory_prediction(patient_id, mode="dynamic")
    stat_res = get_static_prediction(patient_id)

    if dyn_res.get("probability") is None or stat_res.get("probability") is None:
        return {
            "probability": None,
            "dynamic_probability": dyn_res.get("probability"),
            "static_probability": stat_res.get("probability"),
            "model_status": "Dual-stream prediction unavailable (component model failed)",
            "disclaimer": "Both dynamic and static component pipelines must be available.",
        }

    p_dyn = dyn_res["probability"]
    p_stat = stat_res["probability"]
    p_blend = round(0.80 * p_dyn + 0.20 * p_stat, 4)

    # Determine whether components came from trained models or mock scores
    dyn_is_mock  = dyn_res.get("is_illustrative", False)
    stat_is_mock = stat_res.get("is_illustrative", False)
    is_mock_blend = dyn_is_mock or stat_is_mock

    if DEMO_MODE and is_mock_blend:
        model_status = "Demo Mode — Illustrative Placeholder Blend (no trained models)"
        model_type   = "Illustrative Mock 80/20 Blend (static placeholders, not trained models)"
        disclaimer   = (
            "DEMO ONLY: Both component scores are fixed illustrative placeholders. "
            "The 80/20 late-fusion formula is applied correctly, but neither input "
            "comes from a trained model. Do not interpret as clinical risk or validated prediction."
        )
    elif DEMO_MODE:
        model_status = "Exploratory dual-stream active (trained models, synthetic demo input)"
        model_type   = "Late Fusion (80% Dynamic + 20% Static Exploratory Blend) — Synthetic Input"
        disclaimer   = (
            "DEMO ONLY: Trained models running on purely fictional synthetic patient data. "
            "Not for clinical use. The 80/20 weighting is exploratory and not clinically validated."
        )
    else:
        model_status = "Exploratory dual-stream active"
        model_type   = "Late Fusion (80% Dynamic + 20% Static Exploratory Blend)"
        disclaimer   = (
            "Research prototype only. The 80/20 late-fusion weighting is an exploratory "
            "experimental configuration and has not been clinically validated. "
            "Neither component nor the blend is cleared for diagnosis or treatment decisions. "
            "The SHARE dataset contains 139 patients with only 17 vascular-event cases."
        )

    res = {
        "probability": p_blend,
        "dynamic_probability": p_dyn,
        "static_probability": p_stat,
        "weights": {"dynamic": 0.80, "static": 0.20},
        "weights_validated": False,
        "model_status": model_status,
        "model_type": model_type,
        "oof_roc_auc": 0.6553,
        "disclaimer": disclaimer,
    }

    if DEMO_MODE:
        res["is_synthetic"] = True
        res["is_illustrative"] = is_mock_blend

    return res


def get_exploratory_prediction(patient_id, mode="dynamic"):
    """
    Generate exploratory model-derived probability.

    Parameters
    ----------
    patient_id : str or int
        Patient identifier.
    mode : str
        'dynamic' (default, Experiment 11 dynamic-only RF) or
        'dual_stream' (Experiment 13 late-fusion 80/20 blend).

    Returns dict with probability, labels, and disclaimer.
    In demo mode with no model artifacts, returns an illustrative mock score.
    If demo mode is active and artifacts exist, returns a real inference on
    synthetic inputs, clearly flagged with is_synthetic=True.
    """
    if mode == "dual_stream":
        return get_dual_stream_prediction(patient_id)

    # --- Demo mode: trained model absent — return illustrative mock score ---
    if DEMO_MODE and (model_pipeline is None or model_features is None):
        mock_prob = _DEMO_MOCK_DYN.get(str(patient_id), 0.200)
        return {
            "probability": round(mock_prob, 4),
            "is_synthetic": True,
            "is_illustrative": True,
            "model_status": "Demo Mode — Illustrative Placeholder Score (no trained model)",
            "model_type": "Mock Score (Dynamic stream placeholder)",
            "disclaimer": (
                "DEMO ONLY: Fictional patient data. This is a fixed illustrative "
                "placeholder score, NOT generated by a trained statistical model. "
                "Do not interpret as clinical risk or validated prediction."
            ),
        }

    # Default: Dynamic-Only RF (Experiment 11 benchmark)
    if model_pipeline is None or model_features is None:
        return {
            "probability": None,
            "model_status": "Model not available",
            "disclaimer": (
                "Exploratory model has not been trained. "
                "Run experiment_11_train_model.py first."
            ),
        }

    patient_id = str(patient_id).zfill(5)

    patient_row = static_data[
        static_data["patient_id"] == patient_id
    ]

    if patient_row.empty:
        return {
            "probability": None,
            "model_status": "Patient not found",
            "disclaimer": "",
        }

    # Extract features
    feature_values = {}
    missing_features = []

    for feat in model_features:
        if feat in patient_row.columns:
            val = patient_row[feat].iloc[0]
            feature_values[feat] = (
                pd.to_numeric(val, errors="coerce")
            )
        else:
            feature_values[feat] = np.nan
            missing_features.append(feat)

    X_pred = pd.DataFrame([feature_values])[model_features]

    try:
        probability = float(
            model_pipeline.predict_proba(X_pred)[0, 1]
        )
    except Exception as e:
        return {
            "probability": None,
            "model_status": f"Prediction error: {str(e)}",
            "disclaimer": "",
        }

    res = {
        "probability": round(probability, 4),
        "model_status": "Exploratory model active",
        "model_type": "Random Forest (42 dynamic features)",
        "oof_roc_auc": (
            model_metadata.get("oof_roc_auc", None)
            if model_metadata else None
        ),
        "disclaimer": (
            "Research prototype only. This model has not been "
            "clinically validated and is not intended for "
            "diagnosis or treatment decisions. "
            "The SHARE dataset contains 139 patients with "
            "only 17 vascular-event cases."
        ),
    }

    if DEMO_MODE:
        res["is_synthetic"] = True
        res["is_illustrative"] = False
        res["model_status"] = "Exploratory model active (trained model, synthetic demo input)"
        res["disclaimer"] = (
            "DEMO ONLY: Score produced by the trained dynamic model running on "
            "purely fictional synthetic patient data. Not for clinical use."
        )

    return res


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance_data(top_n=15):
    """Return top feature importances for display.

    In demo mode, falls back to demo_data/demo_feature_importance.json
    when the real feature_importance.csv is unavailable.
    """
    if feature_importance is not None:
        top = feature_importance.head(top_n)
        return [
            {"feature": row["feature"], "importance": round(float(row["importance"]), 4)}
            for _, row in top.iterrows()
        ]

    # Demo-mode fallback: load from pre-built JSON
    if DEMO_MODE and os.path.exists(DEMO_IMPORTANCE_FILE):
        with open(DEMO_IMPORTANCE_FILE, "r") as f:
            data = json.load(f)
        return data[:top_n]

    return []


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


@app.after_request
def set_security_headers(response):
    """
    Attach standard HTTP security headers to all responses.

    Configured headers:
    - X-Content-Type-Options: nosniff (prevent MIME-sniffing)
    - X-Frame-Options: DENY (clickjacking protection)
    - Referrer-Policy: strict-origin-when-cross-origin
    - Content-Security-Policy: Tailored for dashboard dependencies:
        * script-src: 'self', 'unsafe-inline' (inline scripts & onclick handlers),
          https://cdn.jsdelivr.net (Chart.js)
        * style-src: 'self', 'unsafe-inline' (inline CSS & style attributes),
          https://fonts.googleapis.com (Google Fonts CSS)
        * font-src: 'self', https://fonts.gstatic.com (Google Fonts WOFF2), data:
        * img-src: 'self', data: (embedded SVG icons)
        * connect-src: 'self' (local AJAX /api/patient and /api/simulate calls)
        * frame-ancestors: 'none' (aligns with X-Frame-Options: DENY)
        * base-uri: 'self'
        * object-src: 'none' (disables plugin execution)
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "object-src 'none';"
    )
    return response


@app.route("/")
def index():
    """Serve the main dashboard page."""
    # Ensure patient IDs remain 5-digit strings for the frontend template
    formatted_pids = [str(pid).zfill(5) for pid in patient_ids]
    return render_template(
        "dashboard.html",
        patient_ids=formatted_pids,
        default_patient=DEFAULT_PATIENT,
    )


@app.route("/api/patient/<patient_id>")
def api_patient(patient_id):
    """
    API endpoint returning full Twin State data for a patient.
    Accepts optional ?mode=dynamic (default) or ?mode=dual_stream.
    """
    patient_id = str(patient_id).zfill(5)
    mode = request.args.get("mode", "dynamic")

    # Build twin state
    twin = engine.build_patient_twin(
        cohort_states, static_data, patient_id
    )

    if twin is None:
        return jsonify({"error": "Patient not found"}), 404

    # Get prediction according to requested mode (default remains dynamic-only)
    prediction = get_exploratory_prediction(patient_id, mode=mode)
    dynamic_prediction = (
        prediction if mode == "dynamic"
        else get_exploratory_prediction(patient_id, mode="dynamic")
    )
    dual_stream_prediction = get_dual_stream_prediction(patient_id)

    # Get feature importance
    importance = get_feature_importance_data(15)

    # Prepare response (exclude DataFrame)
    safe_static_profile = None
    if twin.get("static_profile"):
        safe_static_profile = {
            k: v for k, v in twin["static_profile"].items()
            if k not in ("vascular_event", "vascular_event_raw")
        }

    response = {
        "patient_id": twin["patient_id"],
        # Top-level demo flag — allows the UI to render persistent
        # synthetic-data notices without inspecting prediction sub-objects.
        "is_demo": DEMO_MODE,
        "static_profile": safe_static_profile,
        "baseline": twin["baseline"],
        "num_windows": twin["num_windows"],
        "recording_hours": twin["recording_hours"],
        "current_state": twin["current_state"],
        "trajectory": twin["trajectory"],
        "deviation_distribution": twin["deviation_distribution"],
        "prediction": prediction,
        "dynamic_prediction": dynamic_prediction,
        "dual_stream_prediction": dual_stream_prediction,
        "available_modes": ["dynamic", "dual_stream"],
        "active_mode": mode,
        "feature_importance": importance,
        "static_labels": engine.STATIC_LABELS,
        "static_units": engine.STATIC_UNITS,
        "hrv_labels": engine.HRV_LABELS,
    }

    return jsonify(response)


@app.route("/api/simulate/<patient_id>", methods=["POST"])
def api_simulate(patient_id):
    """
    API endpoint for What-If Twin Simulation (Experiment 12).
    """
    patient_id = str(patient_id).zfill(5)

    data = request.json or {}
    parameters = data.get("parameters", {})

    twin = engine.build_patient_twin(
        cohort_states, static_data, patient_id
    )

    if twin is None:
        return jsonify({"error": "Patient not found"}), 404

    baseline = twin["baseline"]
    scales = twin["scales"]

    observed_state = data.get("observed_state")

    if not observed_state:
        if not twin.get("current_state"):
            return jsonify({
                "error": "No observed states available for patient"
            }), 400

        current_features = {}

        for feat in engine.HRV_FEATURES:
            if feat in twin["current_state"]["features"]:
                current_features[feat] = (
                    twin["current_state"]["features"][feat]["current"]
                )

        observed_state = current_features

    try:
        sim_result = simulator.simulate_what_if_state(
            patient_id,
            observed_state,
            baseline,
            scales,
            parameters,
        )
        return jsonify(sim_result)

    except Exception:
        # Avoid returning internal exception details to the client.
        app.logger.exception("What-if simulation failed")
        return jsonify({
            "error": "Simulation failed. Check the server logs."
        }), 400


@app.route("/api/patients")
def api_patients():
    """Return list of available patient IDs."""
    return jsonify({
        "patient_ids": [
            str(pid).zfill(5) for pid in patient_ids
        ]
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXPERIMENT 11 - DIGITAL TWIN DASHBOARD")
    print("=" * 60)
    if DEMO_MODE:
        print("Mode: SYNTHETIC DEMO (Default)")
        print("Fictional synthetic data · Research prototype · Not for clinical use")
    else:
        print("Mode: REAL RESEARCH DATA (--real)")
        print("[WARNING] Local research use only · Comply with PhysioNet DUA")
    print("\nOpen http://127.0.0.1:5000 in your browser")
    print(f"Default patient: {DEFAULT_PATIENT}")
    print("\nPress Ctrl+C to stop the server.\n")

    # Local-only binding and disabled development debugger.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )

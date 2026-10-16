"""
What-If Digital Twin Simulator (Experiment 12)
==============================================

Simulates mathematical perturbations to physiological states to
demonstrate the Twin State Engine's response.

IMPORTANT:
    This is a counterfactual engineering demonstration only.
    It does not estimate clinical treatment effects.
"""

import copy
import pandas as pd
import numpy as np
import twin_state_engine as engine

def _safe_round(val, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return val

def simulate_what_if_state(patient_id, observed_state, baseline, scales, parameters):
    """
    Applies mathematical transformations to an observed Twin state.

    Parameters
    ----------
    patient_id : str
        The patient ID.
    observed_state : dict
        The current observed state features.
        Expected keys match engine.HRV_FEATURES.
    baseline : dict
        Patient's observed baseline features.
    scales : dict
        Patient's MAD scales.
    parameters : dict
        Percentage changes. e.g., {'HR_bpm': -5.0, 'SDNN_ms': 10.0}

    Returns
    -------
    dict
        Simulated state summary, differences, and model response placeholder.
    """
    if observed_state is None:
        raise ValueError("Observed state cannot be None.")

    simulated_features = copy.deepcopy(observed_state)

    # 1. Validate and Apply transformations
    for feat in ['HR_bpm', 'SDNN_ms', 'RMSSD_ms', 'pNN50_percent']:
        if feat in parameters and feat in simulated_features and simulated_features[feat] is not None:
            pct_change = float(parameters[feat])

            # Validation
            if feat == 'HR_bpm' and (pct_change < -10.0 or pct_change > 10.0):
                raise ValueError(f"Invalid range for {feat}: {pct_change}%. Must be between -10% and +10%.")
            elif feat != 'HR_bpm' and (pct_change < -20.0 or pct_change > 20.0):
                raise ValueError(f"Invalid range for {feat}: {pct_change}%. Must be between -20% and +20%.")

            simulated_features[feat] = simulated_features[feat] * (1.0 + (pct_change / 100.0))

    # Derive Mean RR consistently only if HR_bpm was modified
    if parameters.get('HR_bpm', 0.0) != 0.0 and 'HR_bpm' in simulated_features and simulated_features['HR_bpm'] is not None and simulated_features['HR_bpm'] > 0:
        simulated_features['Mean_RR_ms'] = 60000.0 / simulated_features['HR_bpm']

    # 2. Recalculate Twin State using existing engine logic
    # Wrap in DataFrame for engine compatibility
    df_sim = pd.DataFrame([simulated_features])

    # Add required dummy columns for the engine's summary builder
    df_sim['window_id'] = 0
    df_sim['start_hour'] = 0.0
    df_sim['end_hour'] = 0.0

    df_enhanced = engine.compute_deviations(df_sim, baseline, scales)
    summary = engine.get_twin_state_summary(df_enhanced, baseline)[0]

    # Format the simulated output
    simulated_state = {
        "features": {},
        "Twin_State_Deviation": summary["deviation_score"],
        "State_Deviation_Level": summary["deviation_level"]
    }

    for feat in engine.HRV_FEATURES:
        if feat in summary["features"]:
            simulated_state["features"][feat] = summary["features"][feat]["current"]

    # Calculate differences for display
    differences = {}
    for feat in engine.HRV_FEATURES:
        orig = observed_state.get(feat)
        sim = simulated_state["features"].get(feat)
        if orig is not None and sim is not None:
            differences[feat] = _safe_round(sim - orig, 2)

    # 3. Model integration
    # The 42-feature patient-level model requires full 24h distributions.
    # It cannot be truthfully computed from a single simulated state point.
    model_response = {
        "available": False,
        "message": "Exploratory model response not computed for this simulation because the available data do not support a reliable reconstruction of all 42 patient-level model features."
    }

    return {
        "patient_id": patient_id,
        "parameters": parameters,
        "observed_state": {
            "features": {k: _safe_round(v, 2) for k, v in observed_state.items() if k in engine.HRV_FEATURES},
        },
        "simulated_state": simulated_state,
        "differences": differences,
        "model_response": model_response
    }

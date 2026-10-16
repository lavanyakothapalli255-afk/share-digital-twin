"""
Synthetic Demo Mode Tests
==========================
Verifies that the Digital Twin Dashboard boots correctly in --demo mode
and that all demo-mode safety contracts are met, without requiring real
SHARE dataset files or trained model artifacts.

Tests
-----
  1.  demo_data/ files exist after generation
  2.  CSV schemas match the real data schemas exactly
  3.  Three DEMO-* patients present, no real patient IDs
  4.  96 windows per patient in the states CSV
  5.  Server loads with only demo_data/ files (no real SHARE CSVs)
  6.  /api/patient/<id> returns HTTP 200 for each demo patient
  7.  Top-level is_demo flag is True for every demo patient
  8.  Top-level is_demo flag is absent (or False) for real patients
      (regression test skipped when real data unavailable)
  9.  dynamic_prediction carries is_synthetic and is_illustrative
 10.  dual_stream_prediction carries is_synthetic and is_illustrative
 11.  Disclaimer text contains "DEMO ONLY" for demo patients
 12.  Prediction probability is a float in [0.0, 1.0]
 13.  Dual-stream 80/20 blend formula holds to four decimal places
 14.  What-If simulator returns a valid simulated state for a demo patient
 15.  Active mode defaults to "dynamic"
 16.  /api/patients lists all three demo patient IDs

Research Prototype | Not for Clinical Decision-Making
"""

import sys
import os
import json
import unittest

import importlib

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REAL_DATA_EXISTS = os.path.exists(
    os.path.join(_PROJECT_ROOT, "share_cohort_twin_states.csv")
) and os.path.exists(
    os.path.join(_PROJECT_ROOT, "share_patient_level_dataset.csv")
)

# On a clean clone without real SHARE data, dash requires --demo to boot.
# If real data is present, import normally so discovery does not pollute other test suites.
_ORIGINAL_ARGV = list(sys.argv)
try:
    if not _REAL_DATA_EXISTS and "--demo" not in sys.argv:
        sys.argv.append("--demo")
    import experiment_11_dashboard as dash  # noqa: E402
finally:
    sys.argv[:] = _ORIGINAL_ARGV


def setUpModule():
    """Ensure dash is configured in demo mode before running demo tests."""
    global dash
    if not dash.DEMO_MODE:
        orig = list(sys.argv)
        try:
            sys.argv[:] = [arg for arg in orig if arg not in ("--real", "--real-data")]
            if "--demo" not in sys.argv:
                sys.argv.append("--demo")
            dash = importlib.reload(dash)
        finally:
            sys.argv[:] = orig


def tearDownModule():
    """Restore dash to real mode if real data exists, preventing cross-test pollution."""
    global dash
    if _REAL_DATA_EXISTS:
        orig = list(sys.argv)
        try:
            sys.argv[:] = [arg for arg in orig if arg != "--demo"]
            if "--real" not in sys.argv:
                sys.argv.append("--real")
            dash = importlib.reload(dash)
        finally:
            sys.argv[:] = orig

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
STATES_CSV  = os.path.join(DEMO_DIR, "synthetic_cohort_twin_states.csv")
PATIENT_CSV = os.path.join(DEMO_DIR, "synthetic_patient_level_dataset.csv")
IMPORTANCE_JSON = os.path.join(DEMO_DIR, "demo_feature_importance.json")

DEMO_PATIENTS = ["DEMO-001", "DEMO-002", "DEMO-003"]


class TestDemoFiles(unittest.TestCase):
    """Verify that demo_data/ files were generated correctly."""

    def test_01_states_csv_exists(self):
        self.assertTrue(os.path.exists(STATES_CSV),
                        "Run: python generate_synthetic_demo.py")

    def test_02_patient_csv_exists(self):
        self.assertTrue(os.path.exists(PATIENT_CSV),
                        "Run: python generate_synthetic_demo.py")

    def test_03_importance_json_exists(self):
        self.assertTrue(os.path.exists(IMPORTANCE_JSON),
                        "Run: python generate_synthetic_demo.py")

    def test_04_states_csv_schema(self):
        """Column names match share_cohort_twin_states.csv schema."""
        import pandas as pd
        df = pd.read_csv(STATES_CSV)
        required = [
            "patient_id", "window_id", "start_minute", "end_minute",
            "start_hour", "end_hour", "HR_bpm", "Mean_RR_ms", "SDNN_ms",
            "RMSSD_ms", "pNN50_percent", "RR_count",
        ]
        for col in required:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_05_patient_csv_schema(self):
        """Column names match share_patient_level_dataset.csv schema."""
        import pandas as pd
        df = pd.read_csv(PATIENT_CSV)
        required = [
            "patient_id", "sex", "age", "weight_kg", "height_cm",
            "BSA_m2", "BMI", "smoker", "SBP_mmHg", "DBP_mmHg",
            "IMT_MAX_mm", "LVMi_g_m2", "EF_percent",
            "vascular_event_raw", "vascular_event",
            "num_valid_windows", "recording_hours",
            "HR_bpm_mean", "HR_bpm_std", "SDNN_ms_median",
        ]
        for col in required:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_06_three_demo_patients_only(self):
        """Only DEMO-001/002/003 are present; no five-digit SHARE IDs."""
        import pandas as pd
        df = pd.read_csv(PATIENT_CSV)
        pids = set(df["patient_id"].tolist())
        self.assertEqual(pids, {"DEMO-001", "DEMO-002", "DEMO-003"})
        for pid in pids:
            self.assertFalse(pid.isdigit(), f"Unexpected real-style patient ID: {pid}")

    def test_07_window_count_per_patient(self):
        """Each demo patient has exactly 96 windows."""
        import pandas as pd
        df = pd.read_csv(STATES_CSV)
        for pid in DEMO_PATIENTS:
            count = len(df[df["patient_id"] == pid])
            self.assertEqual(count, 96, f"{pid} has {count} windows (expected 96)")

    def test_08_no_real_patient_data_in_demo_files(self):
        """Demo CSVs must not contain PhysioNet five-digit patient IDs."""
        import pandas as pd
        df = pd.read_csv(STATES_CSV)
        for pid in df["patient_id"].unique():
            self.assertFalse(pid.isdigit(),
                             f"Possible real patient ID '{pid}' found in demo CSV")


class TestDemoAPIResponses(unittest.TestCase):
    """Verify API responses when running in demo mode."""

    @classmethod
    def setUpClass(cls):
        global dash
        if not dash.DEMO_MODE:
            orig = list(sys.argv)
            try:
                sys.argv[:] = [arg for arg in orig if arg not in ("--real", "--real-data")]
                dash = importlib.reload(dash)
            finally:
                sys.argv[:] = orig
        cls.client = dash.app.test_client()

    def _get_patient(self, patient_id, mode="dynamic"):
        url = f"/api/patient/{patient_id}"
        if mode != "dynamic":
            url += f"?mode={mode}"
        with dash.app.app_context():
            resp = self.client.get(url)
        return resp

    def test_09_demo_mode_flag_is_set(self):
        """Dashboard module must be loaded in demo mode."""
        self.assertTrue(dash.DEMO_MODE,
                        "DEMO_MODE not set. Check --demo injection in argv.")

    def test_10_all_demo_patients_return_200(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            self.assertEqual(resp.status_code, 200,
                             f"{pid} returned {resp.status_code}")

    def test_11_is_demo_true_in_top_level_response(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            self.assertIn("is_demo", data, f"{pid}: is_demo missing from response")
            self.assertTrue(data["is_demo"], f"{pid}: is_demo is not True")

    def test_12_dynamic_prediction_is_synthetic_flagged(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            dyn = data.get("dynamic_prediction", {})
            self.assertIn("is_synthetic", dyn, f"{pid}: is_synthetic missing")
            self.assertTrue(dyn["is_synthetic"], f"{pid}: is_synthetic not True")
            self.assertIn("is_illustrative", dyn, f"{pid}: is_illustrative missing")

    def test_13_dual_stream_prediction_is_synthetic_flagged(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            ds = data.get("dual_stream_prediction", {})
            self.assertIn("is_synthetic", ds, f"{pid}: is_synthetic missing in dual_stream")
            self.assertTrue(ds["is_synthetic"], f"{pid}: dual_stream is_synthetic not True")

    def test_14_disclaimer_contains_demo_only(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            pred = data.get("prediction", {})
            disclaimer = pred.get("disclaimer", "")
            self.assertIn("DEMO ONLY", disclaimer,
                          f"{pid}: prediction disclaimer does not contain 'DEMO ONLY'")

    def test_15_prediction_probability_in_range(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            prob = data.get("prediction", {}).get("probability")
            self.assertIsNotNone(prob, f"{pid}: probability is None")
            self.assertGreaterEqual(prob, 0.0, f"{pid}: probability below 0")
            self.assertLessEqual(prob, 1.0, f"{pid}: probability above 1")

    def test_16_dual_stream_blend_formula_exact(self):
        """80/20 fusion formula must hold to 4 decimal places."""
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            ds = data.get("dual_stream_prediction", {})
            if ds.get("probability") is None:
                continue  # skip if both components unavailable
            p_dyn  = ds["dynamic_probability"]
            p_stat = ds["static_probability"]
            expected = round(0.80 * p_dyn + 0.20 * p_stat, 4)
            self.assertAlmostEqual(
                ds["probability"], expected, places=4,
                msg=f"{pid}: dual_stream blend {ds['probability']} != {expected}"
            )

    def test_17_default_mode_is_dynamic(self):
        for pid in DEMO_PATIENTS:
            resp = self._get_patient(pid)
            data = json.loads(resp.data)
            self.assertEqual(data.get("active_mode"), "dynamic",
                             f"{pid}: active_mode != 'dynamic'")

    def test_18_available_modes_contains_both(self):
        resp = self._get_patient("DEMO-001")
        data = json.loads(resp.data)
        modes = data.get("available_modes", [])
        self.assertIn("dynamic", modes)
        self.assertIn("dual_stream", modes)

    def test_19_what_if_simulator_returns_valid_state(self):
        """POST /api/simulate/<id> must return a simulated state."""
        with dash.app.app_context():
            resp = self.client.post(
                "/api/simulate/DEMO-002",
                json={"parameters": {"HR_bpm": -5.0}},
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("simulated_state", data)
        self.assertIn("Twin_State_Deviation", data["simulated_state"])
        self.assertIn("State_Deviation_Level", data["simulated_state"])

    def test_20_api_patients_lists_all_demo_ids(self):
        with dash.app.app_context():
            resp = self.client.get("/api/patients")
        data = json.loads(resp.data)
        ids = data.get("patient_ids", [])
        for pid in DEMO_PATIENTS:
            self.assertIn(pid, ids, f"{pid} not in /api/patients response")

    def test_21_feature_importance_non_empty(self):
        """Feature importance must be available in demo mode."""
        resp = self._get_patient("DEMO-001")
        data = json.loads(resp.data)
        fi = data.get("feature_importance", [])
        self.assertGreater(len(fi), 0, "feature_importance is empty in demo mode")
        self.assertIn("feature", fi[0])
        self.assertIn("importance", fi[0])

    def test_22_vascular_event_not_in_static_profile(self):
        """vascular_event and vascular_event_raw must be stripped from response."""
        resp = self._get_patient("DEMO-001")
        data = json.loads(resp.data)
        profile = data.get("static_profile", {})
        self.assertNotIn("vascular_event", profile)
        self.assertNotIn("vascular_event_raw", profile)

    def test_23_trajectory_has_96_windows(self):
        resp = self._get_patient("DEMO-002")
        data = json.loads(resp.data)
        traj = data.get("trajectory", {})
        self.assertEqual(data.get("num_windows"), 96)
        self.assertEqual(len(traj.get("time_hours", [])), 96)

    def test_23a_security_headers_on_dashboard_html(self):
        """Dashboard HTML route must return all required security headers."""
        with dash.app.app_context():
            resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(resp.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")

        csp = resp.headers.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net", csp)
        self.assertIn("style-src 'self' 'unsafe-inline' https://fonts.googleapis.com", csp)
        self.assertIn("font-src 'self' https://fonts.gstatic.com data:", csp)
        self.assertIn("img-src 'self' data:", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("base-uri 'self'", csp)
        self.assertIn("object-src 'none'", csp)

    def test_23b_security_headers_on_api_endpoints(self):
        """API endpoints must return security headers on JSON responses."""
        with dash.app.app_context():
            endpoints = [
                "/api/patients",
                "/api/patient/DEMO-001",
            ]
            for ep in endpoints:
                resp = self.client.get(ep)
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff", f"Missing on {ep}")
                self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY", f"Missing on {ep}")
                self.assertEqual(resp.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin", f"Missing on {ep}")
                self.assertIn("default-src 'self'", resp.headers.get("Content-Security-Policy", ""), f"Missing on {ep}")

            # POST /api/simulate endpoint
            resp_sim = self.client.post(
                "/api/simulate/DEMO-001",
                json={"parameters": {"HR_bpm": 0}},
                content_type="application/json",
            )
            self.assertEqual(resp_sim.status_code, 200)
            self.assertEqual(resp_sim.headers.get("X-Content-Type-Options"), "nosniff")
            self.assertEqual(resp_sim.headers.get("X-Frame-Options"), "DENY")
            self.assertEqual(resp_sim.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
            self.assertIn("default-src 'self'", resp_sim.headers.get("Content-Security-Policy", ""))


class TestRealDataRegressionGuard(unittest.TestCase):
    """
    Regression guard: confirms that the is_demo flag is False for
    real patients when real SHARE CSVs are present.

    This test class is SKIPPED when real data is unavailable.
    It does NOT import or load real data itself.
    """

    @classmethod
    def setUpClass(cls):
        # Real data files would need to be present and the server
        # started in real mode for these tests.  Since we are running
        # in demo mode here (sys.argv has --demo), we simply skip.
        cls.real_data_available = (
            not dash.DEMO_MODE and
            os.path.exists(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "share_cohort_twin_states.csv"
            ))
        )

    def test_24_real_mode_is_demo_false(self):
        if not self.real_data_available:
            self.skipTest("Real SHARE data not available or demo mode active")
        # This branch runs only in a real-data server context.
        resp = dash.app.test_client().get("/api/patient/01911")
        data = json.loads(resp.data)
        self.assertFalse(data.get("is_demo", True),
                         "is_demo should be False in real-data mode")

    def test_25_real_prediction_not_labeled_illustrative(self):
        if not self.real_data_available:
            self.skipTest("Real SHARE data not available or demo mode active")
        resp = dash.app.test_client().get("/api/patient/01911")
        data = json.loads(resp.data)
        pred = data.get("prediction", {})
        self.assertNotIn("is_illustrative", pred,
                         "Real prediction should not carry is_illustrative flag")


class TestDashboardLaunchModes(unittest.TestCase):
    """
    Focused tests verifying the default launch mode (synthetic demo)
    and explicit real-data mode (--real / --real-data).
    """

    @classmethod
    def tearDownClass(cls):
        global dash
        orig = list(sys.argv)
        try:
            sys.argv[:] = [arg for arg in orig if arg not in ("--real", "--real-data")]
            dash = importlib.reload(dash)
        finally:
            sys.argv[:] = orig

    def tearDown(self):
        global dash
        if not dash.DEMO_MODE:
            orig = list(sys.argv)
            try:
                sys.argv[:] = [arg for arg in orig if arg not in ("--real", "--real-data")]
                dash = importlib.reload(dash)
            finally:
                sys.argv[:] = orig

    def test_26_default_mode_is_demo_without_args(self):
        """Launching without arguments must default to synthetic demo mode."""
        global dash
        orig_argv = list(sys.argv)
        try:
            # Strip any mode flags to simulate bare launch: python experiment_11_dashboard.py
            sys.argv[:] = [arg for arg in orig_argv if arg not in ("--real", "--real-data", "--demo")]
            dash = importlib.reload(dash)
            self.assertTrue(dash.DEMO_MODE, "Default mode without arguments must be DEMO_MODE=True")
            self.assertEqual(dash.DEFAULT_PATIENT, "DEMO-001", "Default patient must be DEMO-001")
            self.assertIn("DEMO-001", dash.patient_ids, "DEMO-001 must be in default patient list")
            self.assertNotIn("01911", dash.patient_ids, "Real patient 01911 must not be in demo patient list")

            # Verify index route uses DEMO-001 as default
            with dash.app.app_context():
                resp = dash.app.test_client().get("/")
                self.assertEqual(resp.status_code, 200)
                self.assertIn(b"DEMO-001", resp.data)

                # Verify API returns is_demo=True for DEMO-001
                api_resp = dash.app.test_client().get("/api/patient/DEMO-001")
                self.assertEqual(api_resp.status_code, 200)
                data = json.loads(api_resp.data)
                self.assertTrue(data.get("is_demo"), "DEMO-001 API response must carry is_demo=True")
        finally:
            sys.argv[:] = orig_argv
            dash = importlib.reload(dash)

    def test_27_explicit_demo_flag(self):
        """Passing --demo explicitly must enable synthetic demo mode."""
        global dash
        orig_argv = list(sys.argv)
        try:
            sys.argv[:] = [arg for arg in orig_argv if arg not in ("--real", "--real-data")]
            if "--demo" not in sys.argv:
                sys.argv.append("--demo")
            dash = importlib.reload(dash)
            self.assertTrue(dash.DEMO_MODE)
            self.assertEqual(dash.DEFAULT_PATIENT, "DEMO-001")
        finally:
            sys.argv[:] = orig_argv
            dash = importlib.reload(dash)

    def test_28_explicit_real_mode_flag(self):
        """Passing --real must enable real-data mode when real SHARE data exists."""
        if not _REAL_DATA_EXISTS:
            self.skipTest("Real SHARE CSV files not present in workspace")

        global dash
        orig_argv = list(sys.argv)
        try:
            sys.argv[:] = [arg for arg in orig_argv if arg != "--demo"]
            if "--real" not in sys.argv:
                sys.argv.append("--real")
            dash = importlib.reload(dash)
            self.assertFalse(dash.DEMO_MODE, "--real must set DEMO_MODE=False")
            self.assertEqual(dash.DEFAULT_PATIENT, "01911", "Real mode default patient must be 01911")
            self.assertIn("01911", dash.patient_ids, "01911 must be in real patient list")

            # Verify index route uses 01911
            with dash.app.app_context():
                resp = dash.app.test_client().get("/")
                self.assertEqual(resp.status_code, 200)
                self.assertIn(b"01911", resp.data)

                # Verify API returns is_demo=False for 01911
                api_resp = dash.app.test_client().get("/api/patient/01911")
                self.assertEqual(api_resp.status_code, 200)
                data = json.loads(api_resp.data)
                self.assertFalse(data.get("is_demo"), "Real patient 01911 API response must carry is_demo=False")
        finally:
            sys.argv[:] = orig_argv
            dash = importlib.reload(dash)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)

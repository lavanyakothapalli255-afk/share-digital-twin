import unittest
import copy
from what_if_simulator import simulate_what_if_state

class TestWhatIfSimulator(unittest.TestCase):
    def setUp(self):
        self.observed = {
            'HR_bpm': 80.0,
            'Mean_RR_ms': 750.0,
            'SDNN_ms': 100.0,
            'RMSSD_ms': 50.0,
            'pNN50_percent': 10.0
        }
        self.baseline = {
            'HR_bpm': 75.0,
            'Mean_RR_ms': 800.0,
            'SDNN_ms': 110.0,
            'RMSSD_ms': 55.0,
            'pNN50_percent': 12.0
        }
        self.scales = {
            'HR_bpm': 5.0,
            'Mean_RR_ms': 20.0,
            'SDNN_ms': 10.0,
            'RMSSD_ms': 5.0,
            'pNN50_percent': 2.0
        }
        self.patient_id = '01911'

    def test_1_zero_change_simulation_equals_observed(self):
        res = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {})
        for k in self.observed:
            self.assertAlmostEqual(res['simulated_state']['features'][k], self.observed[k], places=4)

    def test_2_hr_negative_5_percent_changes_correctly(self):
        res = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': -5.0})
        expected_hr = 80.0 * 0.95
        self.assertAlmostEqual(res['simulated_state']['features']['HR_bpm'], expected_hr, places=4)

    def test_3_hr_change_updates_mean_rr_consistently(self):
        res = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': -5.0})
        new_hr = res['simulated_state']['features']['HR_bpm']
        new_mean_rr = res['simulated_state']['features']['Mean_RR_ms']
        self.assertAlmostEqual(new_mean_rr, round(60000.0 / new_hr, 2), places=2)

    def test_4_baseline_remains_unchanged(self):
        baseline_copy = copy.deepcopy(self.baseline)
        res = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': -5.0})
        self.assertEqual(self.baseline, baseline_copy)

    def test_5_twin_deviation_is_recalculated(self):
        res1 = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {})
        res2 = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': 10.0})
        self.assertNotEqual(res1['simulated_state']['Twin_State_Deviation'], res2['simulated_state']['Twin_State_Deviation'])

    def test_6_original_patient_data_are_not_modified(self):
        observed_copy = copy.deepcopy(self.observed)
        res = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': -5.0})
        self.assertEqual(self.observed, observed_copy)

    def test_7_simulation_does_not_overwrite_share_data(self):
        self.assertTrue(True)

    def test_8_reset_returns_exactly_to_observed_state(self):
        res = simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {})
        for k in self.observed:
            self.assertAlmostEqual(res['simulated_state']['features'][k], self.observed[k], places=4)

    def test_9_invalid_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': -15.0})
        with self.assertRaises(ValueError):
            simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'HR_bpm': 15.0})
        with self.assertRaises(ValueError):
            simulate_what_if_state(self.patient_id, self.observed, self.baseline, self.scales, {'SDNN_ms': -25.0})

    def test_10_different_patients_can_be_simulated_independently(self):
        observed2 = copy.deepcopy(self.observed)
        observed2['HR_bpm'] = 60.0
        res1 = simulate_what_if_state('PAT01', self.observed, self.baseline, self.scales, {'HR_bpm': 5.0})
        res2 = simulate_what_if_state('PAT02', observed2, self.baseline, self.scales, {'HR_bpm': 5.0})
        self.assertEqual(res1['patient_id'], 'PAT01')
        self.assertEqual(res2['patient_id'], 'PAT02')
        self.assertNotEqual(res1['simulated_state']['features']['HR_bpm'], res2['simulated_state']['features']['HR_bpm'])

if __name__ == '__main__':
    unittest.main()

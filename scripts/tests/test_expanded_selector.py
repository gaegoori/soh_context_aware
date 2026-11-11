"""Selection safety properties independent of battery outcomes."""
import unittest
import numpy as np
from scripts.models.expanded_selector_models import gate, select_threshold, THRESHOLDS


class GateTests(unittest.TestCase):
    def test_strict_improvement_and_no_switch_on_ties(self):
        scores=np.array([[.25,.5,.5],[.5,.5,.5],[.5,.5,.25]])
        selected,_=gate(scores,2,.25)
        np.testing.assert_array_equal(selected,[2,2,2])
        selected,_=gate(scores,2,0.)
        np.testing.assert_array_equal(selected,[0,2,2])

    def test_threshold_only_removes_switches_and_never_means_default(self):
        scores=np.random.default_rng(42).normal(size=(100,3))
        previous=set(range(100))
        for threshold in THRESHOLDS:
            selected,_=gate(scores,1,threshold)
            switched=set(np.flatnonzero(selected!=1))
            self.assertTrue(switched.issubset(previous));previous=switched
        self.assertEqual(previous,set())

    def test_negative_scores_retain_order_not_clip_to_false_ties(self):
        selected,gain=gate([[-.01,-.03,.02]],2,.04)
        self.assertEqual(int(selected[0]),1)
        self.assertAlmostEqual(float(gain[0]),.05)
        with self.assertRaises(ValueError):gate([[0,1,2]],0,-.1)

    def test_equal_calibration_risk_prefers_fewer_changes_then_never(self):
        self.assertEqual(select_threshold(np.ones(8),[8,7,6,5,4,3,0,0]),7)
        self.assertEqual(select_threshold([1,1,1,1,1,.5,1,1],[8]*8),5)


if __name__=='__main__':unittest.main()

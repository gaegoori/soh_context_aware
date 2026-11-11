"""Behavioral tests for window semantics, train-only missingness and routing."""
import unittest
from unittest.mock import patch
import numpy as np

from scripts.models.adaptive_models import CycleHistory, context_at
import scripts.models.temporal_context_models as models


def history_fixture():
    rng=np.random.default_rng(81)
    return CycleHistory(rng.normal(size=(80,128,4)).astype(np.float32),
        np.arange(80.)*2+4,np.ones(80),np.arange(80.)+1,
        np.arange(80.)*100,0.,np.zeros(80))


class TemporalContextTest(unittest.TestCase):
    def test_window_semantics_and_c10_reproduction(self):
        h=history_fixture()
        np.testing.assert_array_equal(models.context_at_L(h,60,10),context_at(h,60))
        for length in (10,20,50):
            x=models.context_at_L(h,60,length)
            np.testing.assert_array_equal(x[:6],models.context_at_L(h,60,10)[:6])
            self.assertAlmostEqual(float(x[6]),float(h.duration[60-length:60].mean()),places=5)
            self.assertAlmostEqual(float(x[8]),2.,places=5)
            self.assertAlmostEqual(float(x[9]),float(h.x[60-length:60,:,0].mean()),places=6)
            self.assertAlmostEqual(float(x[10]),float(h.x[60-length:60,:,0].std()),places=6)
        with self.assertRaises(ValueError):models.context_at_L(h,20,50)

    def test_future_does_not_change_inputs(self):
        h=history_fixture();before=[models.context_at_L(h,60,lb) for lb in (10,20,50)]
        for name in ('x','duration','discharge_ah','cumulative_ah','end_time','temperature_missing'):
            getattr(h,name)[60:]=9999
        for x,lb in zip(before,(10,20,50)):
            np.testing.assert_array_equal(x,models.context_at_L(h,60,lb))

    def test_all_missing_training_feature_remains_inactive(self):
        train=np.array([[1.,np.nan,4.],[2.,np.nan,4.],[3.,np.nan,4.]])
        scaler=models.fit_scaler(train)
        a=models.transform(np.array([4.,100.,4.]),scaler)
        b=models.transform(np.array([4.,-100.,4.]),scaler)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a[1],0.);self.assertEqual(scaler['std'][2],1.)
        ridge=models.fit_ridge(train,np.array([.7,.8,.9]))
        self.assertEqual(float(ridge['weight'][1]),0.)

    def test_one_and_two_class_selectors(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(30,15))
        for labels in (np.ones(30,dtype=int),np.arange(30)%2*2):
            classifier,_=models.fit_classifier(x,labels)
            params=dict(mean=np.zeros(15),std=np.ones(15),active=np.ones(15,dtype=bool),**classifier)
            _,prob=models.choose(x[0],params,'Adaptive')
            np.testing.assert_allclose(prob.sum(),1.)
            self.assertTrue(np.all(prob[np.setdiff1d(np.arange(3),np.unique(labels))]==0.))
            if len(np.unique(labels))==1:self.assertEqual(models.choose(x[0],params,'Adaptive')[0],20)

    def test_routes_one_predictor_and_reuses_c10(self):
        h=history_fixture()
        p=dict(mean=np.zeros(15),std=np.ones(15),active=np.ones(15,dtype=bool),
            error_weight=np.zeros((15,3)),error_bias=np.array([0.,1.,2.]))
        model=dict(mean=np.zeros(15),std=np.ones(15),active=np.ones(15,dtype=bool),
                   weight=np.ones(15)*.001,bias=np.asarray(.2))
        accessed=[]
        class Bank(dict):
            def __getitem__(self,key):accessed.append(key);return super().__getitem__(key)
        bank=Bank({10:model})
        with patch.object(models,'window_features',wraps=models.window_features) as f:
            value,lb=models.infer_one(h,60,bank,p,'ErrorAware')
            self.assertEqual(lb,10);self.assertTrue(np.isfinite(value))
            self.assertEqual(f.call_count,1)
        self.assertEqual(accessed,[10])
        with self.assertRaises(ValueError):models.infer_one(h,60,bank,p,'Oracle')


if __name__=='__main__':unittest.main()

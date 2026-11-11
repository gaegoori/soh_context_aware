"""Causality, feature semantics and exported nonlinear inference checks."""
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from scripts.models.adaptive_models import CycleHistory
from scripts.models.change_selector_models import operating_changes,fit_regressor,predict_regressor,choose_length,infer_selected


def history():
    n=25;x=np.ones((n,128,4),dtype=np.float32)
    a=np.ones(n,dtype=float)
    return CycleHistory(x,a.copy(),a.copy(),np.cumsum(a),np.arange(n,dtype=float),0.,np.zeros(n))


class ChangeTests(unittest.TestCase):
    def test_direction_units_and_constant_history(self):
        h=history();np.testing.assert_array_equal(operating_changes(h,20),np.zeros(6))
        h.x[15:20,:,0]+=2;h.x[15:20,:,1]-=.5;h.x[15:20,:,3]+=3
        h.duration[15:20]+=4;h.discharge_ah[15:20]+=.2;h.temperature_missing[15:20]+=.1
        np.testing.assert_allclose(operating_changes(h,20),[2,-.5,3,4,.2,.1])

    def test_no_future_or_older_history_dependency(self):
        h=history();expected=operating_changes(h,20);other=deepcopy(h)
        other.x[:10]=100;other.x[20:]=np.nan
        for name in ('duration','discharge_ah','temperature_missing'):
            a=getattr(other,name);a[:10]=999;a[20:]=999
        np.testing.assert_array_equal(operating_changes(other,20),expected)

    def test_missing_temperature_remains_missing(self):
        h=history();h.x[10:15,:,3]=np.nan
        self.assertTrue(np.isnan(operating_changes(h,20)[2]))
        with self.assertRaises(ValueError):operating_changes(h,9)

    def test_all_missing_training_feature_cannot_affect_inference(self):
        rng=np.random.default_rng(42);raw=rng.normal(size=(40,4));raw[:,3]=np.nan
        target=np.column_stack((raw[:,0]**2,np.abs(raw[:,1]),np.ones(40)))
        for kind in ('Linear','Forest'):
            m=fit_regressor(raw,target,kind);other=raw.copy();other[:,3]=1e12
            np.testing.assert_allclose(predict_regressor(raw,m),predict_regressor(other,m))
            lengths,_=choose_length(raw,m);self.assertTrue(set(lengths).issubset({10,20,50}))

    def test_inference_executes_only_the_chosen_predictor(self):
        selected_model=object()
        with patch('scripts.models.change_selector_models.choose_length',return_value=(np.array([20]),np.array([[2,0,1]]))), \
             patch('scripts.models.change_selector_models.predict_ridge',return_value=.8) as predict:
            value,length=infer_selected(history(),20,{20:selected_model},{},True)
        self.assertEqual((value,length),(.8,20));predict.assert_called_once()
        self.assertIs(predict.call_args.args[1],selected_model)


if __name__=='__main__':unittest.main()

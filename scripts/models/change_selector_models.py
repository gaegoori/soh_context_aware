"""Past-only operating-change features and frozen-setting regression models."""
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scripts.models.temporal_context_models import fit_scaler, transform, context_at_L, predict_ridge
from scripts.models.adaptive_models import fit_context_ridge

CHANGE_NAMES = (
    'recent5_minus_previous5_voltage_mean_V',
    'recent5_minus_previous5_current_mean_A',
    'recent5_minus_previous5_temperature_mean_C',
    'recent5_minus_previous5_duration_mean_s',
    'recent5_minus_previous5_discharge_mean_Ah',
    'recent5_minus_previous5_temperature_missing_fraction',
)
RF_PARAMS = dict(n_estimators=200, criterion='squared_error', max_depth=3,
                 min_samples_split=2, min_samples_leaf=8, max_features=1.0,
                 bootstrap=True, max_samples=None, random_state=42, n_jobs=1)


def operating_changes(history, n):
    n = int(n)
    if not 10 <= n <= len(history.x):
        raise ValueError('Ten completed past cycles are required')
    before, recent = slice(n-10,n-5), slice(n-5,n)
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for ch in (0,1,3):
            values.append(float(np.nanmean(history.x[recent,:,ch]))-
                          float(np.nanmean(history.x[before,:,ch])))
    for a in (history.duration,history.discharge_ah,history.temperature_missing):
        values.append(float(np.mean(a[recent])-np.mean(a[before])))
    return np.asarray(values,dtype=np.float32)


def fit_regressor(raw, targets, kind):
    y = np.asarray(targets,dtype=float)
    if y.ndim==1:y=y[:,None]
    if y.ndim!=2 or len(y)!=len(raw) or not np.isfinite(y).all():
        raise ValueError('Expected finite N x output targets')
    scaler=fit_scaler(raw); x=transform(raw,scaler)
    model=dict(**scaler,kind=np.array(kind))
    if kind=='Linear':
        fits=[fit_context_ridge(x,y[:,j],.1) for j in range(y.shape[1])]
        model.update(weight=np.column_stack([w for w,b in fits]),
                     bias=np.array([b for w,b in fits]))
    elif kind=='Forest':
        forest=RandomForestRegressor(**RF_PARAMS)
        forest.fit(x,y[:,0] if y.shape[1]==1 else y)
        t=len(forest.estimators_); width=max(e.tree_.node_count for e in forest.estimators_)
        model.update(left=np.full((t,width),-1,dtype=np.int64),right=np.full((t,width),-1,dtype=np.int64),
            feature=np.full((t,width),-2,dtype=np.int64),threshold=np.zeros((t,width)),
            value=np.zeros((t,width,y.shape[1])),node_count=np.zeros(t,dtype=np.int64),
            node_samples=np.zeros((t,width),dtype=np.int64),node_weight=np.zeros((t,width)),
            tree_seed=np.zeros(t,dtype=np.int64))
        for i,estimator in enumerate(forest.estimators_):
            tree=estimator.tree_;n=tree.node_count
            model['left'][i,:n]=tree.children_left;model['right'][i,:n]=tree.children_right
            model['feature'][i,:n]=tree.feature;model['threshold'][i,:n]=tree.threshold
            model['value'][i,:n]=tree.value[:,:,0];model['node_count'][i]=n
            model['node_samples'][i,:n]=tree.n_node_samples
            model['node_weight'][i,:n]=tree.weighted_n_node_samples
            model['tree_seed'][i]=estimator.random_state
        expected=forest.predict(x)
        if expected.ndim==1:expected=expected[:,None]
        np.testing.assert_allclose(predict_regressor(raw,model),expected,rtol=1e-12,atol=1e-12)
    else:raise ValueError('Unknown regressor kind')
    return model


def predict_regressor(raw,model):
    x=transform(raw,model)
    if x.ndim!=2:raise ValueError('Expected batch matrix')
    if str(model['kind'])=='Linear':return x@model['weight']+model['bias']
    x=x.astype(np.float32)  # Match sklearn tree traversal precision.
    predictions=np.zeros((len(x),model['value'].shape[2]))
    for t in range(len(model['node_count'])):
        nodes=np.zeros(len(x),dtype=np.int64)
        while True:
            rows=np.flatnonzero(model['left'][t,nodes]!=-1)
            if not len(rows):break
            at=nodes[rows];f=model['feature'][t,at]
            nodes[rows]=np.where(x[rows,f]<=model['threshold'][t,at],
                                 model['left'][t,at],model['right'][t,at])
        predictions+=model['value'][t,nodes]
    return predictions/len(model['node_count'])


def choose_length(raw, model):
    scores=predict_regressor(raw,model)
    if scores.ndim!=2 or scores.shape[1]!=3:raise ValueError('Expected three error estimates')
    return np.array([10,20,50])[np.argmin(scores,axis=1)],scores


def infer_selected(history, n, bank, model, with_changes):
    base=context_at_L(history,n,10)
    selector_input=np.r_[base,operating_changes(history,n)] if with_changes else base
    selected,_=choose_length(selector_input[None,:],model)
    length=int(selected[0])
    predictor_input=base if length==10 else context_at_L(history,n,length)
    return float(predict_ridge(predictor_input,bank[length])),length

"""Rebuild inputs/scalers independently and reproduce every saved prediction."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from battery_data import DEFAULT_ROOT
from context_history_models import BATTERIES, LOOKBACKS, SEEDS, VARIANTS, CONTEXT_GROUPS, HistoryPredictor
from train_context_history import DEFAULT_OUTPUT, neural_folder
from train_lookback_oof import read_csv, write_json, sha256, set_seed, state_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    root, out = args.data_root, args.output
    torch.set_num_threads(4)
    protocol = json.loads((out/'protocol.json').read_text())
    sig = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest = json.loads((out/'artifact_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sig
    for path, digest in protocol['input_sha256'].items():
        assert sha256(Path(path)) == digest, path
    for path, digest in manifest['files_sha256'].items():
        assert sha256(out/path) == digest, path
    assert json.loads((out/'protocol_frozen.json').read_text())['frozen_utc'] < json.loads((out/'fitting_complete.json').read_text())['completed_utc']
    samples = {r['sample_id']: r for r in read_csv(root/'samples.csv')}
    raw, meta = {}, {}
    for b in BATTERIES:
        with np.load(root/f'{b}.npz') as f:
            raw[b] = {k: f[k].copy() for k in ('x','context')}
        rr = read_csv(root/f'{b}_cycles.csv')
        meta[b] = np.array([[float(r[k]) for k in ('duration_s','discharge_Ah','end_time_s')] for r in rr])
    predictions = read_csv(out/'predictions.csv')
    lookup = {}
    for r in predictions:
        lb = '' if r['family'] == 'oracle' else str(r['lookback'])
        key = (r['method'], lb, str(r['seed']), r['sample_id'])
        assert key not in lookup
        lookup[key] = r
        np.testing.assert_allclose(float(r['soh_true']), float(samples[r['sample_id']]['soh']), rtol=0, atol=1e-12)
        np.testing.assert_allclose(float(r['absolute_error']), abs(float(r['soh_pred'])-float(r['soh_true'])), rtol=0, atol=1e-12)
    expected_keys = set()
    for sid in samples:
        expected_keys.update((v,'','deterministic',sid) for v in CONTEXT_GROUPS)
        expected_keys.update((v,str(lb),str(seed),sid) for v in VARIANTS for lb in LOOKBACKS for seed in SEEDS)
        expected_keys.update((v+'Oracle','',str(seed),sid) for v in VARIANTS for seed in SEEDS)
        expected_keys.update(('HistoricalFixed',str(lb),str(seed),sid) for lb in LOOKBACKS for seed in SEEDS)
    assert set(lookup) == expected_keys
    checked = 0
    param_counts = set()
    maximum_error = 0.
    fit_errors = []
    for outer in BATTERIES:
        split = json.loads((root/'folds'/f'test_{outer}__final'/'split.json').read_text())
        folder = out/'models'/f'test_{outer}'
        sc = json.loads((folder/'scalers.json').read_text())
        assert set(split['train_batteries']) == set(BATTERIES)-{outer}
        assert set(sc['fit_sample_ids']) == set(split['train'])
        assert set(split['train']).isdisjoint(split['test'])
        tr = [samples[s] for s in split['train']]
        te = [samples[s] for s in split['test']]
        ix = {}
        for b in split['train_batteries']:
            used = set()
            for r in tr:
                if r['battery'] == b:
                    n = int(r['history_end_exclusive'])
                    used.update(range(n-50,n))
            ix[b] = sorted(used)
        assert ix == sc['fit_cycle_indices']
        sw = np.concatenate([raw[b]['x'][ix[b]].reshape(-1,4) for b in split['train_batteries']])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            wm, ws = np.nanmean(sw,0), np.nanstd(sw,0)
        ws = np.where(ws>1e-6,ws,1.)
        sm = np.concatenate([meta[b][ix[b],:2] for b in split['train_batteries']])
        mm, ms = sm.mean(0), sm.std(0)
        ms = np.where(ms>1e-6,ms,1.)
        rc = np.stack([raw[r['battery']]['context'][int(r['history_end_exclusive'])-1] for r in tr]).astype(np.float64)
        cm, cs = np.nanmean(rc,0), np.nanstd(rc,0)
        cs = np.where(cs>1e-6,cs,1.)
        for key, expected in [('channel_mean',wm),('channel_std',ws),('metadata_mean',mm),('metadata_std',ms),('context_mean',cm),('context_std',cs)]:
            np.testing.assert_allclose(sc[key], expected, rtol=1e-6, atol=1e-6, err_msg=key)
        def arrays(rr, lb):
            waves, metadata, context = [],[],[]
            for r in rr:
                b, n = r['battery'], int(r['history_end_exclusive'])
                assert meta[b][n-1,2] < float(r['target_start_time_s'])
                wave = raw[b]['x'][n-lb:n]
                waves.append((np.where(np.isfinite(wave),wave,wm)-wm)/ws)
                metadata.append((meta[b][n-lb:n,:2]-mm)/ms)
                rc = raw[b]['context'][n-1].astype(np.float64)
                context.append((np.where(np.isfinite(rc),rc,cm)-cm)/cs)
            return tuple(torch.tensor(np.stack(a),dtype=torch.float32) for a in (waves,metadata,context)), np.stack(context)
        _, cxtr = arrays(tr,50)
        xte, cxte = arrays(te,50)
        ytr = np.array([float(r['soh']) for r in tr])
        for v, features in CONTEXT_GROUPS.items():
            fit = json.loads((folder/v/'fit.json').read_text())
            assert fit['train_sample_ids'] == split['train']
            with np.load(folder/v/'parameters.npz') as f:
                np.testing.assert_array_equal(f['features'],features)
                w, bias = f['weight'],float(f['bias'])
            err = cxtr[:,features] @ w + bias-ytr
            np.testing.assert_allclose(cxtr[:,features].T @ err/len(tr)+.1*w, 0, atol=1e-12)
            np.testing.assert_allclose(err.mean(),0,atol=1e-12)
            pred = cxte[:,features] @ w+bias
            expected = [float(lookup[v,'','deterministic',r['sample_id']]['soh_pred']) for r in te]
            np.testing.assert_allclose(pred,expected,rtol=0,atol=1e-8)
        for v in VARIANTS:
            for seed in SEEDS:
                bank = []
                for lb in LOOKBACKS:
                    dest = neural_folder(out,outer,v,seed,lb)
                    fit = json.loads((dest/'fit.json').read_text())
                    assert fit['train_sample_ids'] == split['train']
                    assert fit['train_batteries'] == split['train_batteries']
                    assert fit['epochs'] == 200
                    assert len(read_csv(dest/'training_history.csv')) == 200
                    set_seed(seed)
                    model = HistoryPredictor(v)
                    assert state_hash(model) == fit['initial_state_sha256']
                    ck = torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True)
                    assert ck['protocol_sha256'] == sig
                    assert (ck['variant'],ck['seed'],ck['lookback'],ck['outer_test_battery']) == (v,seed,lb,outer)
                    model.load_state_dict(ck['state_dict'])
                    assert state_hash(model) == fit['final_state_sha256']
                    param_counts.add(sum(p.numel() for p in model.parameters()))
                    model.eval()
                    with torch.inference_mode():
                        pred = model(xte[0][:,-lb:],xte[1][:,-lb:],xte[2]).numpy()
                        xtr,_ = arrays(tr,lb)
                        train_error = float((model(*xtr)-torch.tensor(ytr,dtype=torch.float32)).square().mean())
                    np.testing.assert_allclose(train_error,fit['training_mse'],rtol=1e-4,atol=1e-7)
                    expected = np.array([float(lookup[v,str(lb),str(seed),r['sample_id']]['soh_pred']) for r in te])
                    maximum_error = max(maximum_error,float(abs(pred-expected).max()))
                    np.testing.assert_allclose(pred,expected,rtol=0,atol=2e-6)
                    bank.append(expected)
                    fit_errors.append(dict(method=v,outer=outer,seed=seed,lookback=lb,training_mse=train_error))
                    checked += 1
                bank = np.stack(bank,1)
                truth = np.array([float(r['soh']) for r in te])
                best = np.argmin(abs(bank-truth[:,None]),1)
                for i,r in enumerate(te):
                    oracle = lookup[v+'Oracle','',str(seed),r['sample_id']]
                    assert int(oracle['lookback']) == LOOKBACKS[best[i]]
                    assert float(oracle['soh_pred']) == bank[i,best[i]]
    assert checked == 240 and len(param_counts) == 1
    # Independent aggregation directly from predictions, equally weighting cells.
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in predictions:
        lb = '' if r['family']=='oracle' else r['lookback']
        groups[(r['method'],lb)][r['seed']][r['outer_test_battery']].append(float(r['soh_pred'])-float(r['soh_true']))
    macro_lookup = {}
    summary = read_csv(out/'summary.csv')
    assert len(summary) == len(groups)
    for r in summary:
        seed_errors, seed_rmse = [],[]
        for seed, cells in groups[r['method'],r['lookback']].items():
            assert set(cells)==set(BATTERIES)
            mae = np.mean([np.mean(np.abs(e)) for e in cells.values()])
            seed_errors.append(mae)
            seed_rmse.append(np.mean([np.sqrt(np.mean(np.square(e))) for e in cells.values()]))
            macro_lookup[r['method'],r['lookback'],seed] = mae
        np.testing.assert_allclose([float(r['macro_mae']),float(r['macro_rmse'])], [np.mean(seed_errors),np.mean(seed_rmse)],rtol=0,atol=1e-12)
        sd = np.std(seed_errors,ddof=1) if len(seed_errors)>1 else 0.
        np.testing.assert_allclose(float(r['seed_sd']),sd,atol=1e-12)
    for r in read_csv(out/'paired_comparisons.csv'):
        deltas=[]
        for seed in SEEDS:
            aseed='deterministic' if r['candidate'] in CONTEXT_GROUPS else str(seed)
            bseed='deterministic' if r['reference'] in CONTEXT_GROUPS else str(seed)
            deltas.append(macro_lookup[r['candidate'],r['candidate_lb'],aseed]-macro_lookup[r['reference'],r['reference_lb'],bseed])
        np.testing.assert_allclose([float(r['mean_delta_mae']),float(r['seed_sd_delta'])], [np.mean(deltas),np.std(deltas,ddof=1)],atol=1e-12)
    write_json(out/'verification.json', dict(status='PASS',protocol_sha256=sig,
        verified_models=checked, ridge_models=16, prediction_rows=len(predictions), parameters_per_model=list(param_counts)[0],
        maximum_reproduction_error=maximum_error, checks=['data/code/artifact hashes','train-only scalers rebuilt',
        'causal input endpoints','canonical context reproduction','ridge normal equations','all checkpoint predictions and training losses',
        'same seeded initialization and parameter shapes','prediction grid and per-sample oracles','macro metrics and paired comparisons']))
    print(json.dumps(json.loads((out/'verification.json').read_text()),indent=2))


if __name__ == '__main__':
    main()

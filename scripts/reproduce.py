"""Reconstructed seed-42 LOBO experiment; see generated report for assumptions."""
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reproduction'
BATTERIES = ['RW3', 'RW4', 'RW5', 'RW6']
CONFIG = dict(seed=42, lookbacks=[10,20,50], resampling=128,
              encoder=[512,64,32], head=[32,32,1], pooling='mean',
              optimizer='Adam', lr=0.001, max_epochs=200, patience=25,
              validation_fraction=0.2, purge_steps=50, evaluation_start=50,
              input='steps[t-L:t]; excludes target step t',
              missing='temperature outside [-40,100] C -> NaN; training-only per-channel mean', device='cpu', threads=4)

def stamp(s):
    return datetime.strptime(s, '%d-%b-%Y %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()

def prepare():
    datasets, audit = {}, {}
    for battery in BATTERIES:
        path = ROOT / f'data/raw/{battery}.mat'
        steps = loadmat(path, simplify_cells=True)['data']['step']
        refs = [r for r in steps if r['comment'] == 'reference discharge']
        rt = np.array([stamp(r['date']) for r in refs])
        rq = np.array([np.trapezoid(np.atleast_1d(r['current']), np.atleast_1d(r['relativeTime']))/3600 for r in refs])
        order = np.argsort(rt); rt, rq = rt[order], rq[order]
        assert np.all(rq > 0) and np.all(np.diff(rt) > 0)
        xs, ts, indices = [], [], []
        for i,r in enumerate(steps):
            if r['comment'] != 'discharge (random walk)':
                continue
            t = np.atleast_1d(r['relativeTime']).astype(float)
            if len(t) < 2:
                continue
            assert np.all(np.diff(t) >= 0)
            grid = np.linspace(t[0],t[-1],128)
            v,c,temp = [np.interp(grid,t,np.atleast_1d(r[k])) for k in ['voltage','current','temperature']]
            xs.append(np.stack([v,c,np.diff(v,prepend=v[0]),temp],axis=-1))
            ts.append(stamp(r['date'])); indices.append(i)
        x = np.array(xs,dtype=np.float32); ts=np.array(ts)
        invalid_temp=(x[:,:,3]<-40)|(x[:,:,3]>100)|(~np.isfinite(x[:,:,3]))
        x[:,:,3][invalid_temp]=np.nan
        cap = np.interp(ts,rt,rq).astype(np.float32)
        y = (cap/rq[0]).astype(np.float32)
        assert np.isfinite(y).all() and np.all(np.diff(ts)>=0)
        old = np.load(ROOT/f'cache/{battery}_r128_v2.npz',allow_pickle=False)
        equal_indices = np.array_equal(indices,old['step_index'])
        finite = np.isfinite(x)&np.isfinite(old['x'])
        delta = float(np.max(np.abs(x[finite]-old['x'][finite])))
        audit[battery] = dict(raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            raw_steps=len(steps),input_steps=len(x),reference_measurements=len(rq),
            test_samples=len(x)-50,temperature_missing_fraction=float(np.mean(~np.isfinite(x[:,:,3]))),
            first_capacity_Ah=float(rq[0]), last_capacity_Ah=float(rq[-1]),
            before_first_reference=int(np.sum(ts<rt[0])),after_last_reference=int(np.sum(ts>rt[-1])),
            cache_indices_equal=equal_indices,cache_finite_feature_max_difference=delta,
            cache_missing_mask_equal=bool(np.array_equal(np.isfinite(x),np.isfinite(old['x']))),
            cache_soh_max_difference=float(np.max(np.abs(y-old['soh']))))
        np.savez_compressed(OUT/f'{battery}_rebuilt.npz',x=x,soh=y,time=ts,step_index=indices,ref_time=rt,ref_capacity=rq)
        datasets[battery]=(x,y)
        print('PREPARE',battery,json.dumps(audit[battery]),flush=True)
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    return datasets,audit

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Flatten(1),nn.Linear(512,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x,lb):
        z=self.encoder(x)
        prefix=torch.cat([z.new_zeros((1,32)),z.cumsum(0)],dim=0)
        # Output index 0 corresponds to target index lb.
        means=(prefix[lb:-1]-prefix[:-lb-1])/lb
        return self.head(means).squeeze(-1)

def metrics(y,p):
    return dict(MAE=float(np.mean(np.abs(y-p))),RMSE=float(np.sqrt(np.mean((y-p)**2))),
                MAPE_percent=float(np.mean(np.abs((y-p)/y))*100))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    CONFIG['versions']=dict(numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)
    (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    data,audit=prepare(); rows=[]; history=[]
    for test in BATTERIES:
        train=[b for b in BATTERIES if b!=test]
        cuts={b:int(len(data[b][0])*0.8) for b in train}
        fit=np.concatenate([data[b][0][:cuts[b]-50] for b in train])
        fit=np.where(np.isfinite(fit),fit,np.nan)
        mean=np.nanmean(fit,axis=(0,1)); std=np.nanstd(fit,axis=(0,1)); std=np.maximum(std,1e-6)
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        tensors={b:torch.from_numpy(((np.where(np.isfinite(x),x,mean)-mean)/std).astype(np.float32)) for b,(x,y) in data.items()}
        targets={b:torch.from_numpy(y) for b,(x,y) in data.items()}
        for lb in CONFIG['lookbacks']:
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            model=Predictor(); opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
            best=float('inf'); stale=0; best_state=None; start=time.perf_counter()
            for epoch in range(1,CONFIG['max_epochs']+1):
                model.train(); opt.zero_grad(); losses=[]
                for b in train:
                    end=cuts[b]-50
                    p=model(tensors[b][:end],lb)
                    losses.append(((p[50-lb:]-targets[b][50:end])**2).sum())
                loss=sum(losses)/sum(cuts[b]-100 for b in train)
                assert torch.isfinite(loss)
                loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    vl=[]
                    for b in train:
                        # Validation inputs begin at the chronological boundary;
                        # earlier 50-step purge excludes training input overlap.
                        p=model(tensors[b][cuts[b]-lb:],lb)
                        vl.append(((p-targets[b][cuts[b]:])**2).sum())
                    val=(sum(vl)/sum(len(targets[b])-cuts[b] for b in train)).item()
                history.append(dict(battery=test,lookback=lb,epoch=epoch,train_mse=loss.item(),val_mse=val))
                if val<best-1e-9:
                    best=val; best_epoch=epoch; stale=0
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:
                    stale+=1
                if epoch%50==0:
                    print('TRAIN',test,lb,epoch,'validation MSE',round(val,6),flush=True)
                if stale>=CONFIG['patience']:
                    break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                p=model(tensors[test],lb)[50-lb:].numpy()
            y=data[test][1][50:]
            assert len(y)==len(p) and np.isfinite(p).all()
            row=dict(battery=test,lookback=lb,**metrics(y,p),samples=len(y),best_epoch=best_epoch,
                     epochs_run=epoch,validation_mse=best,training_seconds=time.perf_counter()-start)
            rows.append(row)
            torch.save(dict(state_dict=best_state,mean=mean.tolist(),std=std.tolist(),config=CONFIG,lookback=lb,test_battery=test),OUT/f'{test}_lb{lb}.pt')
            with (OUT/f'{test}_lb{lb}_predictions.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['target_index','soh_true','soh_pred'])
                w.writerows(zip(range(50,len(data[test][1])),y,p))
            write_csv(OUT/'metrics.csv',rows);write_csv(OUT/'training_history.csv',history)
            print('RESULT',json.dumps(row),flush=True)
    lines=['# Fixed Lookback 재실행 결과','',
           '기존 소스가 없어 설명과 기존 캐시를 기준으로 복원한 실험이다. 과거 수치의 정확한 재현은 보장하지 않는다.',
           '', '| Battery | LB10 MAE | LB20 MAE | LB50 MAE | Best |','|---|---:|---:|---:|---:|']
    for b in BATTERIES:
        rr=[r for r in rows if r['battery']==b]
        lines.append('| '+b+' | '+' | '.join(f"{r['MAE']:.4f}" for r in rr)+' | '+str(min(rr,key=lambda r:r['MAE'])['lookback'])+' |')
    means=[np.mean([r['MAE'] for r in rows if r['lookback']==lb]) for lb in [10,20,50]]
    lines.append('| Mean (battery macro) | '+' | '.join(f'{v:.4f}' for v in means)+' | '+str([10,20,50][int(np.argmin(means))])+' |')
    lines += ['', '## 실험 설정과 해석 범위','',
      '- Seed 42, LOBO 4 folds × Lookback 10/20/50 = 12 models. MSE loss; MAE/RMSE/MAPE saved in metrics.csv.',
      '- 입력 단위는 완전한 RW cycle이 아니라 개별 discharge (random walk) step이다. 기존 캐시의 step_index와 비교했다.',
      '- 원본 MAT에서 128점으로 시간 보간한 V/I/dV/T를 재생성했다. dV는 보간한 전압의 인접 차분이며 첫 값은 0이다.',
      '- reference discharge의 전류를 초 단위 시간으로 사다리꼴 적분하고 3600으로 나누어 Ah를 계산했다. 첫 기준 용량으로 SOH를 정규화했다.',
      '- 기준 용량을 step 시작 날짜에 선형 보간했다. 기준 범위 밖은 np.interp의 끝값 유지다. 실제 매-step SOH 측정값이 아니며 미래 reference가 보간 라벨에 사용된다.',
      '- 타깃 t에 대해 [t-L,t)만 입력한다. 모든 길이는 0-based t=50부터 동일한 학습/평가 타깃을 사용한다.',
      '- 각 학습 배터리의 마지막 20%를 validation으로 사용하고 직전 50개 타깃을 학습에서 제외했다. 테스트 배터리는 정규화/결측 대체/early stopping에 사용하지 않았다.',
      '- 원본 온도에 약 -4094°C의 비정상 값이 있었다. 온도가 [-40,100]°C 밖인 경우 NaN으로 바꾸고 학습 구간 채널 평균으로 대체했다. RW3의 상당 부분이 이에 해당한다(data_audit.json). 최초 품질 확인용 실행은 이 처리가 없어 폐기했으며 본 결과는 수정 후 실행이다.',
      '- 복원 모델: Flatten(128×4) → Linear(512,64) → ReLU → Linear(64,32) → ReLU; 과거 embedding 평균 → Linear(32,32) → ReLU → Linear(32,1). 각 LB 모델은 독립 학습한다.',
      '- Adam lr=0.001, full-batch, 최대 200 epochs, validation MSE patience 25. 이 세부 설정은 원래 코드가 없어 새로 명시한 설정이다.',
      '- 시드 1회 결과이므로 최적 길이의 통계적 안정성을 입증하지 않는다. Mean의 최적 길이는 사후 요약이며 독립적으로 선정한 Global Best 성능이 아니다.',
      '', '## 재실행','', '`.venv-repro/bin/python scripts/reproduce.py`', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:10]),flush=True)

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    run()

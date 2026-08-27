#!/usr/bin/env python3
"""Leakage-safe fixed-lookback SOH experiment for NASA RW3--RW6."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA = Path("/Users/kimsj/Downloads/Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab")
BATTERIES = ("RW3", "RW4", "RW5", "RW6")
LOOKBACKS = (10, 20, 50)
FEATURES = ("Voltage", "Current", "dV", "Temperature")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def timestamp(s: str) -> float:
    return datetime.strptime(str(s), "%d-%b-%Y %H:%M:%S").timestamp()


def resample_step(step: dict, n: int = 128) -> np.ndarray:
    t, v, i, temp = (as_1d(step[k]) for k in ("relativeTime", "voltage", "current", "temperature"))
    size = min(map(len, (t, v, i, temp)))
    t, v, i, temp = t[:size], v[:size], i[:size], temp[:size]
    good = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    t, v, i, temp = t[good], v[good], i[good], temp[good]
    if len(t) < 2:
        raise ValueError("cycle has fewer than two valid samples")
    order = np.argsort(t); t, v, i, temp = t[order], v[order], i[order], temp[order]
    t, unique = np.unique(t, return_index=True); v, i, temp = v[unique], i[unique], temp[unique]
    grid = np.linspace(t[0], t[-1], n)
    vr, ir = (np.interp(grid, t, z) for z in (v, i))
    # RW3's temperature sensor later emits -4094 C. Treat physical outliers as
    # missing; a fully missing trace is imputed with the training mean downstream.
    temp[(temp < -20) | (temp > 100)] = np.nan
    tg = np.isfinite(temp)
    tr = np.interp(grid, t[tg], temp[tg]) if tg.sum() >= 2 else np.full(n, np.nan)
    dv = np.diff(vr, prepend=vr[0])
    return np.stack((vr, ir, dv, tr), axis=-1).astype(np.float32)


def load_battery(path: Path, resample: int = 128) -> dict:
    raw = sio.loadmat(path, simplify_cells=True)["data"]
    steps = list(np.atleast_1d(raw["step"]))
    refs, cycles, skipped = [], [], 0
    for idx, s in enumerate(steps):
        comment = str(s["comment"])
        if comment == "reference discharge":
            t, cur = as_1d(s["relativeTime"]), as_1d(s["current"])
            n = min(len(t), len(cur)); cap = abs(float(np.trapezoid(cur[:n], t[:n]) / 3600.0))
            refs.append((timestamp(s["date"]), cap))
        elif comment == "discharge (random walk)":
            try:
                cycles.append((timestamp(s["date"]), idx, resample_step(s, resample)))
            except ValueError:
                skipped += 1
    if not refs or not cycles:
        raise ValueError(f"No reference/cycle data in {path}")
    refs.sort(); cycles.sort()
    rt, cap = np.array([x[0] for x in refs]), np.array([x[1] for x in refs])
    ct = np.array([x[0] for x in cycles])
    # np.interp uses boundary reference values outside the measured interval.
    cycle_cap = np.interp(ct, rt, cap)
    x = np.stack([x[2] for x in cycles])
    return {"x": x, "capacity": cycle_cap.astype(np.float32),
            "soh": (cycle_cap / cap[0]).astype(np.float32), "time": ct,
            "step_index": np.array([x[1] for x in cycles]), "ref_time": rt,
            "ref_capacity": cap, "skipped": skipped, "n_steps": len(steps)}


def cached_load(data_dir: Path, cache_dir: Path, resample: int = 128) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True); out = {}
    for battery in BATTERIES:
        src, cache = data_dir / f"{battery}.mat", cache_dir / f"{battery}_r{resample}_v2.npz"
        if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
            z = np.load(cache); out[battery] = {k: z[k] for k in z.files}
        else:
            d = load_battery(src, resample)
            np.savez_compressed(cache, **d); out[battery] = d
    return out


def structure_table(data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for b, d in data.items():
        x = d["x"]
        rows.append({"Battery ID": b, "Number of Cycles": len(x),
                     "Initial Capacity (Ah)": d["ref_capacity"][0],
                     "Last Capacity (Ah)": d["ref_capacity"][-1],
                     "Min Temperature (C)": np.nanmin(x[..., 3]),
                     "Max Temperature (C)": np.nanmax(x[..., 3]),
                     "Missing Values": int(np.isnan(x).sum()),
                     "Cycles with Missing Temperature": int(np.isnan(x[..., 3]).any(axis=1).sum()),
                     "Cycle Shape": str(tuple(x.shape[1:])),
                     "Feature dtype": str(x.dtype), "Reference Measurements": len(d["ref_capacity"]),
                     "Raw Steps": int(d["n_steps"]), "Skipped Cycles": int(d["skipped"])})
    return pd.DataFrame(rows)


class WindowDataset(Dataset):
    def __init__(self, data, batteries, lookback, mean, std, max_cycles=None):
        self.data, self.lb, self.mean, self.std = data, lookback, mean, std
        self.items = []
        for b in batteries:
            n = len(data[b]["x"]); start = 49
            ids = np.arange(start, n)
            if max_cycles: ids = ids[-max_cycles:]
            self.items.extend((b, int(t)) for t in ids)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        b, t = self.items[idx]
        x = (self.data[b]["x"][t-self.lb+1:t+1] - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.data[b]["soh"][t]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y), b, t


class SOHPredictor(nn.Module):
    """Same parameterization for every LB; only number of pooled cycles changes."""
    def __init__(self, points=128, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(points * 4, 64), nn.ReLU(), nn.Linear(64, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 3, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, x):
        b, l, p, f = x.shape; z = self.encoder(x.reshape(b*l, p, f)).reshape(b, l, -1)
        pooled = torch.cat((z.mean(1), z.std(1, unbiased=False), z[:, -1]), dim=1)
        return self.head(pooled).squeeze(1)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred); err = pred-y
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAPE": float(np.mean(np.abs(err)/np.clip(np.abs(y), 1e-8, None))*100)}


def predict(model, loader, device):
    model.eval(); ys, ps, bs, ts = [], [], [], []
    with torch.no_grad():
        for x, y, b, t in loader:
            p = model(x.to(device)).cpu().numpy(); ys.extend(y.numpy()); ps.extend(p); bs.extend(b); ts.extend(t.numpy())
    return pd.DataFrame({"Battery": bs, "CycleIndex": ts, "SOH": ys, "Prediction": ps})


def train_one(data, train_bats, val_bat, test_bat, lb, args, out_dir):
    train_x = np.concatenate([data[b]["x"] for b in train_bats])
    mean = np.nanmean(train_x, axis=(0, 1), keepdims=True); std = np.nanstd(train_x, axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    ds = {"train": WindowDataset(data, train_bats, lb, mean, std, args.max_cycles),
          "val": WindowDataset(data, [val_bat], lb, mean, std, args.max_cycles),
          "test": WindowDataset(data, [test_bat], lb, mean, std, args.max_cycles)}
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k=="train"), num_workers=0) for k,v in ds.items()}
    seed_all(args.seed); device = torch.device(args.device)
    model = SOHPredictor(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); loss_fn = nn.MSELoss()
    best, best_state, patience, history = math.inf, None, 0, []
    for epoch in range(1, args.epochs+1):
        model.train(); losses=[]
        for x,y,_,_ in loaders["train"]:
            opt.zero_grad(); loss=loss_fn(model(x.to(device)), y.to(device)); loss.backward(); opt.step(); losses.append(loss.item())
        val_df=predict(model, loaders["val"], device); val_mse=float(np.mean((val_df.SOH-val_df.Prediction)**2))
        history.append({"epoch":epoch,"train_mse":float(np.mean(losses)),"val_mse":val_mse})
        if val_mse < best-1e-7:
            best=val_mse; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience += 1
            if patience >= args.patience: break
    model.load_state_dict(best_state); pred=predict(model, loaders["test"], device)
    pred["Lookback"] = lb; pred.to_csv(out_dir/f"pred_{test_bat}_lb{lb}.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir/f"history_{test_bat}_lb{lb}.csv", index=False)
    np.savez(out_dir/f"norm_{test_bat}.npz", mean=mean, std=std, train_batteries=train_bats, validation_battery=val_bat)
    return pred, metrics(pred.SOH, pred.Prediction)


def save_figures(data, battery_results, stage_results, summary, out):
    sns.set_theme(style="whitegrid")
    fig, ax=plt.subplots(figsize=(9,5))
    for b,d in data.items(): ax.plot(np.arange(len(d["soh"]))+1,d["soh"],label=b,linewidth=1.2)
    ax.set(xlabel="Random-walk cycle",ylabel="SOH",title="Battery SOH degradation curves"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure1_soh_curves.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(data=battery_results,x="Battery",y="MAE",hue="Lookback",ax=ax,palette="viridis")
    ax.set_title("Fixed-lookback MAE by unseen battery"); fig.tight_layout(); fig.savefig(out/"figure2_battery_mae.png",dpi=180); plt.close(fig)
    heat=stage_results.pivot(index="Battery",columns="Stage",values="Best Lookback").reindex(columns=["Early","Middle","Late"])
    fig,ax=plt.subplots(figsize=(7,4)); sns.heatmap(heat,annot=True,fmt=".0f",cmap=sns.color_palette(["#440154","#21918c","#fde725"],as_cmap=True),cbar_kws={"label":"Optimal lookback"},ax=ax)
    ax.set_title("Optimal lookback by battery and degradation stage"); fig.tight_layout(); fig.savefig(out/"figure3_stage_heatmap.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); sns.barplot(data=summary,x="Lookback",y="Mean MAE",ax=ax,color="#377eb8")
    ax.set_title("Mean MAE across unseen batteries"); fig.tight_layout(); fig.savefig(out/"figure4_mean_mae.png",dpi=180); plt.close(fig)


def run(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    data=cached_load(Path(args.data_dir),Path(args.cache)); table=structure_table(data)
    table.to_csv(out/"data_structure.csv",index=False); print("\nDATA STRUCTURE\n",table.to_string(index=False))
    # Cyclic validation assignment is fixed before testing any lookback.
    folds=[]; all_preds=[]; result_rows=[]
    for i,test in enumerate(BATTERIES):
        val=BATTERIES[(i+1)%len(BATTERIES)]; train=[b for b in BATTERIES if b not in (test,val)]
        folds.append({"test":test,"validation":val,"train":train})
        for lb in LOOKBACKS:
            print(f"Fold test={test} val={val} train={train} LB={lb}",flush=True)
            pred,m=train_one(data,train,val,test,lb,args,out); all_preds.append(pred)
            result_rows.append({"Battery":test,"Lookback":lb,**m})
    results=pd.DataFrame(result_rows); results.to_csv(out/"battery_results_long.csv",index=False)
    wide=results.pivot(index="Battery",columns="Lookback",values="MAE").rename(columns=lambda x:f"LB{x} MAE")
    wide["Best Lookback"]=wide.idxmin(axis=1).str.extract(r"(\d+)")[0].astype(int); wide.reset_index().to_csv(out/"battery_results.csv",index=False)
    summary=results.groupby("Lookback",as_index=False).agg(**{"Mean MAE":("MAE","mean"),"Mean RMSE":("RMSE","mean"),"Mean MAPE":("MAPE","mean")})
    summary.to_csv(out/"lookback_summary.csv",index=False)
    preds=pd.concat(all_preds,ignore_index=True); stage_rows=[]
    for b in BATTERIES:
        n=len(data[b]["x"])
        for stage,lo,hi in (("Early",0,1/3),("Middle",1/3,2/3),("Late",2/3,1)):
            row={"Battery":b,"Stage":stage}
            for lb in LOOKBACKS:
                z=preds[(preds.Battery==b)&(preds.Lookback==lb)]; frac=(z.CycleIndex+1)/n; z=z[(frac>lo)&(frac<=hi)]
                row[f"LB{lb}"]=metrics(z.SOH,z.Prediction)["MAE"] if len(z) else np.nan
            vals={lb:row[f"LB{lb}"] for lb in LOOKBACKS}; row["Best Lookback"]=min(vals,key=lambda k: vals[k] if np.isfinite(vals[k]) else np.inf); stage_rows.append(row)
    stages=pd.DataFrame(stage_rows); stages.to_csv(out/"stage_results.csv",index=False)
    save_figures(data,results,stages,summary,out)
    config=vars(args).copy(); config.update({"folds":folds,"features":FEATURES,"lookbacks":LOOKBACKS,"model_parameters":sum(p.numel() for p in SOHPredictor(hidden=args.hidden).parameters())})
    (out/"config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False,default=str))
    print("\nBATTERY RESULTS\n",wide.reset_index().to_string(index=False)); print("\nSUMMARY\n",summary.to_string(index=False)); print(f"\nSaved to {out.resolve()}")


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    for name in ("inspect","run"):
        q=sub.add_parser(name); q.add_argument("--data-dir",default=str(DEFAULT_DATA)); q.add_argument("--cache",default="cache"); q.add_argument("--output",default="outputs")
        if name=="run":
            q.add_argument("--epochs",type=int,default=30); q.add_argument("--patience",type=int,default=5); q.add_argument("--batch-size",type=int,default=128)
            q.add_argument("--lr",type=float,default=1e-3); q.add_argument("--hidden",type=int,default=32); q.add_argument("--seed",type=int,default=42)
            q.add_argument("--device",default="cpu"); q.add_argument("--max-cycles",type=int,default=None)
    args=p.parse_args()
    data=cached_load(Path(args.data_dir),Path(args.cache)); table=structure_table(data)
    Path(args.output).mkdir(parents=True,exist_ok=True); table.to_csv(Path(args.output)/"data_structure.csv",index=False)
    print(table.to_string(index=False))
    if args.command=="run": run(args)
if __name__=="__main__": main()

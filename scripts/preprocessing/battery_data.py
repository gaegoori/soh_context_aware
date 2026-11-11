"""NumPy dataset usable directly with torch.utils.data.DataLoader."""
from scripts.paths import PROJECT_ROOT
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = PROJECT_ROOT / 'data/processed/main_v1'


class BatteryDataset:
    def __init__(self, fold, split, lookback=20, root=DEFAULT_ROOT):
        if lookback not in [10,20,50]:
            raise ValueError('lookback must be 10, 20 or 50')
        if split not in ['train','validation','test']:
            raise ValueError('split must be train, validation or test')
        self.root=Path(root); self.lookback=lookback
        folder=self.root/'folds'/fold
        split_info=json.loads((folder/'split.json').read_text())
        by_id={r['sample_id']:r for r in csv.DictReader((self.root/'samples.csv').open())}
        self.rows=[by_id[s] for s in split_info[split]]
        self.arrays={};self.masks={}
        for b in {r['battery'] for r in self.rows}:
            with np.load(folder/f'{b}.npz',allow_pickle=False) as f:
                self.arrays[b]={k:f[k] for k in ['x','context']}
            with np.load(self.root/f'{b}.npz',allow_pickle=False) as f:
                self.masks[b]={k:f[k] for k in ['observed_mask','context_observed_mask']}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r=self.rows[index]; b=r['battery']; n=int(r['history_end_exclusive'])
        return dict(x=self.arrays[b]['x'][n-self.lookback:n].copy(),
            observed_mask=self.masks[b]['observed_mask'][n-self.lookback:n].copy(),
            context=self.arrays[b]['context'][n-1].copy(),
            context_observed_mask=self.masks[b]['context_observed_mask'][n-1].copy(),
            soh=np.float32(r['soh']), sample_id=r['sample_id'], battery=b)

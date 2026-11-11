"""Fixed-capacity input ablations for exploratory battery SOH follow-up."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.models.adaptive_models import CycleHistory, context_at, fit_context_scaler, transform_context
from scripts.experiments.train_adaptive_selector import load_final_split
from scripts.experiments.train_lookback_oof import read_csv

BATTERIES = ('RW3', 'RW4', 'RW5', 'RW6')
LOOKBACKS = (10, 20, 50)
SEEDS = (42, 0, 1, 2, 3)
VARIANTS = {
    'WaveMean': (False, False, False),
    'WaveMetaMean': (True, False, False),
    'WaveMetaContextMean': (True, True, False),
    'WaveMetaContextTrend': (True, True, True),
}
CONTEXT_GROUPS = {
    'ContextAll': list(range(15)),
    'ContextNoDurationAh': [i for i in range(15) if i not in (2, 3, 4, 6, 7, 8)],
    'ContextUsageOnly': [0, 1, 2],
    'ContextNoTemperature': [i for i in range(15) if i not in (5, 13, 14)],
}


class HistoryPredictor(nn.Module):
    """Same allocated parameters and initialization across all input masks.

    Trend = least-squares embedding slope against relative cycle position [-1,1].
    This tests linear order information; it is not a general sequence model.
    """
    def __init__(self, variant):
        super().__init__()
        self.metadata_on, self.context_on, self.trend_on = VARIANTS[variant]
        self.encoder = nn.Sequential(nn.Linear(514, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(79, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, wave, metadata, context):
        batch, length = wave.shape[:2]
        metadata = metadata if self.metadata_on else torch.zeros_like(metadata)
        inputs = torch.cat([wave.reshape(batch, length, 512), metadata], dim=-1)
        embeddings = self.encoder(inputs)
        if self.trend_on:
            position = torch.linspace(-1, 1, length, device=wave.device, dtype=wave.dtype)
            trend = (embeddings * position[None, :, None]).sum(1) / position.square().sum()
        else:
            trend = torch.zeros_like(embeddings[:, 0])
        context = context if self.context_on else torch.zeros_like(context)
        return self.head(torch.cat([embeddings.mean(1), trend, context], -1)).squeeze(-1)


def metadata_array(history):
    return np.column_stack([history.duration, history.discharge_ah])


def train_bundle(root, outer):
    """Load only training-cell histories; derive all additional scalers from them."""
    root = Path(root)
    split, signal_scalers = load_final_split(root, outer, BATTERIES)
    by_id = {r['sample_id']: r for r in read_csv(root / 'samples.csv')}
    rows = [by_id[sid] for sid in split['train']]
    histories = {b: CycleHistory.load(root, b) for b in split['train_batteries']}
    context = np.stack([context_at(histories[r['battery']], int(r['history_end_exclusive'])) for r in rows])
    cm, cs = fit_context_scaler(context)
    indices = {}
    for b in split['train_batteries']:
        used = set()
        for r in rows:
            if r['battery'] == b:
                n = int(r['history_end_exclusive'])
                used.update(range(n - 50, n))
        indices[b] = sorted(used)
    if indices != signal_scalers['fit_cycle_indices']:
        raise ValueError('Metadata and signal normalization use different training cycles')
    metadata = np.concatenate([metadata_array(histories[b])[indices[b]] for b in split['train_batteries']])
    if not np.isfinite(metadata).all():
        raise ValueError('Cycle duration/Ah metadata must be observed')
    mm, ms = metadata.mean(0), metadata.std(0)
    scalers = dict(channel_mean=signal_scalers['channel_mean'], channel_std=signal_scalers['channel_std'],
                   context_mean=cm.tolist(), context_std=cs.tolist(), metadata_mean=mm.tolist(),
                   metadata_std=np.where(ms > 1e-6, ms, 1.).tolist(),
                   fit_sample_ids=split['train'], fit_cycle_indices=indices)
    return split, rows, histories, scalers


def input_arrays(rows, histories, scalers, lookback):
    """All inputs end at n-1, strictly before the measured reference target."""
    waves, metadata, contexts = [], [], []
    wm, ws = [np.asarray(scalers[k], dtype=np.float32) for k in ('channel_mean', 'channel_std')]
    mm, ms = [np.asarray(scalers[k]) for k in ('metadata_mean', 'metadata_std')]
    for row in rows:
        h, n = histories[row['battery']], int(row['history_end_exclusive'])
        if n < lookback or h.end_time[n-1] >= float(row['target_start_time_s']):
            raise ValueError('History overlaps reference target or is too short')
        raw = h.x[n-lookback:n]
        waves.append((np.where(np.isfinite(raw), raw, wm) - wm) / ws)
        metadata.append((metadata_array(h)[n-lookback:n] - mm) / ms)
        contexts.append(context_at(h, n))
    context = transform_context(np.stack(contexts), scalers['context_mean'], scalers['context_std'])
    arrays = [np.asarray(a, dtype=np.float32) for a in (waves, metadata, context)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite normalized model input')
    return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in arrays)

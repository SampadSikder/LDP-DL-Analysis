"""Resumable on-disk storage of generated rounds.

npz and manifest.json file
"""

import json
import os
from collections import defaultdict

import numpy as np
import psutil

from .scenarios import Round, generate_round
from .sources import SOURCES


def shard_name(protocol: str, epsilon: float, source: str) -> str:
    return f"{protocol}_eps{epsilon:g}_{source}.npz"


class RoundStore:
    def __init__(self, root: str):
        self.root = root
        self.manifest_path = os.path.join(root, 'manifest.jsonl')
        os.makedirs(root, exist_ok=True)

    def manifest(self) -> dict:
        out = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        out[rec['round_id']] = rec
        return out

    def _append(self, records):
        with open(self.manifest_path, 'a') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

    def generate(self, specs, banks, log_every: int = 2000):
        """Generate every spec not already in the manifest; writes one shard per group."""
        done = self.manifest()
        todo = [s for s in specs if s.round_id not in done]
        by_shard = defaultdict(list)
        for s in todo:
            by_shard[(s.split, s.protocol, s.epsilon, s.source)].append(s)
        n_done = 0
        for (split, protocol, eps, source), group in sorted(by_shard.items()):
            path = os.path.join(self.root, split, shard_name(protocol, eps, source))
            arrays = self._load_shard(path, drop={s.round_id for s in group})
            records = []
            for s in group:
                pop = banks[source].values(s.split, s.rep)
                r = generate_round(s, pop, SOURCES[source].printed_targets)
                rec = s.to_dict()
                rec.update(status=r.status, labels=r.labels, meta=r.meta)
                if r.generated:
                    arrays['reports'].append(r.reports.astype(np.float32))
                    arrays['groups'].append(r.groups)
                    arrays['is_fake'].append(r.is_fake)
                    arrays['round_ids'].append(s.round_id)
                records.append(rec)
                n_done += 1
                if log_every and n_done % log_every == 0:
                    rss = psutil.Process().memory_info().rss / 2 ** 30
                    print(f"  generated {n_done:,}/{len(todo):,} (rss {rss:.2f} GB)", flush=True)
            self._save_shard(path, arrays)
            self._append(records)    
        return len(todo)

    @staticmethod
    def _load_shard(path, drop=frozenset()):
        """Existing shard content, minus rounds about to be regenerated (crash recovery)."""
        arrays = {'reports': [], 'groups': [], 'is_fake': [], 'round_ids': []}
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                off, rids = z['offsets'], z['round_ids']
                rep, grp, fake = z['reports'], z['groups'], z['is_fake']
            for i, rid in enumerate(rids):
                if str(rid) in drop:
                    continue
                sl = slice(off[i], off[i + 1])
                arrays['reports'].append(rep[sl])
                arrays['groups'].append(grp[sl])
                arrays['is_fake'].append(fake[sl])
                arrays['round_ids'].append(str(rid))
        return arrays

    @staticmethod
    def _save_shard(path, arrays):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lens = [len(r) for r in arrays['reports']]
        offsets = np.r_[0, np.cumsum(lens)].astype(np.int64)
        cat = lambda k, dt: np.concatenate(arrays[k]).astype(dt) if arrays[k] else np.empty(0, dt)
        tmp = path + '.tmp.npz'
        np.savez(tmp, reports=cat('reports', np.float32), groups=cat('groups', np.int8),
                 is_fake=cat('is_fake', bool), offsets=offsets,
                 round_ids=np.asarray(arrays['round_ids'], dtype='U16'))
        os.replace(tmp, path)

    # ---------------- reading ----------------
    def load(self, split: str, protocol: str, epsilon: float, source: str):
        """List of (round_id, observation arrays, is_fake) for one shard."""
        path = os.path.join(self.root, split, shard_name(protocol, epsilon, source))
        if not os.path.exists(path):
            return []
        with np.load(path, allow_pickle=False) as z:
            off, rids = z['offsets'], z['round_ids']
            rep, grp, fake = z['reports'].astype(np.float64), z['groups'], z['is_fake']
        out = []
        for i, rid in enumerate(rids):
            sl = slice(off[i], off[i + 1])
            out.append((str(rid), rep[sl], grp[sl], fake[sl]))
        return out

"""Portable saved splits and deterministic nested training-label budgets."""
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from e0kit.data import load_partition, augmentation_specs


def load_manifest(path, data_root=None):
    path=Path(path)
    m=json.loads(path.read_text())
    root=Path(data_root or m['data_root']).expanduser().resolve()
    m['data_root']=str(root)
    seen=set()
    for split in ('train','validation','test'):
        if not m['splits'][split]: raise ValueError('Empty '+split)
        for row in m['splits'][split]:
            p=(root/row['relative_path']).resolve()
            if not p.is_relative_to(root): raise ValueError('Image path escapes data root')
            if p in seen: raise ValueError('Split leakage: '+str(p))
            if not p.is_file(): raise FileNotFoundError(p)
            seen.add(p)
            if not 0<=row['label']<len(m['classes']): raise ValueError('Invalid label')
    return m


def train_indices(manifest, shots=None, seed=42):
    rows=manifest['splits']['train']
    if shots is None: return list(range(len(rows)))
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[row['label']].append(i)
    if shots<1 or len(groups)!=len(manifest['classes']) or min(map(len,groups.values()))<shots:
        raise ValueError('Training set cannot support requested per-class budget')
    def digest(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    selected=[]
    for label,ids in groups.items():
        ids.sort(key=lambda i:(digest([seed,manifest['dataset'],label,rows[i]['relative_path']]),i))
        selected.extend(ids[:shots])
    return sorted(selected)


def training_data(manifest, shots, seed):
    selected=train_indices(manifest,shots,seed)
    rows=[manifest['splits']['train'][i] for i in selected]
    x,y=load_partition(rows,manifest['data_root'],training=True)
    return x,y,selected


def specs_for(manifest, images, epoch, seed):
    shapes=[tuple(x.shape[-2:]) for x in images]
    specs=augmentation_specs(len(images),epoch,shapes,seed)
    dataset=manifest['dataset']
    if dataset.startswith('KTH'):
        for i,(h,w) in enumerate(shapes): specs[i]=[0,0,h,w,0,0,0]
    return specs

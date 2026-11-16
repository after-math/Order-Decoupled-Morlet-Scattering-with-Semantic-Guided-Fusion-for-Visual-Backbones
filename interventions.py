"""Frozen-model gate-context intervention; static filters never change per image."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from model import Experiment
from data import load_manifest
from e0kit.data import load_partition


def donor_map(labels,kind,seed):
    labels=np.asarray(labels);rng=np.random.default_rng(seed);n=len(labels)
    if n<2: raise ValueError('At least two images required')
    donor=np.empty(n,dtype=np.int64)
    if kind=='same':
        for label in np.unique(labels):
            ids=rng.permutation(np.flatnonzero(labels==label))
            if len(ids)<2: raise ValueError('Same-class swapping requires two samples per class')
            donor[ids]=np.roll(ids,1)
    elif kind=='random':
        ids=rng.permutation(n);donor[ids]=np.roll(ids,1)
    elif kind=='other':
        classes=rng.permutation(np.unique(labels));groups=[rng.permutation(np.flatnonzero(labels==k)) for k in classes]
        shift=max(map(len,groups))
        if shift>n//2: raise ValueError('One-to-one other-class matching is impossible')
        ids=np.concatenate(groups);donor[ids]=np.roll(ids,shift)
    else: raise ValueError(kind)
    assert len(set(donor))==n and np.all(donor!=np.arange(n))
    if kind=='same': assert np.all(labels==labels[donor])
    if kind=='other': assert np.all(labels!=labels[donor])
    return donor


def restore(checkpoint,device):
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Experiment(**cp['spec']).to(device).eval()
    model.load_state_dict(cp['model'],strict=True)
    model.requires_grad_(False)
    return model


@torch.no_grad()
def context_cache(model,images,device,batch):
    values=[]
    for x in images.split(batch):
        with torch.autocast(device.type,dtype=torch.float16 if model.precision=='fp16' else torch.bfloat16,
                            enabled=device.type=='cuda' and model.precision!='fp32'):
            features,_=model.native.features(x.to(device).float()/255)
            values.append(model.branch.context(features).cpu())
    return torch.cat(values)


@torch.no_grad()
def run(args):
    device=torch.device(args.device);model=restore(args.checkpoint,device)
    if model.branch is None or model.variant=='NoGate': raise ValueError('A gated static model is required')
    m=load_manifest(args.manifest,args.data_root)
    x,y=load_partition(m['splits']['test'],m['data_root'])
    contexts=context_cache(model,x,device,args.batch)
    results=[]
    for kind in ('own','same','other','random'):
        for seed in ([42] if kind=='own' else [42,43,44]):
            ids=np.arange(len(y)) if kind=='own' else donor_map(y.numpy(),kind,seed)
            logits=[]
            for start in range(0,len(y),args.batch):
                rgb=x[start:start+args.batch].to(device).float()/255
                context=contexts[ids[start:start+args.batch]].to(device)
                logits.append(model(rgb,gate_context=context).cpu())
            scores=torch.cat(logits)
            results.append(dict(condition=kind,seed=seed,accuracy=100*(scores.argmax(1)==y).float().mean().item(),
                                cross_entropy=torch.nn.functional.cross_entropy(scores,y).item()))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','manifest','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--data-root');p.add_argument('--device',default='cuda:0');p.add_argument('--batch',type=int,default=32)
    run(p.parse_args())

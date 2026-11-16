"""One portable entry point for scratch, limited-label, frozen and ablation runs."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Experiment,load_imagenet,VARIANTS,BACKBONES
from data import load_manifest,training_data,specs_for
from e0kit.data import load_partition
from e0kit.training import mixup_plan,classification_sums,schedule
from input_pipeline import prefetched_batches


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save(path,value):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


@torch.no_grad()
def evaluate(model,images,labels,device,batch=32):
    model.eval();outs=[]
    for i in range(0,len(labels),batch):
        if model.frozen:
            # Original-image FFTs are evaluated one image at a time to bound memory.
            for image in images[i:i+batch]: outs.append(model([image.to(device).float()/255]).cpu())
        else: outs.append(model(images[i:i+batch].to(device).float()/255).cpu())
    logits=torch.cat(outs)
    if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite logits')
    return dict(accuracy=100*(logits.argmax(1)==labels).float().mean().item(),
                loss=F.cross_entropy(logits,labels).item()),logits


from contextlib import contextmanager


@contextmanager
def epoch_batches(model,manifest,images,labels,order,epoch,seed,device):
    if model.frozen:
        flips=np.random.default_rng(np.random.SeedSequence([seed,epoch,28171])).random(len(images))<.5
        def batches():
            for step,ids in enumerate(order.split(32),1):
                plan=mixup_plan(epoch,step,len(ids),seed)
                clean=[images[i].to(device).float()/255 for i in ids.tolist()]
                clean=[x.flip((-1,)) if flips[i] else x for i,x in zip(ids.tolist(),clean)]
                mixed=[]
                for i,anchor in enumerate(clean):
                    partner=clean[plan['permutation'][i]]
                    if plan['applied']:
                        partner=F.interpolate(partner[None],size=anchor.shape[-2:],mode='bilinear',align_corners=False,antialias=True)[0]
                        anchor=plan['lam']*anchor+(1-plan['lam'])*partner
                    mixed.append(anchor)
                target=labels[ids].to(device)
                yield mixed,target,target[plan['permutation']] if plan['applied'] else None,plan['lam']
        yield batches()
    else:
        plans=[];prepared=[]
        for step,ids in enumerate(order.split(32),1):
            plan=mixup_plan(epoch,step,len(ids),seed)
            prepared.append(torch.cat((ids,ids[torch.tensor(plan['permutation'])])) if plan['applied'] else ids)
            plans.append((len(ids),plan))
        with prefetched_batches(images,labels,prepared,specs_for(manifest,images,epoch,seed),device) as iterator:
            def batches():
                for (_,rgb,target),(n,plan) in zip(iterator,plans):
                    partner=target[n:] if plan['applied'] else None
                    rgb=plan['lam']*rgb[:n]+(1-plan['lam'])*rgb[n:] if plan['applied'] else rgb
                    yield rgb[:n],target[:n],partner,plan['lam']
            yield batches()


def main(args):
    if args.frozen != bool(args.imagenet): raise ValueError('--frozen and --imagenet must be used together')
    if args.epochs!=200: raise ValueError('The paper schedule requires 200 maximum epochs')
    if args.patience<1: raise ValueError('Patience must be positive')
    if (Path(args.output)/'metrics.json').exists():
        if args.resume: return
        raise FileExistsError('Completed run; choose a new output directory')
    device=torch.device(args.device)
    if device.type!='cuda': raise ValueError('Training uses CUDA; CPU is supported for verification/inference')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_all(args.seed)
    manifest=load_manifest(args.manifest,args.data_root)
    model=Experiment(args.backbone,len(manifest['classes']),args.variant,args.width,args.precision,args.frozen)
    if args.frozen:
        meta=json.loads(Path(args.imagenet_metadata).read_text())[args.backbone]
        if meta.get('sha256') and hashlib.sha256(Path(args.imagenet).read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('ImageNet weight SHA256 differs from the archived source')
        load_imagenet(model,args.imagenet,meta['mean'],meta['std'])
    model.to(device)
    x,y,ids=training_data(manifest,args.shots,args.seed)
    vx,vy=load_partition(manifest['splits']['validation'],manifest['data_root'],training=args.frozen)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    contract=dict(spec=model.spec,seed=args.seed,shots=args.shots,train_ids=ids,
        splits=manifest['splits'],classes=manifest['classes'],dataset=manifest['dataset'],
        patience=args.patience,source_sha256={str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.rglob('*.py') if '__pycache__' not in p.parts},
        imagenet_sha256=hashlib.sha256(Path(args.imagenet).read_bytes()).hexdigest() if args.imagenet else None)
    identity=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    optimizer=torch.optim.AdamW(model.optimizer_groups())
    scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='fp16',init_scale=4096.)
    history=[];best=-1.;best_epoch=0;best_state=None
    if args.resume:
        checkpoint=torch.load(output/'last.pt',map_location=device,weights_only=False)
        if checkpoint['identity']!=identity: raise ValueError('Resume configuration mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scaler.load_state_dict(checkpoint['scaler'])
        history,best,best_epoch,best_state=[checkpoint[k] for k in ('history','best','best_epoch','best_state')]
    elif (output/'last.pt').exists(): raise FileExistsError('Use a new output directory or --resume')
    for epoch in range(len(history)+1,201):
        if history and len(history)-best_epoch>=args.patience: break
        seed_all(args.seed+epoch*100003);model.train();schedule(optimizer,epoch)
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(args.seed+epoch*(1009 if args.frozen else 1)))
        skipped=0
        with epoch_batches(model,manifest,x,y,order,epoch,args.seed,device) as iterator:
            for rgb,target,partner,lam in iterator:
                optimizer.zero_grad(set_to_none=True)
                n=len(target)
                for start in range(0,n,1 if args.frozen else n):
                    stop=start+(1 if args.frozen else n)
                    scores=model(rgb[start:stop])
                    loss=classification_sums(scores,target[start:stop],None if partner is None else partner[start:stop],lam)[0]/n
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                skipped+=int(scaler.get_scale()<scale)
        metrics,_=evaluate(model,vx,vy,device)
        if metrics['accuracy']>best:
            best,best_epoch=metrics['accuracy'],epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save(output/'best.pt',dict(spec=model.spec,model=best_state,epoch=epoch,contract=contract))
        history.append(dict(epoch=epoch,validation=metrics,best_validation=best,amp_skips=skipped))
        save(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
             history=history,best=best,best_epoch=best_epoch,best_state=best_state,identity=identity))
        print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(best_state)
    tx,ty=load_partition(manifest['splits']['test'],manifest['data_root'],training=args.frozen)
    metrics,logits=evaluate(model,tx,ty,device)
    np.savez_compressed(output/'predictions.npz',logits=logits.numpy(),labels=ty.numpy())
    (output/'metrics.json').write_text(json.dumps(dict(test=metrics,best_epoch=best_epoch,history=history),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--data-root')
    p.add_argument('--backbone',choices=BACKBONES,required=True);p.add_argument('--variant',choices=VARIANTS,default='FullStatic')
    p.add_argument('--width',type=int);p.add_argument('--shots',type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--precision',choices=['fp16','bf16','fp32'],default='fp16')
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=25)
    p.add_argument('--frozen',action='store_true');p.add_argument('--imagenet')
    p.add_argument('--imagenet-metadata',default='configs/imagenet.json')
    p.add_argument('--resume',action='store_true');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    main(p.parse_args())

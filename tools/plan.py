"""Generate portable commands for main-text experiment families; never launch jobs."""
import argparse,itertools,json,shlex
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import BACKBONES


def jobs(suite):
    if suite=='main':
        for ds,b,v in itertools.product(['AID','Flowers102','RSSCN7']+['KTHb_f'+str(i) for i in range(1,5)],BACKBONES,['Native','FullStatic']):
            yield dict(dataset=ds,backbone=b,variant=v,width=BACKBONES[b])
    elif suite=='limited':
        budgets={'AID':[5,10,20,40],'Flowers17':[5,10,20,40],
                 'Flowers102':[5,10],**{'KTHb_f'+str(i):[5,10,20,30,40] for i in range(1,5)}}
        for ds,ks in budgets.items():
            for b,k,v in itertools.product(['edgenext_small','shufflenet_v2_x1_0','convnext_tiny','efficientnet_b0','mambaout_femto'],ks,['Native','FullStatic']):
                yield dict(dataset=ds,backbone=b,variant=v,width=BACKBONES[b],shots=k)
    elif suite=='frozen':
        for ds,b,v in itertools.product(['AID','RSSCN7'],BACKBONES,['Native','FullStatic']):
            yield dict(dataset=ds,backbone=b,variant=v,width=128,frozen=True)
    elif suite=='gates':
        for ds,b in itertools.product(['Flowers102','RSSCN7'],BACKBONES):
            yield dict(dataset=ds,backbone=b,variant='FullStatic',width=128)
    elif suite=='orders':
        for ds,b,v in itertools.product(['Flowers102','RSSCN7'],BACKBONES,['FixedBanks','Shared','FullStatic']):
            yield dict(dataset=ds,backbone=b,variant=v,width=128)
    elif suite=='components':
        for ds,b,v in itertools.product(['Flowers102','RSSCN7'],['edgenext_small','mobilevitv2_100'],
                   ['FirstOrderOnly','SecondOrderOnly','NoGate','FixedAlpha','FullStatic']):
            yield dict(dataset=ds,backbone=b,variant=v,width=128)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',choices=['main','limited','frozen','orders','components','gates'])
    p.add_argument('--manifest-dir',default='inputs');p.add_argument('--output-root',default='runs');p.add_argument('--weights-dir',default='weights')
    a=p.parse_args()
    for j in jobs(a.suite):
        name='_'.join(str(j.get(k,'full')) for k in ['dataset','backbone','variant','shots'])
        if a.suite=='gates':
            checkpoint=f'{a.output_root}/orders/{j["dataset"]}_{j["backbone"]}_FullStatic_full/best.pt'
            cmd=['python','interventions.py','--checkpoint',checkpoint,'--manifest',f'{a.manifest_dir}/{j["dataset"]}.json','--output',f'{a.output_root}/gates/{name}.json']
            print(shlex.join(cmd));continue
        cmd=['python','train.py','--manifest',f'{a.manifest_dir}/{j["dataset"]}.json','--backbone',j['backbone'],
             '--variant',j['variant'],'--width',str(j['width']),'--output',f'{a.output_root}/{a.suite}/{name}']
        if 'shots' in j:cmd+=['--shots',str(j['shots'])]
        if j.get('frozen'):
            metadata=json.loads((Path(__file__).resolve().parents[1]/'configs/imagenet.json').read_text())
            extension=metadata[j['backbone']]['extension']
            cmd+=['--frozen','--imagenet',f'{a.weights_dir}/{j["backbone"]}{extension}']
        print(shlex.join(cmd))

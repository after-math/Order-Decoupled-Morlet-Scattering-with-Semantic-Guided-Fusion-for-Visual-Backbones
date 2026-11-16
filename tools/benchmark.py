"""Static implementation parameter counts, explicit-FFT FLOPs and CUDA latency."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import Experiment,BACKBONES
from flop_counter import Counter

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backbone',choices=BACKBONES,default='edgenext_small')
    p.add_argument('--width',type=int,default=128);p.add_argument('--device',default='cpu');p.add_argument('--classes',type=int,default=7)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--iterations',type=int,default=30)
    a=p.parse_args();torch.set_num_threads(1)
    for variant in ['Native','FullStatic']:
        torch.manual_seed(42);m=Experiment(a.backbone,a.classes,variant,a.width,precision='fp32').eval()
        x=torch.rand(1,3,256,256);counter=Counter()
        with torch.inference_mode(),counter.model_scope(m),counter:m(x)
        row=dict(variant=variant,parameters=sum(p.numel() for p in m.parameters()),flops=sum(counter.categories.values()),categories=dict(counter.categories))
        if a.device.startswith('cuda'):
            m.to(a.device);x=x.to(a.device)
            with torch.inference_mode():
                for _ in range(a.warmup):m(x)
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(a.iterations):m(x)
                torch.cuda.synchronize()
            row['latency_ms']=(time.perf_counter()-t)*1000/a.iterations
        print(json.dumps(row))

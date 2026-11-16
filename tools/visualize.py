"""Static spectra, second-order coefficients, gate maps and signed contributions."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use(['science','no-latex','grid',])
from interventions import restore,donor_map
from data import load_manifest
from e0kit.data import load_partition
from e0kit.scattering import MorletKernels
from torch.nn import functional as F


def integrated_contribution(model,final,own,other,label,steps=64):
    difference=own-other;gradient=torch.zeros_like(own)
    for fraction in (torch.arange(steps,device=own.device)+.5)/steps:
        z=(other+fraction*difference).detach().requires_grad_(True)
        residual=model.branch.output(F.interpolate(z,size=final.shape[-2:],mode='bilinear',align_corners=False))
        score=model.native.head(final+model.alpha*residual).log_softmax(1)[0,label]
        gradient+=torch.autograd.grad(score,z)[0]/steps
    return (difference*gradient).sum(1)[0].detach().cpu().numpy()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['checkpoint','manifest','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--data-root');p.add_argument('--sample',type=int,required=True);p.add_argument('--kernel',type=int,default=15)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    m=load_manifest(a.manifest,a.data_root);device=torch.device(a.device);model=restore(a.checkpoint,device)
    if model.branch is None or model.variant=='NoGate':raise ValueError('A gated static model is required')
    labels=[r['label'] for r in m['splits']['test']];donors=donor_map(labels,'other',42)
    rows=[m['splits']['test'][i] for i in [a.sample,int(donors[a.sample])]]
    images,_=load_partition(rows,m['data_root']);rgb=images.to(device).float()/255;label=labels[a.sample]
    with torch.no_grad():
        f,final=model.native.features(rgb[:1]);df,_=model.native.features(rgb[1:])
        donor_context=model.branch.context(df)
        own,info=model(rgb[:1],details=True)
        other,wrong=model(rgb[:1],gate_context=donor_context,details=True)
        parameters=model.branch.parameters_by_order()
        kernels=MorletKernels().to(device)
        indices=[model.branch.shared_indices[a.kernel],a.kernel]
        spectra=[torch.fft.fftshift(torch.fft.fft2(kernels(v)[k],s=(256,256))).abs().cpu().numpy() for v,k in zip(parameters,indices)]
        texture=info['texture'];own_z=(info['gate']*texture).detach();other_z=(wrong['gate']*texture).detach()
    signed=integrated_contribution(model,final.detach(),own_z,other_z,label)
    fig,axes=plt.subplots(1,6,figsize=(16,3))
    axes[0].imshow(rgb[0].permute(1,2,0).cpu());axes[0].set_title('Target')
    axes[1].imshow(spectra[1],cmap='Greens',origin='lower')
    for spectrum,style in zip(spectra,['solid','dashed']):
        axes[1].contour(spectrum,levels=[.5*spectrum.max()],colors=['green'],linestyles=style)
    axes[1].set_title('Static order spectra')
    response=info['second'][0].reshape(3,108,32,32).mean((0,1)).log1p().cpu()
    axes[2].imshow(response,cmap='magma');axes[2].set_title('Second-order response')
    for ax,key,title in [(axes[3],info,'Own gate'),(axes[4],wrong,'Other gate')]:
        ax.imshow(key['gate'][0].mean(0).cpu(),cmap='viridis',vmin=0,vmax=1);ax.set_title(title)
    limit=max(abs(signed).max(),1e-12)
    axes[5].imshow(signed,cmap='RdBu_r',vmin=-limit,vmax=limit);axes[5].set_title('Class contribution')
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'Own p_y={own.softmax(1)[0,label].item():.3f}; other p_y={other.softmax(1)[0,label].item():.3f}')
    fig.tight_layout();Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,bbox_inches='tight');plt.close(fig)

"""Compare separately trained static shared and independent-bank checkpoints."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
plt.style.use(['science','no-latex','grid',])
from interventions import restore
from data import load_manifest
from e0kit.data import load_partition
from e0kit.scattering import MorletKernels

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['shared','separate','manifest','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--sample',type=int,required=True);p.add_argument('--path',type=int,default=99)
    p.add_argument('--data-root');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    m=load_manifest(a.manifest,a.data_root);image,_=load_partition([m['splits']['test'][a.sample]],m['data_root'])
    device=torch.device(a.device);models=[restore(a.shared,device),restore(a.separate,device)]
    if [m.variant for m in models]!=['Shared','FullStatic']:raise ValueError('Expected Shared and FullStatic checkpoints')
    if models[0].spec['backbone']!=models[1].spec['backbone']:raise ValueError('Use the same backbone')
    fig,axes=plt.subplots(1,5,figsize=(12,3));axes[0].imshow(image[0].permute(1,2,0));axes[0].set_title('Target')
    responses=[]
    with torch.no_grad():
        for index,model in enumerate(models):
            q1,q2=model.branch.parameters_by_order();k2=int(model.branch.scattering.second_ids[a.path]);k1=model.branch.shared_indices[k2]
            kernels=MorletKernels().to(device)
            spectra=[torch.fft.fftshift(torch.fft.fft2(kernels(q)[k],s=(256,256))).abs().cpu() for q,k in [(q1,k1),(q2,k2)]]
            axes[1+index].imshow(spectra[1],cmap='Greens',origin='lower')
            for spectrum,style in zip(spectra,['solid','dashed']):axes[1+index].contour(spectrum,levels=[.5*spectrum.max()],colors=['green'],linestyles=style)
            axes[1+index].set_title(['Shared filters','Separate filters'][index])
            _,details=model(image.to(device).float()/255,details=True)
            responses.append(details['second'][0].reshape(3,108,32,32)[:,a.path].log1p().mean(0).cpu())
    limit=max(x.max() for x in responses)
    for ax,response,title in zip(axes[3:],responses,['Shared response','Separate response']):ax.imshow(response,cmap='magma',vmin=0,vmax=limit);ax.set_title(title)
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,bbox_inches='tight');plt.close(fig)

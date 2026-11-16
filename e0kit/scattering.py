"""Shared finite complex Morlet + full-resolution order1/2 + real EdgeNeXt-Small."""
import math
import torch
import torch.nn.functional as F
from torch import nn

def reference_parameters():
    j=torch.arange(3,dtype=torch.float32).repeat_interleave(6)
    theta=torch.arange(6,dtype=torch.float32).repeat(3)*(math.pi/6)
    xi=(3*math.pi/4)*2.**(-j)
    # Use the identical float32 kappa/xi expression in all three groups.
    sigma=torch.full_like(xi,.6*math.pi)/xi
    return torch.stack([sigma,theta,xi,torch.full_like(xi,2/3)],-1)

def anchored_sigmoid(raw,reference,lo,hi):
    p=(reference-lo)/(hi-lo)
    anchor=torch.log(p/(1-p))
    return reference+(hi-lo)*(torch.sigmoid(anchor+raw)-torch.sigmoid(anchor))

class E0Bank(nn.Module):
    """Exactly72 independent controls; raw theta/xi/sigma/gamma, physical sigma/theta/xi/gamma."""
    def __init__(self):
        super().__init__()
        self.register_buffer('reference',reference_parameters())
        self.local=nn.Parameter(torch.zeros(18,4))
        self.group=None
    def forward(self):
        r=self.reference;u=self.local
        theta=r[:,1]+u[:,0]
        xi=anchored_sigmoid(u[:,1],r[:,2],.1,2.9)
        sigma=anchored_sigmoid(u[:,2],r[:,0],.3,8.)
        gamma=anchored_sigmoid(u[:,3],r[:,3],.3,1.5)
        return torch.stack([sigma,theta,xi,gamma],-1)

class MorletKernels(nn.Module):
    def __init__(self):
        super().__init__()
        a=torch.arange(-28,29,dtype=torch.float32)
        y,x=torch.meshgrid(a,a,indexing='ij')
        self.register_buffer('x',x,persistent=False);self.register_buffer('y',y,persistent=False)

    def forward(self,q):
        sigma,theta,xi,gamma=[x[:,None,None] for x in q.unbind(-1)]
        p=self.x*theta.cos()+self.y*theta.sin()
        r=-self.x*theta.sin()+self.y*theta.cos()
        g=torch.exp(-(p.square()+(gamma*r).square())/(2*sigma.square()))
        carrier=torch.polar(torch.ones_like(p),xi*p)
        beta=(g*carrier).sum((-2,-1),keepdim=True)/g.sum((-2,-1),keepdim=True)
        psi=g*(carrier-beta);psi=psi-psi.mean((-2,-1),keepdim=True)
        return psi/(psi.abs().square().sum((-2,-1),keepdim=True)+1e-12).sqrt()

def kernel_fft(k,size):
    # Coordinates -28..28: top-left embedding then roll(-28), not ifftshift57
    # before padding. This is the DFT's exact modular spatial-origin convention.
    canvas=F.pad(k,(0,size-57,0,size-57))
    return torch.fft.fft2(torch.roll(canvas,(-28,-28),(-2,-1)),norm='backward')

class FullResolutionScattering(nn.Module):
    def __init__(self,second_order_chunk=18):
        super().__init__()
        x=torch.arange(-13,14,dtype=torch.float32)
        lp=torch.exp(-x.square()/(2*3.2**2));lp/=lp.sum()
        self.register_buffer('lowpass',lp,persistent=False)
        self.second_order_chunk=int(second_order_chunk)
        assert 1<=self.second_order_chunk<=108
        j=torch.arange(18)//6
        pairs=[(a,b) for a in range(18) for b in range(18) if j[a]<j[b]]
        self.register_buffer('first_ids',torch.tensor([a for a,b in pairs]),persistent=False)
        self.register_buffer('second_ids',torch.tensor([b for a,b in pairs]),persistent=False)

    def lowpass_downsample(self,u):
        shape=u.shape[:-2];x=u.reshape(-1,1,256,256)
        # Exact separable Gaussian smoothing then stride8 in both coordinates.
        # Moving each axis' subsampling before the other axis' filtering is exact.
        x=F.conv2d(F.pad(x,(0,0,13,13),mode='reflect'),self.lowpass.reshape(1,1,27,1),stride=(8,1))
        x=F.conv2d(F.pad(x,(13,13,0,0),mode='reflect'),self.lowpass.reshape(1,1,1,27),stride=(1,8))
        return x.reshape(*shape,32,32)

    def padded_fft(self,x):
        # Reflect halo28 provides every sample needed for valid same convolution.
        # 320 FFT avoids prime-factor13 in312; extra zeros do not enter valid crop.
        x=F.pad(x,(28,28,28,28),mode='reflect')
        return torch.fft.fft2(x,s=(320,320),norm='backward')

    def forward(self,raw,kernels):
        n=raw.shape[0]
        x=raw.reshape(n*3,256,256)
        filters=kernel_fft(kernels,320)
        z=torch.fft.ifft2(self.padded_fft(x)[:,None]*filters[None],norm='backward')
        u1=z[...,28:284,28:284].abs()
        s1=self.lowpass_downsample(u1).reshape(n,3*18,32,32)
        # Unaveraged, unstrided U1, not S1, is the second-order input.
        u1_fft=self.padded_fft(u1.reshape(n*3*18,256,256)).reshape(n*3,18,320,320)
        output=[]
        for begin in range(0,108,self.second_order_chunk):
            a=self.first_ids[begin:begin+self.second_order_chunk]
            b=self.second_ids[begin:begin+self.second_order_chunk]
            z2=torch.fft.ifft2(u1_fft[:,a]*filters[b][None],norm='backward')
            u2=z2[...,28:284,28:284].abs()
            output.append(self.lowpass_downsample(u2))
        s2=torch.cat(output,dim=1).reshape(n,3*108,32,32)
        desc=s1.reshape(n,3,18,32,32).mean((1,3,4))
        return s1,s2,desc

class ChannelNorm(nn.Module):
    def __init__(self,c):super().__init__();self.norm=nn.LayerNorm(c,eps=1e-6)
    def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

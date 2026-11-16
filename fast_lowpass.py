"""FP32 separable 27-tap Gaussian + reflect boundary + stride8.

The two passes and their exact linear adjoints are fused CUDA kernels. Filter
coefficients are passed from the reference module; no kernel truncation or
precision reduction. This module does not generate or cache Morlet filters.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, W, Y, SIZE:tl.constexpr, VERTICAL:tl.constexpr, BLOCK:tl.constexpr):
    ids=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=ids<SIZE
    if VERTICAL:
        channel=ids//8192
        row=ids//256%32
        col=ids%256
    else:
        channel=ids//1024
        row=ids//32%32
        col=ids%32
    value=tl.full((BLOCK,),0,tl.float32)
    for k in tl.static_range(27):
        if VERTICAL:
            coordinate=row*8+k-13
        else:
            coordinate=col*8+k-13
        coordinate=tl.where(coordinate<0,-coordinate,coordinate)
        coordinate=tl.where(coordinate>255,510-coordinate,coordinate)
        if VERTICAL:
            offset=channel*65536+coordinate*256+col
        else:
            offset=channel*8192+row*256+coordinate
        w=tl.load(W+k)
        value+=tl.load(X+offset,valid,other=0)*w
    tl.store(Y+ids,value,valid)


@triton.jit
def _backward(DY,W,DX,SIZE:tl.constexpr,VERTICAL:tl.constexpr,BLOCK:tl.constexpr):
    ids=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=ids<SIZE
    if VERTICAL:
        channel=ids//65536
        row=ids//256%256
        col=ids%256
        coordinate=row
    else:
        channel=ids//8192
        row=ids//256%32
        col=ids%256
        coordinate=col
    value=tl.full((BLOCK,),0,tl.float32)
    for reflection in tl.static_range(3):
        if reflection==0:
            unreflected=coordinate
            enabled=valid
        elif reflection==1:
            unreflected=-coordinate
            enabled=valid&(coordinate>0)&(coordinate<=13)
        else:
            unreflected=510-coordinate
            enabled=valid&(coordinate<255)&(coordinate>=249)
        for shift in tl.static_range(-1,3):
            out=unreflected//8+shift
            tap=unreflected-out*8+13
            mask=enabled&(out>=0)&(out<32)&(tap>=0)&(tap<27)
            if VERTICAL:
                offset=channel*8192+out*256+col
            else:
                offset=channel*1024+row*32+out
            value+=tl.load(DY+offset,mask,other=0)*tl.load(W+tap,mask,other=0)
    tl.store(DX+ids,value,valid)


class _GaussianStride(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weights):
        assert x.is_cuda and x.dtype==torch.float32 and x.shape[-2:]==(256,256)
        assert weights.shape==(27,) and not weights.requires_grad
        x=x.contiguous()
        shape=x.shape[:-2]
        vertical=torch.empty((*shape,32,256),device=x.device,dtype=x.dtype)
        output=torch.empty((*shape,32,32),device=x.device,dtype=x.dtype)
        _forward[(triton.cdiv(vertical.numel(),256),)](x,weights,vertical,vertical.numel(),True,256,enable_fp_fusion=False)
        _forward[(triton.cdiv(output.numel(),256),)](vertical,weights,output,output.numel(),False,256,enable_fp_fusion=False)
        ctx.save_for_backward(weights)
        ctx.shape=shape
        return output

    @staticmethod
    def backward(ctx,grad):
        weights,=ctx.saved_tensors
        grad=grad.contiguous()
        horizontal=torch.empty((*ctx.shape,32,256),device=grad.device,dtype=grad.dtype)
        result=torch.empty((*ctx.shape,256,256),device=grad.device,dtype=grad.dtype)
        _backward[(triton.cdiv(horizontal.numel(),256),)](grad,weights,horizontal,horizontal.numel(),False,256,enable_fp_fusion=False)
        _backward[(triton.cdiv(result.numel(),256),)](horizontal,weights,result,result.numel(),True,256,enable_fp_fusion=False)
        return result,None


def gaussian_stride(x,weights):
    return _GaussianStride.apply(x,weights)


@triton.jit
def _magnitude_forward(X,Y,SIZE:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,BLOCK:tl.constexpr):
    ids=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=ids<SIZE
    channel=ids//65536
    row=ids//256%256
    col=ids%256
    src=channel*102400+(row+28)*SY+(col+28)*SX
    re=tl.load(X+src*2,mask,other=0)
    im=tl.load(X+src*2+1,mask,other=0)
    tl.store(Y+ids,tl.sqrt(re*re+im*im),mask)


@triton.jit
def _magnitude_backward(X,DY,DX,SIZE:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,BLOCK:tl.constexpr):
    ids=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=ids<SIZE
    channel=ids//102400
    row=ids//320%320
    col=ids%320
    crop=mask&(row>=28)&(row<284)&(col>=28)&(col<284)
    src=channel*102400+row*SY+col*SX
    re=tl.load(X+src*2,crop,other=0)
    im=tl.load(X+src*2+1,crop,other=0)
    magnitude=tl.sqrt(re*re+im*im)
    grad=tl.load(DY+channel*65536+(row-28)*256+col-28,crop,other=0)
    denominator=tl.where(magnitude>0,magnitude,1.)
    tl.store(DX+ids*2,grad*(re/denominator),mask)
    tl.store(DX+ids*2+1,grad*(im/denominator),mask)


class _MagnitudeCrop(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        assert x.dtype==torch.complex64 and x.shape[-2:]==(320,320)
        # cuFFT may return either spatial stride order. Leading dimensions are dense.
        assert x.stride(-2)*x.stride(-1)==320
        y=torch.empty((*x.shape[:-2],256,256),device=x.device,dtype=torch.float32)
        _magnitude_forward[(triton.cdiv(y.numel(),256),)](torch.view_as_real(x),y,y.numel(),x.stride(-2),x.stride(-1),256,enable_fp_fusion=False)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx,grad):
        x,=ctx.saved_tensors
        grad=grad.contiguous()
        dx=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        _magnitude_backward[(triton.cdiv(dx.numel(),256),)](torch.view_as_real(x),grad,torch.view_as_real(dx),dx.numel(),x.stride(-2),x.stride(-1),256,enable_fp_fusion=False)
        return dx


def magnitude_crop(x):
    return _MagnitudeCrop.apply(x)

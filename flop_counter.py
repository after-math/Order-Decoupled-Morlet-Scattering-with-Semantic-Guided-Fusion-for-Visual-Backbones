"""Forward operator accounting; MAC=2, complex FFT=5P log2(P)."""
import collections,contextlib,math
import torch,timm,torchvision
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


ZERO={'alias','arange','bitwise_not','cat','chunk','clamp','clamp_min','contiguous','detach','detach_','dropout','empty','empty_like','empty_strided','expand','expand_as','flatten','full','full_like','index','index_select','lift_fresh','max_pool2d','max_pool2d_with_indices','ones','ones_like','pad','permute','rand','relu','relu_','reshape','reflection_pad2d','replication_pad2d','constant_pad_nd','resize_','roll','select','slice','split','split_with_sizes','squeeze','stack','t','to','transpose','unbind','unsqueeze','view','view_as_real','view_as_complex','zeros','zeros_like','_to_copy','copy_','clone','split_copy','resolve_conj','_conj','conj_physical','is_nonzero','_local_scalar_dense','detach_copy'}

def first_tensor(value):
 return next((x for x in tree_flatten(value)[0] if isinstance(x,torch.Tensor)),None)
def product(shape):return math.prod(tuple(int(x) for x in shape))
def floating(x):return isinstance(x,torch.Tensor) and (x.is_floating_point() or x.is_complex())
def complex_value(x):return isinstance(x,torch.Tensor) and x.is_complex()
def optional(args,kwargs,index,name,default=None):return args[index] if len(args)>index else kwargs.get(name,default)

class Counter(TorchDispatchMode):
 def __init__(self):
  super().__init__();self.ops={};self.categories=collections.defaultdict(float);self.stack=[];self.in_fft=0;self.fft_events=[]
 def add(self,name,cost,category,scope,metadata=None):
  row=self.ops.setdefault((name,scope),dict(operator=name,scope=scope,calls=0,estimated_flops=0.,categories=collections.defaultdict(float)))
  row['calls']+=1;row['estimated_flops']+=cost;row['categories'][category]+=cost;self.categories[category]+=cost
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {};out=func(*args,**kwargs)
  name=func._schema.name.split('::')[-1];scope=self.stack[-1] if self.stack else '<root>'
  if self.in_fft:
   self.add(str(func),0,'fft_internal_counted_once',scope);return out
  value=first_tensor(out)
  if value is None or not floating(value):
   self.add(str(func),0,'nonfloating_or_metadata',scope);return out
  E=value.numel();x=first_tensor(args)
  if name in ZERO or name.startswith('_fft_'):
   cost=0;category='memory_index_comparison_or_fft_internal'
  elif name in {'convolution','_convolution','conv2d','conv1d','conv3d'}:
   weight=args[1];transposed=optional(args,kwargs,6,'transposed',False) if name in {'convolution','_convolution'} else False
   assert not transposed,'No transposed convolution expected in forward'
   cost=2*E*int(weight.shape[1])*product(weight.shape[2:]);category='convolution_matmul'
   if len(args)>2 and args[2] is not None:cost+=E
   if 'scattering' in scope and product(weight.shape[2:])==27:category='lowpass'
  elif name=='linear':
   cost=2*E*args[1].shape[-1]+(E if optional(args,kwargs,2,'bias') is not None else 0);category='convolution_matmul'
  elif name in {'matmul','mm','bmm','addmm','baddbmm'}:
   a,b=(args[1],args[2]) if name in {'addmm','baddbmm'} else (args[0],args[1])
   K=a.shape[-1];cost=2*E*K;category='convolution_matmul'
   if name in {'addmm','baddbmm'}:
    if kwargs.get('beta',1)!=0:cost+=E
    if kwargs.get('alpha',1)!=1:cost+=E
  elif 'scaled_dot_product' in name:
   q,k,v=args[:3];batch=product(torch.broadcast_shapes(q.shape[:-2],k.shape[:-2],v.shape[:-2]))
   scores=batch*q.shape[-2]*k.shape[-2];rows=batch*q.shape[-2]
   cost=2*scores*(q.shape[-1]+v.shape[-1])+scores+4*scores-rows;category='attention'
  elif name in {'layer_norm','native_layer_norm'}:
   shape=tuple(args[1]);rows=x.numel()//product(shape)
   affine=int(optional(args,kwargs,2,'weight') is not None)+int(optional(args,kwargs,3,'bias') is not None)
   cost=(6+affine)*x.numel()+2*rows;category='normalization'
  elif name in {'group_norm','native_group_norm'}:
   native=name=='native_group_norm'
   groups=int(args[6] if native else args[1]);w=args[1] if native else optional(args,kwargs,2,'weight');b=args[2] if native else optional(args,kwargs,3,'bias')
   cost=(6+int(w is not None)+int(b is not None))*x.numel()+2*x.shape[0]*groups;category='normalization'
  elif 'batch_norm' in name:
   if 'no_training' not in name:assert not bool(optional(args,kwargs,5,'training',False))
   affine=int(optional(args,kwargs,1,'weight') is not None)+int(optional(args,kwargs,2,'bias') is not None)
   cost=(2+affine)*x.numel()+2*x.shape[1];category='normalization'
  elif name in {'sum','mean','cumsum'}:
   components=2 if x.is_complex() else 1
   if name=='cumsum':cost=components*(x.numel()-x.numel()//x.shape[int(args[1])])
   else:cost=components*(x.numel()-E+(E if name=='mean' else 0))
   category='reduction'
  elif name in {'adaptive_avg_pool2d','_adaptive_avg_pool2d','avg_pool2d'}:
   if name=='avg_pool2d':
    kernel=args[1];kernel=(kernel,kernel) if isinstance(kernel,int) else kernel;cost=E*product(kernel)
   else:
    H,W=x.shape[-2:];h,w=value.shape[-2:]
    hh=sum(math.ceil((i+1)*H/h)-math.floor(i*H/h) for i in range(h))
    ww=sum(math.ceil((i+1)*W/w)-math.floor(i*W/w) for i in range(w))
    cost=product(x.shape[:-2])*hh*ww
   category='pooling'
  elif name.startswith('upsample_bilinear'):
   cost=7*E;category='interpolation'
  elif name in {'softmax','_softmax','_safe_softmax'}:
   rows=E//x.shape[int(args[1])];cost=4*E-rows;category='nonlinear'
  elif name in {'add','add_','sub','rsub','mul','mul_','div','div_','neg','square','pow','reciprocal','abs','sqrt','rsqrt','sin','cos','exp','log','log1p','sigmoid','silu','silu_','gelu','tanh'}:
   category='elementwise'
   if name in {'add','add_','sub','rsub'}:
    cost=E*(2 if value.is_complex() and complex_value(args[0]) and len(args)>1 and complex_value(args[1]) else 1)
    if kwargs.get('alpha',1)!=1:cost+=E*(2 if value.is_complex() else 1)
   elif name in {'mul','mul_'}:
    cost=E*(6 if complex_value(args[0]) and complex_value(args[1]) else (2 if value.is_complex() else 1))
    if value.is_complex() and value.shape[-2:]==(320,320) and 'scattering' in scope:category='spectral_product'
   elif name in {'div','div_'}:
    cost=E*(11 if len(args)>1 and complex_value(args[1]) else (2 if value.is_complex() else 1))
    if kwargs.get('rounding_mode') is not None:cost+=E
   elif name=='abs':
    cost=E*(4 if complex_value(x) else 0)
    if complex_value(x) and value.shape[-2:]==(256,256) and 'scattering' in scope:category='scattering_modulus'
   elif name=='square':cost=E*(5 if value.is_complex() else 1)
   elif name=='sigmoid':cost=4*E;category='nonlinear'
   elif name in {'silu','silu_'}:cost=5*E;category='nonlinear'
   elif name=='gelu':
    approx=optional(args,kwargs,1,'approximate','none');cost=(5 if approx=='none' else 9)*E;category='nonlinear'
   else:cost=E*(2 if value.is_complex() else 1)
  elif name=='linalg_vector_norm':
   p=optional(args,kwargs,1,'ord',2)
   assert float(p)==2
   cost=(4 if x.is_complex() else 2)*x.numel();category='normalization'
  elif name=='polar':cost=4*E;category='elementwise'
  else:
   raise RuntimeError('Uncounted floating operation '+str(func)+' '+str([getattr(a,'shape',None) for a in args]))
  if 'scattering.kernels' in scope:category='filter_synthesis'
  if scope.startswith('branch.bank') or scope.startswith('branch.second_bank'):category='parameter_decode'
  self.add(str(func),float(cost),category,scope)
  return out
 @contextlib.contextmanager
 def model_scope(self,model):
  handles=[]
  for name,module in model.named_modules():
   def enter(m,args,n=name):self.stack.append(n or '<root>')
   def leave(m,args,out,n=name):
    assert self.stack[-1]==(n or '<root>'),(self.stack,n)
    self.stack.pop()
   handles.extend([module.register_forward_pre_hook(enter),module.register_forward_hook(leave)])
  original_fft,original_ifft=torch.fft.fft2,torch.fft.ifft2
  def wrap(original,inverse):
   def run(*args,**kwargs):
    self.in_fft+=1
    try:out=original(*args,**kwargs)
    finally:self.in_fft-=1
    dims=optional(args,kwargs,2,'dim',(-2,-1));P=product(out.shape[d] for d in dims);maps=out.numel()//P
    norm=optional(args,kwargs,3,'norm','backward') or 'backward'
    scaled=norm=='ortho' or (inverse and norm=='backward') or (not inverse and norm=='forward')
    estimate=5*out.numel()*math.log2(P);normalization=2*out.numel() if scaled else 0
    scope=self.stack[-1] if self.stack else '<root>'
    self.add('full_complex_ifft2' if inverse else 'full_complex_fft2',estimate,'fft_estimate',scope)
    self.add('fft_normalization',normalization,'fft_normalization',scope)
    self.fft_events.append(dict(inverse=inverse,shape=list(out.shape),points_per_map=P,maps=maps,estimated_flops=estimate,normalization_flops=normalization,norm=norm))
    return out
   return run
  torch.fft.fft2=wrap(original_fft,False);torch.fft.ifft2=wrap(original_ifft,True)
  try:yield
  finally:
   torch.fft.fft2,torch.fft.ifft2=original_fft,original_ifft
   for h in handles:h.remove()


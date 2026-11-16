"""Static, order-decoupled Morlet scattering with semantic-conditioned fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from backbone_core import Native, WaveBranch as BaseBranch, align, BACKBONES

VARIANTS = ('Native', 'FullStatic', 'Shared', 'FixedBanks', 'FirstOrderOnly',
            'SecondOrderOnly', 'NoGate', 'FixedAlpha')


def classifier(native):
    if native.name in ('resnet18', 'shufflenet_v2_x1_0'):
        return native.body.fc
    if native.name == 'mambaout_femto':
        return native.body.head.fc2
    return native.body.get_classifier()


class WaveBranch(BaseBranch):
    def __init__(self, native, width, variant):
        super().__init__(native, width)
        self.variant = variant
        # Shared uses one bank and the original scale-transition mapping.
        self.register_buffer('shared_indices', torch.tensor(list(range(6,12))+list(range(12,18))*2))
        if variant == 'Shared':
            self.second_bank = None
        if variant == 'FixedBanks':
            self.bank.requires_grad_(False)
            self.second_bank.requires_grad_(False)
        if variant == 'FirstOrderOnly':
            self.second_bank.requires_grad_(False)
            self.texture2.requires_grad_(False)
        if variant == 'SecondOrderOnly':
            self.texture1.requires_grad_(False)
        if variant == 'NoGate':
            self.projections.requires_grad_(False)
            self.semantic_fuse.requires_grad_(False)
            self.gate.requires_grad_(False)

    def parameters_by_order(self):
        first = self.bank()
        second = first[self.shared_indices] if self.second_bank is None else self.second_bank()
        return first, second

    def context(self, features):
        return self.semantic_fuse(torch.cat([align(p(x)) for p,x in zip(self.projections,features)],1))

    def measure(self, rgb):
        with torch.autocast(rgb.device.type, enabled=False):
            first,second = self.scattering(rgb.float(), *self.parameters_by_order())
        a = self.texture1(F.adaptive_avg_pool2d(torch.log1p(first.clamp_min(0)), (16,16)))
        b = self.texture2(F.adaptive_avg_pool2d(torch.log1p(second.clamp_min(0)), (16,16)))
        if self.variant == 'FirstOrderOnly': b = torch.zeros_like(b)
        if self.variant == 'SecondOrderOnly': a = torch.zeros_like(a)
        return self.texture_fuse(torch.cat((a,b),1)), first, second

    def forward(self, rgb, features, final, gate_context=None, details=False):
        texture,first,second = self.measure(rgb)
        if self.variant == 'NoGate':
            gate = torch.ones_like(texture)
        else:
            context = self.context(features) if gate_context is None else gate_context
            gate = torch.sigmoid(self.gate(torch.cat((context,texture),1)))
        residual = self.output(F.interpolate(gate*texture,size=final.shape[-2:],mode='bilinear',align_corners=False))
        if details:
            return residual, dict(first=first,second=second,texture=texture,gate=gate)
        return residual


class Experiment(nn.Module):
    def __init__(self, backbone, classes, variant='FullStatic', width=None, precision='fp16', frozen=False):
        super().__init__()
        if backbone not in BACKBONES or variant not in VARIANTS:
            raise ValueError((backbone,variant))
        width = BACKBONES[backbone] if width is None else width
        if width < 4 or width % 4: raise ValueError('Width must be a positive multiple of four')
        self.spec = dict(backbone=backbone,classes=classes,variant=variant,width=width,precision=precision,frozen=frozen)
        self.variant,self.precision,self.frozen = variant,precision,frozen
        self.native = Native(backbone,classes)
        self.branch = None if variant == 'Native' else WaveBranch(self.native,width,variant)
        self.register_parameter('alpha',None if self.branch is None else nn.Parameter(torch.tensor(.1),requires_grad=variant!='FixedAlpha'))
        if frozen:
            if self.branch is not None:
                from static_original import StaticOriginalScattering
                self.branch.scattering = StaticOriginalScattering()
            self.freeze_native()

    def freeze_native(self):
        self.native.requires_grad_(False)
        classifier(self.native).requires_grad_(True)
        self.native.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen: self.native.eval()  # Includes frozen BN running statistics.
        return self

    def forward(self, rgb, gate_context=None, details=False):
        device_type = rgb[0].device.type if isinstance(rgb,(list,tuple)) else rgb.device.type
        is_cuda = rgb[0].is_cuda if isinstance(rgb,(list,tuple)) else rgb.is_cuda
        with torch.autocast(device_type,dtype=torch.float16 if self.precision=='fp16' else torch.bfloat16,
                            enabled=is_cuda and self.precision!='fp32'):
            if self.frozen:
                from full_frame import frozen_forward
                return frozen_forward(self,rgb,gate_context,details)
            if self.branch is None: return self.native(rgb).float()
            features,final = self.native.features(rgb)
            result = self.branch(rgb,features,final,gate_context,details)
            residual,info = result if details else (result,None)
            logits = self.native.head(final+self.alpha*residual).float()
            return (logits,info) if details else logits

    def optimizer_groups(self):
        banks = [] if self.branch is None else [p for n,p in self.branch.named_parameters() if n in ('bank.local','second_bank.local') and p.requires_grad]
        ordinary = [p for p in self.parameters() if p.requires_grad and p is not self.alpha and all(p is not q for q in banks)]
        groups=[dict(params=ordinary,lr=4e-4,base_lr=4e-4,weight_decay=.05)]
        if self.alpha is not None and self.alpha.requires_grad:
            groups.append(dict(params=[self.alpha],lr=4e-4,base_lr=4e-4,weight_decay=0.))
        if banks: groups.append(dict(params=banks,lr=3e-3,base_lr=3e-3,weight_decay=0.))
        return groups


def load_imagenet(model, path, mean, std):
    """Load local ImageNet body weights, retaining the paired random task head."""
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state=load_file(str(path))
    else:
        state=torch.load(path,map_location='cpu',weights_only=True)
        for key in ('state_dict','model'):
            if key in state and isinstance(state[key],dict): state=state[key]
    state={k.removeprefix('module.'):v for k,v in state.items()}
    head=classifier(model.native)
    excluded={k for k,p in model.native.body.named_parameters() if any(p is q for q in head.parameters())}
    current=model.native.body.state_dict()
    missing=[k for k in current if k not in excluded and (k not in state or state[k].shape!=current[k].shape)]
    if missing: raise ValueError('Incomplete or incompatible ImageNet body: '+str(missing))
    model.native.body.load_state_dict({k:v for k,v in state.items() if k in current and k not in excluded},strict=False)
    model.native.mean.copy_(torch.tensor(mean).view(1,3,1,1))
    model.native.std.copy_(torch.tensor(std).view(1,3,1,1))

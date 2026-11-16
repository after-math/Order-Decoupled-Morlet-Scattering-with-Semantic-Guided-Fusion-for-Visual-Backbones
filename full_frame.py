"""Full field-of-view preprocessing and static frozen-backbone residual alignment."""
import torch
from torch.nn import functional as F
from backbone_core import align


def letterbox(images,mean):
    native=[];rectangles=[]
    for image in images:
        h,w=image.shape[-2:]
        scale=256/max(h,w);rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
        top,left=(256-rh)//2,(256-rw)//2
        canvas=mean.reshape(3,1,1).expand(3,256,256).clone()
        canvas[:,top:top+rh,left:left+rw]=F.interpolate(image[None],size=(rh,rw),mode='bilinear',align_corners=False,antialias=True)[0]
        native.append(canvas);rectangles.append((top,left,rh,rw))
    return torch.stack(native),rectangles


def sample_content(semantic, rectangles):
    """Sample semantics over the original image, excluding mean-padded margins."""
    n, _, height, width = semantic.shape
    box = torch.as_tensor(rectangles, dtype=torch.float32, device=semantic.device)
    top, left, rh, rw = [v[:, None] for v in box.unbind(1)]
    x = (torch.arange(width, device=semantic.device, dtype=torch.float32)+.5)/width
    y = (torch.arange(height, device=semantic.device, dtype=torch.float32)+.5)/height
    gx, gy = 2*(left+rw*x)/256-1, 2*(top+rh*y)/256-1
    grid = torch.stack((gx[:, None, :].expand(n, height, width),
                        gy[:, :, None].expand(n, height, width)), -1)
    with torch.autocast(semantic.device.type, enabled=False):
        return F.grid_sample(semantic.float(), grid, mode='bilinear',
                             padding_mode='border', align_corners=False).to(semantic.dtype)


def content_to_canvas(content, rectangles, size):
    """Bilinear content alignment and exact valid-pixel coverage at the injection grid."""
    n = len(rectangles)
    height, width = size
    box = torch.as_tensor(rectangles, dtype=torch.float32, device=content.device)
    top, left, rh, rw = [v[:, None] for v in box.unbind(1)]
    ex = torch.arange(width+1, dtype=torch.float32, device=content.device)*256/width
    ey = torch.arange(height+1, dtype=torch.float32, device=content.device)*256/height
    gx = 2*((ex[:-1]+ex[1:])/2-left)/rw-1
    gy = 2*((ey[:-1]+ey[1:])/2-top)/rh-1
    grid = torch.stack((gx[:, None, :].expand(n, height, width),
                        gy[:, :, None].expand(n, height, width)), -1)
    with torch.autocast(content.device.type, enabled=False):
        canvas = F.grid_sample(content.float(), grid, mode='bilinear',
                               padding_mode='border', align_corners=False).to(content.dtype)
    ox = (torch.minimum(ex[1:], left+rw)-torch.maximum(ex[:-1], left)).clamp_min(0)/(256/width)
    oy = (torch.minimum(ey[1:], top+rh)-torch.maximum(ey[:-1], top)).clamp_min(0)/(256/height)
    return canvas, (oy[:, :, None]*ox[:, None, :])[:, None].to(canvas.dtype)


def frozen_forward(model,images,gate_context=None,details=False):
    native_rgb,rectangles=letterbox(images,model.native.mean)
    if model.branch is None:return model.native(native_rgb).float()
    branch=model.branch
    features,final=model.native.features(native_rgb)
    context=sample_content(branch.context(features),rectangles) if gate_context is None else gate_context
    with torch.autocast(native_rgb.device.type,enabled=False):
        first,second=branch.scattering(images,*branch.parameters_by_order())
    a,b=branch.texture1(first),branch.texture2(second)
    if model.variant=='FirstOrderOnly':b=torch.zeros_like(b)
    if model.variant=='SecondOrderOnly':a=torch.zeros_like(a)
    texture=branch.texture_fuse(torch.cat((a,b),1))
    gate=torch.ones_like(texture) if model.variant=='NoGate' else torch.sigmoid(branch.gate(torch.cat((context,texture),1)))
    z,coverage=content_to_canvas(gate*texture,rectangles,final.shape[-2:])
    residual=branch.output(z)*coverage
    logits=model.native.head(final+model.alpha*residual).float()
    return (logits,dict(first=first,second=second,gate=gate,texture=texture)) if details else logits

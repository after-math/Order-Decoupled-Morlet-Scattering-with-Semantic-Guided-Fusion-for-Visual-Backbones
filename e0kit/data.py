"""Native crop before256 resize; only explicitly supplied Train/Val records are opened."""
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load_partition(records,data_root,training=False):
    DATA_ROOT=Path(data_root)
    images=[]
    for record in records:
        with Image.open(DATA_ROOT/record['relative_path']) as image:
            image=image.convert('RGB')
            if not training:
                image=image.resize((256,256),Image.Resampling.BILINEAR)
            array=np.array(image,copy=True)
        images.append(torch.from_numpy(array).permute(2,0,1).contiguous())
    cache=images if training else torch.stack(images)
    return cache,torch.tensor([r['label'] for r in records])


def augmentation_specs(n,epoch,shapes,seed=42):
    assert len(shapes)==n
    rng=np.random.default_rng(seed+epoch*100003)
    specs=np.empty((n,7),dtype=np.int32)
    for i in range(n):
        source_height,source_width=shapes[i]
        for attempt in range(10):
            area=source_height*source_width*rng.uniform(.8,1.)
            ratio=math.exp(rng.uniform(math.log(.75),math.log(4/3)))
            width=int(round(math.sqrt(area*ratio)));height=int(round(math.sqrt(area/ratio)))
            if 0<width<=source_width and 0<height<=source_height:
                top=int(rng.integers(0,source_height+1-height));left=int(rng.integers(0,source_width+1-width));break
        else:
            # All observed native UCMerced aspect ratios lie in[.75,4/3].
            top=left=0;width=source_width;height=source_height
        specs[i]=[top,left,height,width,int(rng.random()<.5),int(rng.random()<.5),int(rng.integers(0,4))]
    return specs


def input_batch(images,indices,device,specs=None):
    if specs is None:
        return images[indices].to(device,non_blocking=True).float()/255
    results=[]
    rotations={1:Image.Transpose.ROTATE_90,2:Image.Transpose.ROTATE_180,3:Image.Transpose.ROTATE_270}
    for index in indices.tolist():
        top,left,height,width,hflip,vflip,turn=map(int,specs[index])
        # Crop the native image before resizing, including non256 source images.
        a=images[index][:,top:top+height,left:left+width].permute(1,2,0).numpy()
        image=Image.fromarray(a).resize((256,256),Image.Resampling.BILINEAR)
        if hflip:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if vflip:image=image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if turn:image=image.transpose(rotations[turn])
        results.append(torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).contiguous())
    return torch.stack(results).to(device).float()/255

"""Two-order Morlet scattering on original pixels, with globally phased overlapping tiles."""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from e0kit.scattering import FullResolutionScattering


@lru_cache(maxsize=4096)
def fft_extent(length):
    """First 2/3/5-smooth FFT size that fits the reflected image, including both halos."""
    candidate = length + 56
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


class OriginalGridScattering(FullResolutionScattering):
    def __init__(self, core=384, halo=72, direct_limit=300000, path_chunk=18):
        super().__init__(second_order_chunk=path_chunk)
        if halo < 28 + 28 + 13 or core % 8 or halo % 8 or core <= 0:
            raise ValueError('Tile centers and sufficient halos must align with the stride-eight grid')
        self.core = core
        self.halo = halo
        self.direct_limit = direct_limit

    def smooth(self, values):
        height, width = values.shape[-2:]
        x = values.reshape(-1, 1, height, width)
        x = F.conv2d(F.pad(x, (0,0,13,13), mode='reflect'), self.lowpass.view(1,1,27,1), stride=(8,1))
        x = F.conv2d(F.pad(x, (13,13,0,0), mode='reflect'), self.lowpass.view(1,1,1,27), stride=(1,8))
        return x.reshape(*values.shape[:-2], x.shape[-2], x.shape[-1])

    @staticmethod
    def image_fft(x, size):
        return torch.fft.fft2(F.pad(x, (28,28,28,28), mode='reflect'), s=size)

    def order_two(self, parents, filters, first, second, height, width):
        response = torch.fft.ifft2(parents[:, first] * filters[second].unsqueeze(0))
        return self.smooth(response[..., 28:28+height, 28:28+width].abs())

    def dense(self, image, kernels, checkpoint_paths=False):
        channels, height, width = image.shape
        if channels != 3 or min(height, width) <= 28:
            raise ValueError(f'Invalid RGB image {image.shape}')
        size = fft_extent(height), fft_extent(width)
        filters = F.pad(kernels, (0, size[1]-57, 0, size[0]-57))
        filters = torch.fft.fft2(torch.roll(filters, (-28,-28), (-2,-1)))
        first = torch.fft.ifft2(self.image_fft(image, size)[:, None] * filters)
        u1 = first[..., 28:28+height, 28:28+width].abs()
        s1 = self.smooth(u1)
        parents = self.image_fft(u1[:, :12], size)
        path_maps = []
        for start in range(0, len(self.first_ids), self.second_order_chunk):
            end = min(start + self.second_order_chunk, len(self.first_ids))
            args = parents, filters, self.first_ids[start:end], self.second_ids[start:end], height, width
            if checkpoint_paths and end <= 54 and torch.is_grad_enabled():
                values = checkpoint(self.order_two, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                values = self.order_two(*args)
            path_maps.append(values)
        return s1.flatten(0,1), torch.cat(path_maps, dim=1).flatten(0,1)

    def tile_regions(self, height, width, core):
        """Yield full-resolution source extents and stride-eight center extents within each tile."""
        for top in range(0, height, core):
            row = []
            bottom = min(top+core, height)
            for left in range(0, width, core):
                right = min(left+core, width)
                y0, x0 = max(0, top-self.halo), max(0, left-self.halo)
                y1, x1 = min(height, bottom+self.halo), min(width, right+self.halo)
                centers = ((top-y0)//8, (bottom-y0+7)//8, (left-x0)//8, (right-x0+7)//8)
                row.append(((y0,y1,x0,x1), centers))
            yield row

    def tile_center(self, image, kernels, centers):
        y0,y1,x0,x1 = centers
        return tuple(s[:, y0:y1, x0:x1].contiguous() for s in self.dense(image, kernels))

    def maps(self, image, kernels, *, force_core=None):
        height, width = image.shape[-2:]
        if force_core == 0 or (force_core is None and height * width <= self.direct_limit):
            return self.dense(image, kernels, checkpoint_paths=True)
        core = self.core if force_core is None else force_core
        if core <= 0 or core % 8:
            raise ValueError('The tile core must be a positive multiple of eight')
        rows1, rows2 = [], []
        for regions in self.tile_regions(height, width, core):
            first, second = [], []
            for (y0,y1,x0,x1), centers in regions:
                src = image[:, y0:y1, x0:x1]
                if torch.is_grad_enabled():
                    a, b = checkpoint(self.tile_center, src, kernels, centers,
                                      use_reentrant=False, preserve_rng_state=False)
                else:
                    a, b = self.tile_center(src, kernels, centers)
                first.append(a)
                second.append(b)
            rows1.append(torch.cat(first, dim=-1))
            rows2.append(torch.cat(second, dim=-1))
        s1, s2 = torch.cat(rows1, dim=-2), torch.cat(rows2, dim=-2)
        expected = ((height+7)//8, (width+7)//8)
        if s1.shape != (54,*expected) or s2.shape != (324,*expected):
            raise RuntimeError('Stitched scattering grids lost image coverage')
        return s1, s2

    def forward(self, images, kernels):
        first, second = [], []
        for im in images:
            s1, s2 = self.maps(im, kernels)
            first.append(F.adaptive_avg_pool2d(torch.log1p(s1.clamp_min(0)), (16,16)))
            second.append(F.adaptive_avg_pool2d(torch.log1p(s2.clamp_min(0)), (16,16)))
        return torch.stack(first), torch.stack(second)

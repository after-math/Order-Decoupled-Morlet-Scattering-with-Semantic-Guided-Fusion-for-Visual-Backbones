"""Static independent order banks on original pixels with overlapping FFT tiles."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from original_grid import OriginalGridScattering, fft_extent
from order_scattering import OrderScattering


class StaticOriginalScattering(OriginalGridScattering):
    def __init__(self, core=384, halo=72, direct_limit=400000, path_chunk=18):
        super().__init__(core, halo, direct_limit, path_chunk)
        indices = OrderScattering()
        self.first_ids.copy_(indices.first_ids)
        self.second_ids.copy_(indices.second_ids)
        self.kernels = indices.kernels

    def dense(self, image, kernels, checkpoint_paths=False):
        channels, height, width = image.shape
        if channels != 3 or min(height, width) <= 28:
            raise ValueError('Original-pixel RGB dimensions must exceed the 28-pixel halo')
        first_kernels, second_kernels = kernels
        size = fft_extent(height), fft_extent(width)

        def spectrum(values):
            values = F.pad(values, (0, size[1]-57, 0, size[0]-57))
            return torch.fft.fft2(torch.roll(values, (-28, -28), (-2, -1)))

        filters1, filters2 = spectrum(first_kernels), spectrum(second_kernels)
        response = torch.fft.ifft2(self.image_fft(image, size)[:, None] * filters1)
        u1 = response[..., 28:28+height, 28:28+width].abs()
        s1 = self.smooth(u1)
        parents = self.image_fft(u1[:, :12], size)
        paths = []
        for start in range(0, 108, self.second_order_chunk):
            stop = min(start + self.second_order_chunk, 108)
            args = (parents, filters2, self.first_ids[start:stop],
                    self.second_ids[start:stop], height, width)
            if checkpoint_paths and torch.is_grad_enabled():
                value = checkpoint(self.order_two, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                value = self.order_two(*args)
            paths.append(value)
        return s1.flatten(0, 1), torch.cat(paths, dim=1).flatten(0, 1)

    def _image(self, image, q1, q2):
        first, second = self.maps(image, (self.kernels(q1), self.kernels(q2)))
        return tuple(F.adaptive_avg_pool2d(torch.log1p(x.clamp_min(0)), (16, 16))
                     for x in (first, second))

    def forward(self, images, q1, q2):
        if q1.shape != (18, 4) or q2.shape != (18, 4):
            raise ValueError('Two batch-shared static order banks are required')
        outputs = []
        for index, image in enumerate(images):
            with torch.autocast(image.device.type, enabled=False):
                args = image.float(), q1.float(), q2.float()
                if torch.is_grad_enabled():
                    pair = checkpoint(self._image, *args, use_reentrant=False, preserve_rng_state=False)
                else:
                    pair = self._image(*args)
            outputs.append(pair)
        return tuple(torch.stack([pair[i] for pair in outputs]) for i in range(2))

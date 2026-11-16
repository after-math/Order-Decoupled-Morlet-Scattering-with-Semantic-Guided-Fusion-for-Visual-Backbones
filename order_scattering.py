"""Independent learned order banks with batch-shared second-order parameters."""
import torch
from torch.utils.checkpoint import checkpoint

from e0kit.scattering import (
    E0Bank, FullResolutionScattering, MorletKernels, anchored_sigmoid, kernel_fft,
)


class SecondOrderBank(E0Bank):
    def __init__(self):
        super().__init__()
        indices = torch.tensor(list(range(6, 12)) + list(range(12, 18)) * 2)
        self.reference.copy_(self.reference[indices].clone())

    def decode(self, raw):
        reference = self.reference
        theta = reference[:, 1] + raw[:, 0]
        xi = anchored_sigmoid(raw[:, 1], reference[:, 2], 0.1, 2.9)
        sigma = anchored_sigmoid(raw[:, 2], reference[:, 0], 0.3, 8.0)
        gamma = anchored_sigmoid(raw[:, 3], reference[:, 3], 0.3, 1.5)
        return torch.stack((sigma, theta, xi, gamma), -1)

    def forward(self):
        return self.decode(self.local)


class ReferenceOrderScattering(FullResolutionScattering):
    def __init__(self, image_chunk=2, path_chunk=18, checkpoint_chunks=True):
        super().__init__(second_order_chunk=path_chunk)
        old_second = self.second_ids.clone()
        first = self.first_ids
        transition = torch.where(
            (first // 6 == 0) & (old_second // 6 == 1), 0,
            torch.where((first // 6 == 0) & (old_second // 6 == 2), 1, 2),
        )
        self.second_ids.copy_(transition * 6 + old_second % 6)
        self.kernels = MorletKernels()
        self.image_chunk = image_chunk
        self.checkpoint_chunks = checkpoint_chunks
        assert self.first_ids.numel() == self.second_ids.numel() == 108

    def partition(self, raw, first_parameters, second_parameters):
        batch = raw.shape[0]
        with torch.autocast(raw.device.type, enabled=False):
            raw = raw.float()
            first_filters = kernel_fft(self.kernels(first_parameters.float()), 320)
            second_filters = kernel_fft(self.kernels(second_parameters.float()), 320)
            response = torch.fft.ifft2(
                self.padded_fft(raw.reshape(batch * 3, 256, 256))[:, None] * first_filters[None],
                norm="backward",
            )
            first_amplitude = response[..., 28:284, 28:284].abs()
            del response
            first = FullResolutionScattering.lowpass_downsample(self, first_amplitude).reshape(batch, 54, 32, 32)
            parents = self.padded_fft(
                first_amplitude[:, :12].reshape(batch * 3 * 12, 256, 256)
            ).reshape(batch, 3, 12, 320, 320)
            outputs = []
            for begin in range(0, 108, self.second_order_chunk):
                end = begin + self.second_order_chunk
                first_indices = self.first_ids[begin:end]
                second_indices = self.second_ids[begin:end]
                response = torch.fft.ifft2(
                    parents[:, :, first_indices] * second_filters[None, None, second_indices],
                    norm="backward",
                )
                outputs.append(FullResolutionScattering.lowpass_downsample(self, response[..., 28:284, 28:284].abs()))
            second = torch.cat(outputs, dim=2).reshape(batch, 324, 32, 32)
            return first, second

    def forward(self, raw, first_parameters, second_parameters):
        assert raw.shape[1:] == (3, 256, 256)
        assert first_parameters.shape == second_parameters.shape == (18, 4)
        outputs = []
        for begin in range(0, raw.shape[0], self.image_chunk):
            part = raw[begin:begin + self.image_chunk]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                result = checkpoint(
                    self.partition, part, first_parameters, second_parameters,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                result = self.partition(part, first_parameters, second_parameters)
            outputs.append(result)
        return tuple(torch.cat([result[index] for result in outputs]) for index in range(2))


class OrderScattering(ReferenceOrderScattering):
    """Same 108 paths using rectangular broadcasts and one spectrum per batch.

    No cross-forward cache: each call regenerates filters from current parameters.
    """
    def __init__(self, image_chunk=2, path_chunk=18, checkpoint_chunks=True):
        super().__init__(image_chunk, path_chunk, checkpoint_chunks)
        self.group_first = 2
        self.group_second = 6
        self.lowpass_backend = "triton"
        self.fused_magnitude = True
        self.single_image_reference = True

    def lowpass_downsample(self, amplitude):
        if self.lowpass_backend == "triton" and amplitude.is_cuda:
            from fast_lowpass import gaussian_stride
            return gaussian_stride(amplitude, self.lowpass)
        return super().lowpass_downsample(amplitude)

    def magnitude_crop(self, response):
        if self.fused_magnitude and response.is_cuda:
            from fast_lowpass import magnitude_crop
            return magnitude_crop(response)
        return response[...,28:284,28:284].abs()

    def partition_spectra(self, raw, first_filters, second_filters):
        batch = raw.shape[0]
        with torch.autocast(raw.device.type, enabled=False):
            raw = raw.float()
            response = torch.fft.ifft2(
                self.padded_fft(raw.reshape(batch * 3, 256, 256))[:, None] * first_filters[None],
                norm="backward",
            )
            amplitude = self.magnitude_crop(response)
            del response
            first = self.lowpass_downsample(amplitude).reshape(batch, 54, 32, 32)
            parents = self.padded_fft(
                amplitude[:, :12].reshape(batch * 3 * 12, 256, 256)
            ).reshape(batch, 3, 12, 320, 320)
            transitions = []
            # Original lexicographic pairs: 0:6 x 0:12, then 6:12 x 12:18.
            # Rectangular slices remove advanced-index copies of repeated parents.
            for p0, p1, f0, f1 in ((0, 6, 0, 12), (6, 12, 12, 18)):
                parent_blocks = []
                for start in range(p0, p1, self.group_first):
                    frequency_blocks = []
                    for k0 in range(f0, f1, self.group_second):
                        product = parents[:, :, start:min(start+self.group_first,p1), None] * second_filters[None, None, None, k0:min(k0+self.group_second,f1)]
                        response = torch.fft.ifft2(product, norm="backward")
                        del product
                        frequency_blocks.append(self.lowpass_downsample(self.magnitude_crop(response)))
                    parent_blocks.append(torch.cat(frequency_blocks, dim=3))
                transitions.append(torch.cat(parent_blocks, dim=2).flatten(2, 3))
            second = torch.cat(transitions, dim=2).reshape(batch, 324, 32, 32)
            return first, second

    def forward(self, raw, first_parameters, second_parameters):
        assert raw.shape[1:] == (3,256,256)
        assert first_parameters.shape == second_parameters.shape == (18,4)
        # CUDA launch overhead dominates a single image. Keep the original exact
        # path there; batched inference/training use the fused implementation.
        if self.single_image_reference and raw.shape[0] == 1 and not torch.is_grad_enabled():
            return super().forward(raw, first_parameters, second_parameters)
        with torch.autocast(raw.device.type, enabled=False):
            first_filters=kernel_fft(self.kernels(first_parameters.float()),320)
            second_filters=kernel_fft(self.kernels(second_parameters.float()),320)
            outputs=[]
            for start in range(0,raw.shape[0],self.image_chunk):
                part=raw[start:start+self.image_chunk]
                if self.checkpoint_chunks and torch.is_grad_enabled():
                    result=checkpoint(self.partition_spectra,part,first_filters,second_filters,
                        use_reentrant=False,preserve_rng_state=False)
                else:
                    result=self.partition_spectra(part,first_filters,second_filters)
                outputs.append(result)
            return tuple(torch.cat([result[i] for result in outputs]) for i in range(2))

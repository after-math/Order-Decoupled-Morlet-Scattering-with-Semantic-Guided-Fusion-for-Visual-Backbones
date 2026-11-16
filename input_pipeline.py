"""Original deterministic PIL pixels, bounded CPU prefetch, and asynchronous copies."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time
import numpy as np
import torch
from PIL import Image


def prepare_pinned_batch(images, labels, indices, specs):
    begin = time.perf_counter()
    rotations = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                 3: Image.Transpose.ROTATE_270}
    augmented = []
    for index in indices.tolist():
        top, left, height, width, horizontal, vertical, turn = map(int, specs[index])
        crop = images[index][:, top:top + height, left:left + width].permute(1, 2, 0).numpy()
        image = Image.fromarray(crop).resize((256, 256), Image.Resampling.BILINEAR)
        if horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if turn:
            image = image.transpose(rotations[turn])
        augmented.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous())
    pixels = torch.stack(augmented).pin_memory()
    target = labels[indices].contiguous().pin_memory()
    return pixels, target, time.perf_counter() - begin


class PreparedPrefetcher:
    def __init__(self, images, labels, batches, specs, device, depth=2, workers=2):
        assert depth >= 1 and workers >= 1 and torch.device(device).type == 'cuda'
        self.images, self.labels, self.specs = images, labels, specs
        self.device = torch.device(device)
        self.depth = depth
        self.source = iter(batches)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.stream = torch.cuda.Stream(device=self.device)
        self.pending = deque()
        self.inflight = deque()
        self.ready = None
        self.cpu_prepare_seconds = 0.
        self.cpu_wait_seconds = 0.
        self.copy_events = []
        self.started = False

    def fill(self):
        while len(self.pending) < self.depth:
            try:
                ids = next(self.source)
            except StopIteration:
                break
            future = self.pool.submit(prepare_pinned_batch, self.images, self.labels,
                                      ids, self.specs)
            self.pending.append((ids, future))

    def preload(self):
        if not self.pending:
            self.ready = None
            return
        ids, future = self.pending.popleft()
        begin = time.perf_counter()
        pixels, host_target, cpu_seconds = future.result()
        self.cpu_prepare_seconds += cpu_seconds
        self.cpu_wait_seconds += time.perf_counter() - begin
        self.fill()
        with torch.cuda.stream(self.stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            rgb = pixels.to(self.device, non_blocking=True).float() / 255
            target = host_target.to(self.device, non_blocking=True)
            end.record(self.stream)
        self.copy_events.append((start, end))
        self.inflight.append((end, pixels, host_target))
        while self.inflight and self.inflight[0][0].query():
            self.inflight.popleft()
        while len(self.inflight) > self.depth + 1:
            self.inflight.popleft()[0].synchronize()
        self.ready = ids, rgb, target, end

    def __iter__(self):
        if not self.started:
            self.started = True
            self.fill()
            self.preload()
        return self

    def __next__(self):
        if self.ready is None:
            raise StopIteration
        ids, rgb, target, ready = self.ready
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(ready)
        rgb.record_stream(consumer)
        target.record_stream(consumer)
        # Stage the NEXT batch before returning so its copy can overlap this step.
        self.preload()
        return ids, rgb, target

    def statistics(self):
        return dict(cpu_prepare_seconds=self.cpu_prepare_seconds,
                    cpu_future_wait_seconds=self.cpu_wait_seconds,
                    copy_stream_seconds=sum(a.elapsed_time(b) for a, b in self.copy_events) / 1000,
                    depth=self.depth)

    def close(self):
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.stream.synchronize()
            self.inflight.clear()


@contextmanager
def prefetched_batches(images, labels, batches, specs, device, depth=2, workers=2):
    iterator = PreparedPrefetcher(images, labels, batches, specs, device, depth, workers)
    try:
        yield iterator
    finally:
        iterator.close()


def mixup_prepared_batch(rgb, target, plan):
    if not plan['applied']:
        return rgb, target, None
    permutation = torch.as_tensor(plan['permutation'], device=rgb.device, dtype=torch.long)
    assert permutation.numel() == len(rgb)
    lam = float(plan['lam'])
    return lam * rgb + (1 - lam) * rgb[permutation], target, target[permutation]

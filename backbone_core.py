"""Native classifiers and order-decoupled static Morlet residual branches."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from e0kit.scattering import ChannelNorm, E0Bank
from order_scattering import OrderScattering, SecondOrderBank

BACKBONES = {
    "edgenext_small": 128,
    "shufflenet_v2_x1_0": 64,
    "mobilevitv2_100": 128,
    "resnet18": 128,
    "convnext_tiny": 256,
    "efficientnet_b0": 128,
    "mambaout_femto": 128,
}


class Native(nn.Module):
    def __init__(self, name, classes=7):
        super().__init__()
        self.name = name
        if name == "shufflenet_v2_x1_0":
            from torchvision.models import shufflenet_v2_x1_0
            self.body = shufflenet_v2_x1_0(weights=None, num_classes=classes)
            self.channels, self.out_channels = [116, 232, 464], 1024
        elif name == "resnet18":
            from torchvision.models import resnet18
            self.body = resnet18(weights=None, num_classes=classes)
            self.channels, self.out_channels = [128, 256, 512], 512
        elif name == "mambaout_femto":
            from mambaout_official import mambaout_femto
            self.body = mambaout_femto(pretrained=False, num_classes=classes, head_dropout=0.0, drop_path_rate=0.0)
            self.channels, self.out_channels = [96, 192, 288], 288
        else:
            import timm
            self.body = timm.create_model(
                name, pretrained=False, num_classes=classes, drop_rate=0.0,
                drop_path_rate=0.1 if name == "edgenext_small" else 0.0,
            )
            rows = [next(row for row in reversed(self.body.feature_info) if row["reduction"] == stride)
                    for stride in (8, 16, 32)]
            self.nodes = [row["module"] for row in rows]
            self.channels = [row["num_chs"] for row in rows]
            self.out_channels = self.body.num_features
        self.register_buffer("mean", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("std", torch.full((1, 3, 1, 1), 0.25))

    def forward(self, rgb):
        return self.body((rgb - self.mean) / self.std)

    def features(self, rgb):
        body = self.body
        value = (rgb - self.mean) / self.std
        if self.name == "shufflenet_v2_x1_0":
            first = body.stage2(body.maxpool(body.conv1(value)))
            second = body.stage3(first)
            third = body.stage4(second)
            return [first, second, third], body.conv5(third)
        if self.name == "resnet18":
            first = body.layer2(body.layer1(body.maxpool(body.relu(body.bn1(body.conv1(value))))))
            second = body.layer3(first)
            third = body.layer4(second)
            return [first, second, third], third
        if self.name == "mambaout_femto":
            outputs = []
            for index in range(body.num_stage):
                value = body.stages[index](body.downsample_layers[index](value))
                if index in (1, 2, 3):
                    outputs.append(value.permute(0, 3, 1, 2).contiguous())
            return outputs, outputs[-1]
        outputs = {}
        def capture(key):
            def hook(module, inputs, output):
                outputs[key] = output
            return hook
        hooks = [body.get_submodule(key).register_forward_hook(capture(key)) for key in self.nodes]
        try:
            final = body.forward_features(value)
        finally:
            for hook in hooks:
                hook.remove()
        return [outputs[key] for key in self.nodes], final

    def head(self, features):
        if self.name == "shufflenet_v2_x1_0":
            return self.body.fc(features.mean((2, 3)))
        if self.name == "resnet18":
            return self.body.fc(torch.flatten(self.body.avgpool(features), 1))
        if self.name == "mambaout_femto":
            pooled = features.permute(0, 2, 3, 1).contiguous().mean((1, 2))
            return self.body.head(self.body.norm(pooled))
        return self.body.forward_head(features)


def initialized(name, factory):
    with torch.random.fork_rng(devices=[]):
        value = int(hashlib.sha256(("42:" + name).encode()).hexdigest()[:12], 16)
        torch.manual_seed(value)
        return factory()


def projection(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 1), ChannelNorm(outputs), nn.GELU())


def align(value, size=(16, 16)):
    if value.shape[-2:] == size:
        return value
    if value.shape[-2] >= size[0] and value.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(value, size)
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


class WaveBranch(nn.Module):
    def __init__(self, native, width):
        super().__init__()
        self.width = width
        self.projections = initialized("projections", lambda: nn.ModuleList(projection(c, width) for c in native.channels))
        self.semantic_fuse = initialized("semantic_fuse", lambda: projection(len(native.channels) * width, width))
        self.gate = initialized("gate", lambda: nn.Conv2d(2 * width, width, 1))
        self.output = initialized("out", lambda: nn.Conv2d(width, native.out_channels, 1))
        self.bank = E0Bank()
        self.second_bank = SecondOrderBank()
        self.scattering = OrderScattering(image_chunk=2, path_chunk=18, checkpoint_chunks=True)
        def texture_projection(inputs, outputs):
            return nn.Sequential(ChannelNorm(inputs), nn.Conv2d(inputs, outputs, 1), nn.GELU())
        self.texture1 = initialized("t1", lambda: texture_projection(54, width // 4))
        self.texture2 = initialized("t2", lambda: texture_projection(324, width // 4))
        self.texture_fuse = initialized("tfuse", lambda: nn.Sequential(
            ChannelNorm(width // 2), nn.Conv2d(width // 2, width, 1), ChannelNorm(width),
        ))

    def forward(self, rgb, features, final):
        context = self.semantic_fuse(torch.cat([align(layer(value)) for layer, value in zip(self.projections, features)], 1))
        with torch.autocast(rgb.device.type, enabled=False):
            first, second = self.scattering(rgb.float(), self.bank(), self.second_bank())
        first = F.adaptive_avg_pool2d(torch.log1p(first.clamp_min(0)), (16, 16))
        second = F.adaptive_avg_pool2d(torch.log1p(second.clamp_min(0)), (16, 16))
        texture = self.texture_fuse(torch.cat((self.texture1(first), self.texture2(second)), 1))
        weighted = torch.sigmoid(self.gate(torch.cat((context, texture), 1))) * texture
        return self.output(F.interpolate(weighted, size=final.shape[-2:], mode="bilinear", align_corners=False))

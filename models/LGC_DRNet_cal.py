import os
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision.models import mobilenet_v3_large

def make_group_norm(channels: int, max_groups: int=4) -> nn.GroupNorm:
    for groups in (max_groups, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
    return nn.GroupNorm(num_groups=1, num_channels=channels)

class ConvGNAct(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.norm = make_group_norm(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super().__init__()
        self.dw = ConvGNAct(in_ch, in_ch, k=3, s=stride, groups=in_ch, act=True)
        self.pw = ConvGNAct(in_ch, out_ch, k=1, s=1, p=0, groups=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))

class DCNv2Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.offset_mask_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.offset_mask_conv.weight, 0.0)
        nn.init.constant_(self.offset_mask_conv.bias, 0.0)
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = make_group_norm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        offset_mask = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        y = self.deform_conv(x, offset, mask)
        return self.act(self.norm(y))

class StripPooling(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = self.conv_h(self.pool_h(x)).expand(-1, -1, h, w)
        x_w = self.conv_w(self.pool_w(x)).expand(-1, -1, h, w)
        y = F.relu(x_h + x_w, inplace=True)
        attention = torch.sigmoid(self.conv_out(y))
        return x * attention

class ECALite(nn.Module):

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class LiteTransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x, norm_x)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MobileViTContextBlock(nn.Module):

    def __init__(self, channels=80, attn_dim=64, num_heads=4, mlp_ratio=2.0, pool_size=2, depth=1):
        super().__init__()
        self.pool_size = pool_size
        self.local_rep = nn.Sequential(ConvGNAct(channels, channels, k=3, s=1, p=1, act=True), ConvGNAct(channels, attn_dim, k=1, s=1, p=0, act=True))
        self.global_blocks = nn.ModuleList([LiteTransformerEncoder(attn_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.fuse = nn.Sequential(ConvGNAct(attn_dim + channels, channels, k=1, s=1, p=0, act=True), DSConv(channels, channels, stride=1, act=True), ECALite(channels))

    def forward(self, x):
        identity = x
        h, w = x.shape[-2:]
        local_feat = self.local_rep(x)
        pooled = F.avg_pool2d(local_feat, kernel_size=self.pool_size, stride=self.pool_size)
        ph, pw = pooled.shape[-2:]
        tokens = pooled.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            tokens = block(tokens)
        global_feat = tokens.transpose(1, 2).reshape(x.size(0), -1, ph, pw)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        out = self.fuse(torch.cat([x, global_feat], dim=1))
        return out + identity

class DCNDecoder(nn.Module):

    def __init__(self, in_ch, skip_ch_list, out_ch, use_eca=True):
        super().__init__()
        total_in = in_ch + sum(skip_ch_list)
        self.fuse = nn.Sequential(ConvGNAct(total_in, out_ch, k=1, s=1, p=0, act=True), DCNv2Block(out_ch, out_ch))
        self.eca = ECALite(out_ch) if use_eca else nn.Identity()

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        aligned = [x]
        for skip in skips:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            aligned.append(skip)
        return self.eca(self.fuse(torch.cat(aligned, dim=1)))

class LGCDRNet(nn.Module):

    def __init__(self, in_chans=3, num_classes=2, pretrained=False, weight_path=None, freeze_backbone_bn=True):
        super().__init__()
        self.variant_name = 'LGC-DRNet'
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        if in_chans != 3:
            raise ValueError('LGC_DRNet requires three-channel RGB input.')
        if pretrained and weight_path is None:
            raise ValueError('Provide weight_path when pretrained=True, or set pretrained=False.')
        mv3_model = mobilenet_v3_large(weights=None)
        if pretrained:
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            mv3_model.load_state_dict(state_dict, strict=True)
        mv3 = mv3_model.features
        self.s_layer0 = mv3[0:2]
        self.s_layer1 = mv3[2:4]
        self.s_layer2 = mv3[4:7]
        self.s_layer3 = mv3[7:11]
        self.d0_adapter = ConvGNAct(16, 8, k=1, s=1, p=0, act=True)
        self.d1_adapter = ConvGNAct(24, 16, k=1, s=1, p=0, act=True)
        self.deep_context = nn.Sequential(MobileViTContextBlock(channels=80, attn_dim=64, num_heads=4, mlp_ratio=2.0, pool_size=2, depth=1), StripPooling(in_channels=80))
        self.dec2 = DCNDecoder(in_ch=80, skip_ch_list=[40], out_ch=24, use_eca=True)
        self.dec1 = DCNDecoder(in_ch=24, skip_ch_list=[24, 16], out_ch=12, use_eca=True)
        self.dec0 = DCNDecoder(in_ch=12, skip_ch_list=[8, 16], out_ch=8, use_eca=True)
        self.final_refine = DCNv2Block(8, 8)
        self.seg_head = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0)

    def _backbone_bn_modules(self) -> Iterable[nn.BatchNorm2d]:
        for block in (self.s_layer0, self.s_layer1, self.s_layer2, self.s_layer3):
            for module in block.modules():
                if isinstance(module, nn.BatchNorm2d):
                    yield module

    def train(self, mode: bool=True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for module in self._backbone_bn_modules():
                module.eval()
        return self

    def num_frozen_backbone_bn_layers(self) -> int:
        return sum((1 for _ in self._backbone_bn_modules())) if self.freeze_backbone_bn else 0

    def forward(self, x):
        s0 = self.s_layer0(x)
        s1 = self.s_layer1(s0)
        d0 = self.d0_adapter(s0)
        d1 = self.d1_adapter(s1)
        s2 = self.s_layer2(s1)
        s3 = self.s_layer3(s2)
        s3 = self.deep_context(s3)
        y = self.dec2(s3, [s2])
        y = self.dec1(y, [s1, d1])
        y = self.dec0(y, [d0, s0])
        y = self.final_refine(y)
        seg = self.seg_head(y)
        seg = F.interpolate(seg, scale_factor=2.0, mode='bilinear', align_corners=False)
        return seg
LGC_DRNet = LGCDRNet

def print_model_statistics(input_size=(1, 3, 512, 512), device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = LGCDRNet(
        num_classes=2,
        pretrained=False,
        freeze_backbone_bn=True
    ).to(device)

    model.eval()

    inputs = torch.randn(*input_size, device=device)

    with torch.no_grad():
        outputs = model(inputs)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print('=' * 70)
    print(f'Model                   : {model.variant_name}')
    print(f'Device                  : {device}')
    print(f'Input shape             : {tuple(inputs.shape)}')
    print(f'Output shape            : {tuple(outputs.shape)}')
    print(f'Total parameters        : {total_params:,}')
    print(f'Total parameters (M)    : {total_params / 1e6:.6f}')
    print(f'Trainable parameters    : {trainable_params:,}')
    print(f'Trainable parameters (M): {trainable_params / 1e6:.6f}')
    print(
        f'Frozen backbone BN      : '
        f'{model.num_frozen_backbone_bn_layers()}'
    )

    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, inputs)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)

        total_flops = analysis.total()
        unsupported_ops = dict(analysis.unsupported_ops())

        print(f'FLOPs                   : {total_flops:,.0f}')
        print(f'FLOPs (G)               : {total_flops / 1e9:.6f}')

        if unsupported_ops:
            print(f'Unsupported operators   : {unsupported_ops}')
        else:
            print('Unsupported operators   : None')

    except ImportError:
        print('FLOPs                   : fvcore is not installed')
        print('Install command         : pip install fvcore')

    print('=' * 70)


if __name__ == '__main__':
    print_model_statistics(
        input_size=(1, 3, 512, 512),
        device=None
    )
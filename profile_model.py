import argparse
import json

import torch

from models import LGCDRNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=512, help='Input size, at least 32 and divisible by 16')
    parser.add_argument('--device', default='cpu', help='Profiling device, such as cpu or cuda:0')
    parser.add_argument('--flops', action='store_true', help='Report fvcore counted operations and unsupported operators')
    args = parser.parse_args()
    if args.img_size < 32 or args.img_size % 16:
        parser.error('Use img_size >=32 divisible by 16.')
    model = LGCDRNet(pretrained=False).eval().to(args.device)
    inputs = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    with torch.no_grad():
        outputs = model(inputs)
    result = {'parameters': sum(p.numel() for p in model.parameters()), 'input_shape': list(inputs.shape), 'output_shape': list(outputs.shape), 'frozen_backbone_bn_layers': model.num_frozen_backbone_bn_layers()}
    if args.flops:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(model, inputs)
        result['fvcore_counted_operations'] = analysis.total()
        result['unsupported_operators'] = dict(analysis.unsupported_ops())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

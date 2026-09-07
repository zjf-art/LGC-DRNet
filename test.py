import argparse
import json
from pathlib import Path

import torch

from datasets.crack_dataset import validate_dataset
from inference import load_model
from trainer import evaluate_from_saved_pngs, save_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./public_datasets/CamCrack789/weights/seed1729/best_epoch_IoU_0.738_mIoU_0.865_ODS_0.845.pth', help='LGC_DRNet checkpoint path')
    parser.add_argument('--test_dir', default='./dataset/CamCrack789/test', help='Dataset split containing images and labels or masks')
    parser.add_argument('--dataset', default='CamCrack789', help='Dataset name; CamCrack789 enables image-to-target mask mapping')
    parser.add_argument('--output_dir', default='./test_results/CamCrack789', help='New evaluation output directory')
    parser.add_argument('--img_size', type=int, default=512, help='Evaluation size, at least 32 and divisible by 16')
    parser.add_argument('--binary_threshold', type=float, default=0.5, help='Fixed threshold for per-image metrics and binary masks')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0')
    args = parser.parse_args()
    if args.img_size < 32 or args.img_size % 16 or not 0 <= args.binary_threshold <= 1:
        parser.error('Invalid image size or threshold.')
    validate_dataset(args.test_dir, args.dataset)
    args.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model = load_model(args.checkpoint, args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_predictions(args, model, 0, str(output / 'images'))
    result = evaluate_from_saved_pngs(str(output / 'images'), args)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    config = vars(args).copy()
    config['device'] = str(config['device'])
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

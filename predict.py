import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from datasets.crack_dataset import list_images
from inference import load_model, predict_image, save_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='LGC_DRNet checkpoint path')
    parser.add_argument('--input', required=True, help='Image file or image directory')
    parser.add_argument('--output_dir', default='./predictions', help='New output directory')
    parser.add_argument('--mode', choices=['resize', 'sliding'], default='sliding', help='Resize to square or use sliding windows at original resolution')
    parser.add_argument('--img_size', type=int, default=512, help='Input patch size, at least 32 and divisible by 16')
    parser.add_argument('--stride', type=int, default=384, help='Sliding-window stride, between 1 and img_size')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sliding windows per inference batch')
    parser.add_argument('--binary_threshold', type=float, default=0.5, help='Probability threshold for binary masks')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0')
    args = parser.parse_args()
    if args.img_size < 32 or args.img_size % 16 or args.batch_size < 1 or not 0 <= args.binary_threshold <= 1:
        parser.error('Invalid size, batch size or threshold.')
    if args.mode == 'sliding' and not 1 <= args.stride <= args.img_size:
        parser.error('Stride must be between 1 and img_size.')
    source = Path(args.input)
    files = [source] if source.is_file() else list_images(source)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model = load_model(args.checkpoint, device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    for path in tqdm(files):
        with Image.open(path) as image:
            image = image.convert('RGB')
            probability = predict_image(model, image, device, args.img_size, args.mode, args.stride, args.batch_size)
            save_outputs(probability, image, args.output_dir, path.stem, args.binary_threshold)


if __name__ == '__main__':
    main()

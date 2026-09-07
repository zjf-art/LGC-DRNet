from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
    if not files:
        raise ValueError(f'No images found in {directory}')
    stems = [p.stem.casefold() for p in files]
    if len(stems) != len(set(stems)):
        raise ValueError(f'Duplicate image stems in {directory}')
    return files


def resolve_mask(data_dir, stem, dataset=''):
    root = Path(data_dir)
    label_dir = root / 'labels'
    if not label_dir.is_dir():
        label_dir = root / 'masks'
    name = stem.replace('image', 'target') if dataset == 'CamCrack789' else stem
    path = label_dir / (name + '.png')
    if not path.is_file():
        raise FileNotFoundError(f'Missing mask: {path}')
    return path


def read_mask(path):
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim == 3:
        mask = np.array(Image.open(path).convert('L'))
    if mask.ndim != 2:
        raise ValueError(f'Expected a 2D mask: {path}')
    return (mask > 0).astype(np.uint8)


def validate_dataset(data_dir, dataset=''):
    files = list_images(Path(data_dir) / 'images')
    for path in files:
        mask_path = resolve_mask(data_dir, path.stem, dataset)
        with Image.open(path) as image:
            size = image.size
            image.verify()
        mask = read_mask(mask_path)
        if mask.shape != (size[1], size[0]):
            raise ValueError(f'Image/mask size mismatch: {path.name}')
    return files


class PlainCrackTransform:
    def __init__(self, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
        self.normalize = transforms.Normalize(normalize_mean, normalize_std)

    def __call__(self, sample):
        image = torch.from_numpy(sample['image'].transpose(2, 0, 1).astype(np.float32))
        label = torch.from_numpy(sample['label'].astype(np.int64))
        return {'image': self.normalize(image), 'label': label}


class Crack_dataset(Dataset):
    def __init__(self, data_dir, resize=512, transform=None, dataset=''):
        self.data_dir = data_dir
        self.resize = resize
        self.transform = transform
        self.dataset = dataset
        self.files = list_images(Path(data_dir) / 'images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            image = np.asarray(image.convert('RGB').resize((self.resize, self.resize)), dtype=np.float32) / 255.0
        mask = read_mask(resolve_mask(self.data_dir, path.stem, self.dataset))
        label = cv2.resize(mask, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = path.name
        return sample

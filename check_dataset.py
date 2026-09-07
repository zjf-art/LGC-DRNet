import argparse
import hashlib
import json
from pathlib import Path

from datasets.crack_dataset import validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True, help='Training split directory')
    parser.add_argument('--test_dir', required=True, help='Test split directory')
    parser.add_argument('--dataset', default='CrackMap', help='Dataset name; CamCrack789 enables target mask names')
    args = parser.parse_args()
    train = validate_dataset(args.train_dir, args.dataset)
    test = validate_dataset(args.test_dir, args.dataset)
    hashes = {}
    for path in train:
        hashes.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(path.name)
    duplicates = []
    for path in test:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            duplicates.append({'train': hashes[digest], 'test': path.name})
    print(json.dumps({'train_count': len(train), 'test_count': len(test), 'byte_identical_cross_split_images': duplicates}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

import argparse
import os
import random
import copy
import logging
import csv
import json
from pathlib import Path
from models import LGCDRNet
from datasets.crack_dataset import validate_dataset
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_crack

SEED_LIST = [1729, 2825, 3141]

def reset_logging():
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

def save_three_seed_statistics(all_seed_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if len(all_seed_results) == 0:
        print('[Warning] No seed results available for statistics.')
        return {}
    result_csv = os.path.join(save_dir, 'three_seed_best_results.csv')
    result_fields = ['seed', 'best_epoch', 'IoU', 'mIoU', 'Precision', 'Recall', 'ODS', 'OIS', 'F1', 'best_threshold', 'PRF_threshold']
    with open(result_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()
        for result in all_seed_results:
            writer.writerow({'seed': int(result['seed']), 'best_epoch': int(result['best_epoch']), 'IoU': f"{result['IoU']:.6f}", 'mIoU': f"{result['mIoU']:.6f}", 'Precision': f"{result['Precision']:.6f}", 'Recall': f"{result['Recall']:.6f}", 'ODS': f"{result['ODS']:.6f}", 'OIS': f"{result['OIS']:.6f}", 'F1': f"{result['F1']:.6f}", 'best_threshold': f"{result['best_threshold']:.4f}", 'PRF_threshold': f"{result['PRF_threshold']:.4f}"})
    metric_names = ['IoU', 'mIoU', 'Precision', 'Recall', 'ODS', 'OIS', 'F1']
    statistics = {}
    for metric in metric_names:
        values = np.array([float(result[metric]) for result in all_seed_results], dtype=np.float64)
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if len(values) > 1:
            std_value = float(np.std(values, ddof=1))
        else:
            std_value = 0.0
        statistics[metric] = {'mean': mean_value, 'std': std_value, 'median': median_value, 'min': min_value, 'max': max_value}
    statistics_csv = os.path.join(save_dir, 'three_seed_statistics.csv')
    with open(statistics_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Mean±Std'])
        for metric in metric_names:
            stat = statistics[metric]
            writer.writerow([metric, f"{stat['mean']:.6f}", f"{stat['std']:.6f}", f"{stat['median']:.6f}", f"{stat['min']:.6f}", f"{stat['max']:.6f}", f"{stat['mean']:.4f} ± {stat['std']:.4f}"])
    summary_txt = os.path.join(save_dir, 'three_seed_summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('=' * 110 + '\n')
        f.write('Three-Seed Experiment Summary\n')
        f.write('=' * 110 + '\n\n')
        f.write('Seeds: ' + ', '.join((str(result['seed']) for result in all_seed_results)) + '\n\n')
        f.write('Best result of each seed (selected by best mIoU epoch)\n')
        f.write('-' * 110 + '\n')
        for result in all_seed_results:
            f.write(f"Seed {int(result['seed']):4d} | Best Epoch {int(result['best_epoch']):3d} | IoU {result['IoU']:.6f} | mIoU {result['mIoU']:.6f} | P {result['Precision']:.6f} | R {result['Recall']:.6f} | ODS {result['ODS']:.6f} | OIS {result['OIS']:.6f} | F1 {result['F1']:.6f} | mIoUThr {result['best_threshold']:.4f} | F1Thr {result['PRF_threshold']:.4f}\n")
        f.write('\n')
        f.write('-' * 110 + '\n')
        f.write('Mean ± Std (sample standard deviation, ddof=1)\n')
        f.write('-' * 110 + '\n')
        for metric in metric_names:
            stat = statistics[metric]
            f.write(f"{metric:<10}: {stat['mean']:.6f} ± {stat['std']:.6f} | Median={stat['median']:.6f} | Min={stat['min']:.6f} | Max={stat['max']:.6f}\n")
        f.write('\n')
        f.write('Paper-style results (4 decimals)\n')
        f.write('-' * 110 + '\n')
        for metric in metric_names:
            stat = statistics[metric]
            f.write(f"{metric:<10}: {stat['mean']:.4f} ± {stat['std']:.4f}\n")
    print('\n' + '=' * 110)
    print('3-SEED FINAL STATISTICS')
    print('=' * 110)
    for result in all_seed_results:
        print(f"Seed {int(result['seed']):4d} | Epoch {int(result['best_epoch']):3d} | IoU {result['IoU']:.4f} | mIoU {result['mIoU']:.4f} | P {result['Precision']:.4f} | R {result['Recall']:.4f} | ODS {result['ODS']:.4f} | OIS {result['OIS']:.4f} | F1 {result['F1']:.4f}")
    print('-' * 110)
    print('Mean ± Std (ddof=1)')
    print('-' * 110)
    for metric in metric_names:
        stat = statistics[metric]
        print(f"{metric:<10}: {stat['mean']:.4f} ± {stat['std']:.4f} (Median={stat['median']:.4f}, Min={stat['min']:.4f}, Max={stat['max']:.4f})")
    print('=' * 110)
    print(f'Per-seed results : {result_csv}')
    print(f'Statistics       : {statistics_csv}')
    print(f'Summary          : {summary_txt}')
    print('=' * 110)
    return statistics
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='./dataset/CrackMap/train', help='training data directory')
parser.add_argument('--test_dir', type=str, default='./dataset/CrackMap/test', help='testing data directory')
parser.add_argument('--dataset', type=str, default='CrackMap', help='CamCrack789, Crack500, CrackMap, DeepCrack, fine_crack_530 ')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./output/CrackMap_3seed_pretrained/', help='output dir')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input size, at least 32 and divisible by 16')
parser.add_argument('--freeze_backbone_bn', type=int, default=1, choices=[0, 1], help='freeze MobileNetV3 backbone BN running statistics during training')
parser.add_argument('--binary_threshold', type=float, default=0.5, help='threshold only for saving *_binary.png visualization masks; metrics still use *_pre.png probability maps')
parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['cosine', 'poly'], help='learning rate scheduler: cosine or poly')
parser.add_argument('--ce_weight', type=float, default=0.5, help='weight for CE loss')
parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1, help='Load local MobileNetV3 backbone weights')
parser.add_argument('--backbone_weights', type=str, default=r'.\vit_checkpoint\mobilenet_v3_large-8738ca79.pth', help='Path to full torchvision MobileNetV3-Large state_dict')
parser.add_argument('--drop_last', type=int, choices=[0, 1], default=1, help='Drop incomplete training batch, matching original training')
parser.add_argument('--device', type=str, default='auto', help='auto, cpu, or cuda:0')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.num_classes != 2 or args.img_size < 32 or args.img_size % 16:
        parser.error('Use num_classes=2 and img_size >=32 divisible by 16.')
    if args.max_epochs < 1 or args.batch_size < 1 or args.base_lr <= 0:
        parser.error('epochs, batch size and learning rate must be positive.')
    if not 0 <= args.binary_threshold <= 1 or min(args.ce_weight, args.dice_weight) < 0 or args.ce_weight + args.dice_weight <= 0:
        parser.error('Invalid threshold or loss weights.')
    if args.pretrained and not args.backbone_weights:
        parser.error('--pretrained 1 requires --backbone_weights.')
    validate_dataset(args.train_dir, args.dataset)
    validate_dataset(args.test_dir, args.dataset)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    all_seed_results = []
    for seed in SEED_LIST:
        print('\n' + '=' * 80)
        print(f'Start training with seed = {seed}')
        print('=' * 80)
        reset_logging()
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        if not run_args.deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
        random.seed(run_args.seed)
        np.random.seed(run_args.seed)
        torch.manual_seed(run_args.seed)
        torch.cuda.manual_seed(run_args.seed)
        torch.cuda.manual_seed_all(run_args.seed)
        dataset_config = {'fine_crack_530': 2, 'crack': 2, 'crack500': 2, 'deepcrack': 2, 'CamCrack789': 2, 'CrackMap': 2}
        if run_args.dataset in dataset_config:
            run_args.num_classes = dataset_config[run_args.dataset]
        run_name = f'{run_args.dataset}_seed{run_args.seed}_lr{run_args.base_lr}_bs{run_args.batch_size}_{run_args.lr_scheduler}_ce{run_args.ce_weight}_dice{run_args.dice_weight}_bin{run_args.binary_threshold}_pos'
        run_args.output_dir = os.path.join(base_output_dir, run_name)
        os.makedirs(run_args.output_dir, exist_ok=True)
        run_args.test_save_path = run_args.output_dir
        net = LGCDRNet(num_classes=2, pretrained=bool(run_args.pretrained), weight_path=run_args.backbone_weights, freeze_backbone_bn=bool(run_args.freeze_backbone_bn)).to(device)
        config_path = Path(run_args.output_dir) / 'config.json'
        if config_path.exists():
            raise FileExistsError(f'Run already exists: {run_args.output_dir}. Choose another --output_dir.')
        config_path.write_text(json.dumps(vars(run_args), indent=2), encoding='utf-8')
        snapshot_path = os.path.join(run_args.output_dir, 'weights')
        os.makedirs(snapshot_path, exist_ok=True)
        run_args.save_images_dir = os.path.join(run_args.output_dir, 'images')
        os.makedirs(run_args.save_images_dir, exist_ok=True)
        print('Current seed       :', run_args.seed)
        print('Current num_classes:', run_args.num_classes)
        print('Current batch_size :', run_args.batch_size)
        print('Output dir         :', run_args.output_dir)
        best_result = trainer_crack(run_args, net, snapshot_path, device=device)
        best_result['seed'] = int(seed)
        all_seed_results.append(best_result)
        Path(run_args.output_dir, 'best_metrics.json').write_text(json.dumps(best_result, indent=2), encoding='utf-8')
        print('\n' + '-' * 100)
        print(f'Best result of seed {seed}')
        print('-' * 100)
        print(f"Epoch: {best_result['best_epoch']} | IoU: {best_result['IoU']:.4f} | mIoU: {best_result['mIoU']:.4f} | P: {best_result['Precision']:.4f} | R: {best_result['Recall']:.4f} | ODS: {best_result['ODS']:.4f} | OIS: {best_result['OIS']:.4f} | F1: {best_result['F1']:.4f} | mIoUThr: {best_result['best_threshold']:.2f} | F1Thr: {best_result['PRF_threshold']:.2f}")
        print('-' * 100)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'[Finished] seed = {seed}')
        print(f'Results saved to: {run_args.output_dir}')
    print('\n' + '=' * 80)
    print('All seed experiments finished.')
    print('=' * 80)
    if len(all_seed_results) > 0:
        save_three_seed_statistics(all_seed_results=all_seed_results, save_dir=base_output_dir)

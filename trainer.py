import logging
import os
import shutil
import sys
import csv
import cv2
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import DiceLoss
from datasets.crack_dataset import list_images, resolve_mask, read_mask

def get_statistics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        final_accuracy_all.append([thresh, precision, recall, f1])
    return final_accuracy_all

def cal_best_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = np.array(cal_prf_metrics(pred_list, gt_list, thresh_step=thresh_step))
    best_idx = int(np.argmax(final_accuracy_all[:, 3]))
    return {'PRF_threshold': float(final_accuracy_all[best_idx, 0]), 'Precision': float(final_accuracy_all[best_idx, 1]), 'Recall': float(final_accuracy_all[best_idx, 2]), 'F1': float(final_accuracy_all[best_idx, 3])}

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            F1_list.append(f1)
        final_F1_list.append(np.max(np.array(F1_list)))
    return np.sum(np.array(final_F1_list)) / len(final_F1_list)

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            ODS_list.append(f1)
        final_ODS.append(np.mean(np.array(ODS_list)))
    return np.max(np.array(final_ODS))

def cal_mIoU_IoU_metrics(pred_list, gt_list, thresh_step=0.01):
    best_miou = -1.0
    best_iou_crack = 0.0
    best_iou_background = 0.0
    best_threshold = 0.0
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_crack_list = []
        iou_background_list = []
        miou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            iou_crack = 0.0 if TP + FP + FN <= 0 else TP / (TP + FP + FN)
            iou_background = 0.0 if TN + FP + FN <= 0 else TN / (TN + FP + FN)
            miou = (iou_crack + iou_background) / 2.0
            iou_crack_list.append(iou_crack)
            iou_background_list.append(iou_background)
            miou_list.append(miou)
        mean_miou = float(np.mean(np.array(miou_list)))
        mean_iou_crack = float(np.mean(np.array(iou_crack_list)))
        mean_iou_background = float(np.mean(np.array(iou_background_list)))
        if mean_miou > best_miou:
            best_miou = mean_miou
            best_iou_crack = mean_iou_crack
            best_iou_background = mean_iou_background
            best_threshold = float(thresh)
    return {'mIoU': float(best_miou), 'IoU': float(best_iou_crack), 'IoU_background': float(best_iou_background), 'best_threshold': float(best_threshold)}

def get_test_label_dir(args):
    label_dir = os.path.join(args.test_dir, 'labels')
    mask_dir = os.path.join(args.test_dir, 'masks')
    if os.path.isdir(label_dir):
        return label_dir
    if os.path.isdir(mask_dir):
        return mask_dir
    return label_dir

def get_train_label_dir(data_dir):
    label_dir = os.path.join(data_dir, 'labels')
    mask_dir = os.path.join(data_dir, 'masks')
    if os.path.isdir(label_dir):
        return label_dir
    if os.path.isdir(mask_dir):
        return mask_dir
    return label_dir

def cal_single_image_metrics(pred_prob_255, gt_255, threshold=0.5):
    pred_prob = pred_prob_255.astype(np.float32) / 255.0
    gt = (gt_255 > 127).astype(np.uint8)
    pred = (pred_prob >= threshold).astype(np.uint8)
    TP = int(np.sum((pred == 1) & (gt == 1)))
    TN = int(np.sum((pred == 0) & (gt == 0)))
    FP = int(np.sum((pred == 1) & (gt == 0)))
    FN = int(np.sum((pred == 0) & (gt == 1)))
    precision = 1.0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0.0 if TP + FN == 0 else TP / (TP + FN)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    iou_crack = 0.0 if TP + FP + FN == 0 else TP / (TP + FP + FN)
    iou_background = 0.0 if TN + FP + FN == 0 else TN / (TN + FP + FN)
    miou = (iou_crack + iou_background) / 2.0
    total = TP + TN + FP + FN
    accuracy = 0.0 if total == 0 else (TP + TN) / total
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'Precision': float(precision), 'Recall': float(recall), 'F1': float(f1), 'IoU_crack': float(iou_crack), 'IoU_background': float(iou_background), 'mIoU': float(miou), 'Accuracy': float(accuracy), 'gt_crack_pixels': int(np.sum(gt == 1)), 'pred_crack_pixels': int(np.sum(pred == 1))}

def cal_single_image_best_f1(pred_prob_255, gt_255, thresh_step=0.01):
    pred_prob = pred_prob_255.astype(np.float32) / 255.0
    gt = (gt_255 > 127).astype(np.uint8)
    best_f1 = 0.0
    best_threshold = 0.0
    for threshold in np.arange(0.0, 1.0, thresh_step):
        pred = (pred_prob > threshold).astype(np.uint8)
        TP = np.sum((pred == 1) & (gt == 1))
        FP = np.sum((pred == 1) & (gt == 0))
        FN = np.sum((pred == 0) & (gt == 1))
        precision = 1.0 if TP + FP == 0 else TP / (TP + FP)
        recall = 0.0 if TP + FN == 0 else TP / (TP + FN)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return (float(best_f1), float(best_threshold))

def save_per_image_metrics_csv(rows, save_path):
    if len(rows) == 0:
        return
    fieldnames = ['image_name', 'mask_name', 'threshold', 'Precision', 'Recall', 'F1', 'IoU_crack', 'IoU_background', 'mIoU', 'Accuracy', 'OIS_F1', 'OIS_best_threshold', 'gt_crack_pixels', 'pred_crack_pixels', 'TP', 'FP', 'FN', 'TN']
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def evaluate_from_saved_pngs(save_dir, args):
    pred_files = sorted([f for f in os.listdir(save_dir) if f.endswith('_pre.png')])
    test_label_dir = get_test_label_dir(args)
    pred_list, gt_list = ([], [])
    per_image_rows = []
    fixed_threshold = float(getattr(args, 'binary_threshold', 0.5))
    for pred_name in pred_files:
        root_name = pred_name[:-8]
        label_file = str(resolve_mask(args.test_dir, root_name, args.dataset))
        mask_name = os.path.basename(label_file)
        pred = cv2.imread(os.path.join(save_dir, pred_name), cv2.IMREAD_GRAYSCALE)
        mask_orig = read_mask(label_file)
        if pred is None or mask_orig is None:
            raise ValueError(f'Failed to read: {pred_name} or {mask_name}')
        mask_resized = cv2.resize(mask_orig, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        gt = (mask_resized > 0).astype(np.uint8) * 255
        pred_float = pred.astype(np.float32)
        gt_float = gt.astype(np.float32)
        pred_list.append(pred_float)
        gt_list.append(gt_float)
        single_metrics = cal_single_image_metrics(pred_prob_255=pred_float, gt_255=gt_float, threshold=fixed_threshold)
        ois_f1, ois_best_threshold = cal_single_image_best_f1(pred_prob_255=pred_float, gt_255=gt_float, thresh_step=0.01)
        row = {'image_name': root_name, 'mask_name': mask_name, 'threshold': fixed_threshold, 'Precision': round(single_metrics['Precision'], 6), 'Recall': round(single_metrics['Recall'], 6), 'F1': round(single_metrics['F1'], 6), 'IoU_crack': round(single_metrics['IoU_crack'], 6), 'IoU_background': round(single_metrics['IoU_background'], 6), 'mIoU': round(single_metrics['mIoU'], 6), 'Accuracy': round(single_metrics['Accuracy'], 6), 'OIS_F1': round(ois_f1, 6), 'OIS_best_threshold': round(ois_best_threshold, 4), 'gt_crack_pixels': single_metrics['gt_crack_pixels'], 'pred_crack_pixels': single_metrics['pred_crack_pixels'], 'TP': single_metrics['TP'], 'FP': single_metrics['FP'], 'FN': single_metrics['FN'], 'TN': single_metrics['TN']}
        per_image_rows.append(row)
    per_image_csv = os.path.join(save_dir, 'per_image_metrics.csv')
    save_per_image_metrics_csv(per_image_rows, per_image_csv)
    if len(pred_list) == 0:
        raise ValueError("No predictions available for evaluation.")
    prf_metrics = cal_best_prf_metrics(pred_list, gt_list, thresh_step=0.01)
    miou_iou_metrics = cal_mIoU_IoU_metrics(pred_list, gt_list, thresh_step=0.01)
    mIoU = miou_iou_metrics['mIoU']
    IoU = miou_iou_metrics['IoU']
    best_threshold = miou_iou_metrics['best_threshold']
    ODS = cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01)
    OIS = cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01)
    return {'IoU': float(IoU), 'mIoU': float(mIoU), 'Precision': float(prf_metrics['Precision']), 'Recall': float(prf_metrics['Recall']), 'ODS': float(ODS), 'OIS': float(OIS), 'F1': float(prf_metrics['F1']), 'best_threshold': float(best_threshold), 'PRF_threshold': float(prf_metrics['PRF_threshold'])}

def save_predictions(args, model, epoch_num, save_dir):
    test_image_dir = os.path.join(args.test_dir, 'images')
    test_label_dir = get_test_label_dir(args)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    binary_dir = os.path.join(save_dir, 'binary')
    lab_dir = os.path.join(save_dir, 'lab')
    os.makedirs(binary_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    sample_list = [p.name for p in list_images(test_image_dir)]
    transform = transforms.Compose([transforms.Resize(size=(args.img_size, args.img_size), antialias=True), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    model.eval()
    with torch.no_grad():
        for file_name in tqdm(sample_list, desc=f'Epoch {epoch_num + 1} test', ncols=80):
            image_file = os.path.join(test_image_dir, file_name)
            root_name = file_name.rsplit('.', 1)[0]
            label_file = str(resolve_mask(args.test_dir, root_name, args.dataset))
            image = Image.open(image_file).convert('RGB')
            patch_image = np.array(image).transpose(2, 0, 1) / 255.0
            patch_image = torch.from_numpy(patch_image.astype(np.float32))
            patch_image = transform(patch_image).unsqueeze(0).to(args.device)
            outputs = model(patch_image)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if outputs.shape[1] == 1:
                pred_map = torch.sigmoid(outputs)[0, 0, :, :].detach().cpu().numpy()
            else:
                pred_map = torch.softmax(outputs, dim=1)[0, 1, :, :].detach().cpu().numpy()
            pred_map = np.clip(pred_map, 0.0, 1.0)
            pred_prob_img = np.rint(pred_map * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f'{root_name}_pre.png'), pred_prob_img)
            binary_threshold = float(getattr(args, 'binary_threshold', 0.5))
            pred_binary_img = (pred_map >= binary_threshold).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(binary_dir, f'{root_name}_binary.png'), pred_binary_img)
            mask_orig = read_mask(label_file)
            if mask_orig is not None:
                mask_resized = cv2.resize(mask_orig, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
                lab_img = (mask_resized > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(lab_dir, f'{root_name}_lab.png'), lab_img.astype(np.uint8))
            else:
                print(f'[Warning] Label not found when saving lab: {label_file}')

def trainer_crack(args, model, snapshot_path, device):
    from datasets.crack_dataset import Crack_dataset, PlainCrackTransform
    log_file = os.path.join(snapshot_path, 'log.txt')
    if sys.version_info >= (3, 9):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', encoding='utf-8')
    else:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(model)
    args.device = device
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    db_train = Crack_dataset(data_dir=args.train_dir, resize=args.img_size, dataset=args.dataset, transform=PlainCrackTransform(normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)))
    print('The length of train set is: {}'.format(len(db_train)))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", drop_last=bool(args.drop_last))
    if len(trainloader) == 0:
        raise ValueError("No training batches; reduce batch_size or set --drop_last 0.")
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)
    if hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'poly':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / args.max_epochs) ** 0.9)
        logging.info('Using Poly learning rate scheduler.')
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-05)
        logging.info('Using Cosine Annealing learning rate scheduler.')
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    iter_num = 0
    best_miou = -1.0
    best_epoch = 0
    best_metrics = {}
    result_file = os.path.join(snapshot_path, 'epoch_result.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write('Epoch\tIoU\tmIoU\tPrecision\tRecall\tODS\tOIS\tF1\tmIoUThr\tF1Thr\n')
        f.write('-' * 90 + '\n')
    temp_eval_dir = os.path.join(snapshot_path, 'epoch_eval_pngs')
    os.makedirs(temp_eval_dir, exist_ok=True)
    logging.info('{} iterations per epoch. {} max iterations '.format(len(trainloader), max_epoch * len(trainloader)))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = (sampled_batch['image'], sampled_batch['label'])
            image_batch, label_batch = (image_batch.to(device), label_batch.to(device))
            outputs = model(image_batch)
            if isinstance(outputs, (list, tuple)):
                loss_ce = 0.0
                loss_dice = 0.0
                for output in outputs:
                    loss_ce += ce_loss(output, label_batch.long())
                    loss_dice += dice_loss(output, label_batch, softmax=True)
                loss_ce = loss_ce / len(outputs)
                loss_dice = loss_dice / len(outputs)
                outputs_for_vis = outputs[0]
            else:
                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                outputs_for_vis = outputs
            ce_w = args.ce_weight if hasattr(args, 'ce_weight') else 0.5
            dice_w = args.dice_weight if hasattr(args, 'dice_weight') else 0.5
            loss = ce_w * loss_ce + dice_w * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/lr', current_lr, iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)
            if iter_num % 10 == 0:
                logging.info('epoch: %d, iter: %d, loss : %f, loss_ce: %f' % (epoch_num + 1, iter_num, loss.item(), loss_ce.item()))
            if iter_num % 400 == 0 and image_batch.shape[0] > 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-08)
                writer.add_image('train/Image', image, iter_num)
                pred_vis = torch.argmax(torch.softmax(outputs_for_vis, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred_vis[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        lr_scheduler.step()
        save_predictions(args, model, epoch_num, temp_eval_dir)
        metrics = evaluate_from_saved_pngs(temp_eval_dir, args)
        logging.info('【Epoch %d Test】IoU: %.4f | mIoU: %.4f | P: %.4f | R: %.4f | ODS: %.4f | OIS: %.4f | F1: %.4f | mIoUThr: %.2f | F1Thr: %.2f' % (epoch_num + 1, metrics['IoU'], metrics['mIoU'], metrics['Precision'], metrics['Recall'], metrics['ODS'], metrics['OIS'], metrics['F1'], metrics['best_threshold'], metrics['PRF_threshold']))
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch_num + 1}\t{metrics['IoU']:.4f}\t{metrics['mIoU']:.4f}\t{metrics['Precision']:.4f}\t{metrics['Recall']:.4f}\t{metrics['ODS']:.4f}\t{metrics['OIS']:.4f}\t{metrics['F1']:.4f}\t{metrics['best_threshold']:.2f}\t{metrics['PRF_threshold']:.2f}\n")
        writer.add_scalar('test/IoU', metrics['IoU'], epoch_num + 1)
        writer.add_scalar('test/mIoU', metrics['mIoU'], epoch_num + 1)
        writer.add_scalar('test/Precision', metrics['Precision'], epoch_num + 1)
        writer.add_scalar('test/Recall', metrics['Recall'], epoch_num + 1)
        writer.add_scalar('test/ODS', metrics['ODS'], epoch_num + 1)
        writer.add_scalar('test/OIS', metrics['OIS'], epoch_num + 1)
        writer.add_scalar('test/F1', metrics['F1'], epoch_num + 1)
        writer.add_scalar('test/mIoU_threshold', metrics['best_threshold'], epoch_num + 1)
        writer.add_scalar('test/F1_threshold', metrics['PRF_threshold'], epoch_num + 1)
        savefile_name = 'epoch_{}_IoU_{:.3f}_mIoU_{:.3f}_ODS_{:.3f}.pth'.format(epoch_num + 1, metrics['IoU'], metrics['mIoU'], metrics['ODS'])
        save_mode_path = os.path.join(snapshot_path, savefile_name)
        torch.save(model.state_dict(), save_mode_path)
        logging.info('save model to {}'.format(save_mode_path))
        if metrics['mIoU'] > best_miou:
            best_miou = metrics['mIoU']
            best_epoch = epoch_num + 1
            best_metrics = metrics
            logging.info(f'[*] Found Best mIoU: {best_miou:.4f}! Updating images to {args.save_images_dir}...')
            if os.path.exists(args.save_images_dir):
                shutil.rmtree(args.save_images_dir)
            shutil.copytree(temp_eval_dir, args.save_images_dir)
            best_model_path = os.path.join(snapshot_path, 'best_mIoU.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info('save best model to {}'.format(best_model_path))
    writer.close()
    best_summary = f"[SUCCESS] Training Finished! Best Params -> Epoch: {best_epoch} | IoU: {best_metrics['IoU']:.4f} | mIoU: {best_metrics['mIoU']:.4f} | Precision: {best_metrics['Precision']:.4f} | Recall: {best_metrics['Recall']:.4f} | ODS: {best_metrics['ODS']:.4f} | OIS: {best_metrics['OIS']:.4f} | F1: {best_metrics['F1']:.4f} | mIoUThr: {best_metrics['best_threshold']:.2f} | F1Thr: {best_metrics['PRF_threshold']:.2f}"
    logging.info('-' * 90)
    logging.info(best_summary)
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write('-' * 90 + '\n')
        f.write(best_summary + '\n')
    if os.path.exists(temp_eval_dir):
        shutil.rmtree(temp_eval_dir)
    return {'best_epoch': int(best_epoch), 'IoU': float(best_metrics['IoU']), 'mIoU': float(best_metrics['mIoU']), 'Precision': float(best_metrics['Precision']), 'Recall': float(best_metrics['Recall']), 'ODS': float(best_metrics['ODS']), 'OIS': float(best_metrics['OIS']), 'F1': float(best_metrics['F1']), 'best_threshold': float(best_metrics['best_threshold']), 'PRF_threshold': float(best_metrics['PRF_threshold'])}

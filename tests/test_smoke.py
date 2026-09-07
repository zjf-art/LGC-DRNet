import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.crack_dataset import validate_dataset
from inference import load_model, predict_image
from models import LGCDRNet
from trainer import cal_best_prf_metrics, cal_mIoU_IoU_metrics, cal_ODS_metrics, cal_OIS_metrics
from utils import DiceLoss


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_backward_and_bn(self):
        model = LGCDRNet(pretrained=False).train()
        self.assertTrue(all(not layer.training for layer in model._backbone_bn_modules()))
        inputs = torch.randn(1, 3, 32, 32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (1, 2, 32, 32))
        labels = torch.randint(0, 2, (1, 32, 32))
        loss = 0.5 * torch.nn.functional.cross_entropy(outputs, labels) + 0.5 * DiceLoss(2)(outputs, labels, softmax=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.seg_head.weight.grad).all())
        unfrozen = LGCDRNet(pretrained=False, freeze_backbone_bn=False).train()
        self.assertTrue(all(layer.training for layer in unfrozen._backbone_bn_modules()))

    def test_checkpoint_and_sliding(self):
        model = LGCDRNet(pretrained=False).eval()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pth'
            torch.save(model.state_dict(), checkpoint)
            restored = load_model(checkpoint, torch.device('cpu'))
            image = Image.fromarray(np.zeros((41, 55, 3), dtype=np.uint8))
            result = predict_image(restored, image, 'cpu', 32, 'sliding', 24, 2)
            self.assertEqual(result.shape, (41, 55))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(((result >= 0) & (result <= 1)).all())
            small = predict_image(restored, image.resize((17, 19)), 'cpu', 32, 'sliding', 24, 1)
            self.assertEqual(small.shape, (19, 17))

    def test_metrics(self):
        gt = np.array([[0, 255], [255, 0]], dtype=np.float32)
        self.assertEqual(cal_best_prf_metrics([gt], [gt])['F1'], 1.0)
        self.assertEqual(cal_mIoU_IoU_metrics([gt], [gt])['mIoU'], 1.0)
        self.assertEqual(cal_ODS_metrics([gt], [gt]), 1.0)
        self.assertEqual(cal_OIS_metrics([gt], [gt]), 1.0)
        zero = np.zeros_like(gt)
        self.assertEqual(cal_best_prf_metrics([zero], [gt])['F1'], 0.0)
        self.assertEqual(cal_mIoU_IoU_metrics([zero], [zero])['mIoU'], 0.5)

    def test_dataset_mapping_and_missing_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'images').mkdir()
            (root / 'masks').mkdir()
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root / 'images' / 'image-001.jpg')
            mask = root / 'masks' / 'target-001.png'
            Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(mask)
            self.assertEqual(len(validate_dataset(root, 'CamCrack789')), 1)
            mask.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_dataset(root, 'CamCrack789')


if __name__ == '__main__':
    unittest.main()

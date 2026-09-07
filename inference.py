from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import LGCDRNet


def load_model(checkpoint, device):
    state = torch.load(
        checkpoint,
        map_location='cpu',
        weights_only=True
    )

    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break

    state = {
        k[len('module.'):] if k.startswith('module.') else k: v
        for k, v in state.items()
    }

    model = LGCDRNet(pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def image_tensor(image, size=None):
    value = torch.from_numpy(
        np.array(
            image.convert('RGB'),
            dtype=np.float32
        ).transpose(2, 0, 1) / 255.0
    )

    if size is not None:
        value = transforms.Resize(
            (size, size),
            antialias=True
        )(value)

    return transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )(value)


def starts(length, patch, stride):
    if length <= patch:
        return [0]

    return sorted(
        set(
            list(range(0, length - patch + 1, stride))
            + [length - patch]
        )
    )


@torch.inference_mode()
def predict_image(
    model,
    image,
    device,
    size=512,
    mode='resize',
    stride=384,
    batch_size=1
):
    if mode == 'resize':
        inputs = image_tensor(image, size).unsqueeze(0).to(device)
        return model(inputs).softmax(1)[0, 1].cpu().numpy()

    value = image_tensor(image)
    _, height, width = value.shape

    pad_h = max(0, size - height)
    pad_w = max(0, size - width)

    value = torch.nn.functional.pad(
        value,
        (0, pad_w, 0, pad_h),
        mode='replicate'
    )

    coordinates = [
        (y, x)
        for y in starts(value.shape[1], size, stride)
        for x in starts(value.shape[2], size, stride)
    ]

    total = np.zeros(value.shape[1:], dtype=np.float32)
    count = np.zeros_like(total)

    for offset in range(0, len(coordinates), batch_size):
        group = coordinates[offset:offset + batch_size]

        inputs = torch.stack([
            value[:, y:y + size, x:x + size]
            for y, x in group
        ]).to(device)

        probs = model(inputs).softmax(1)[:, 1].cpu().numpy()

        for (y, x), prob in zip(group, probs):
            total[y:y + size, x:x + size] += prob
            count[y:y + size, x:x + size] += 1

    return (total / count)[:height, :width]


def save_outputs(probability, image, output, stem, threshold):
    output = Path(output)

    for folder in ('probability', 'binary', 'overlay'):
        (output / folder).mkdir(parents=True, exist_ok=True)

    probability = np.clip(probability, 0, 1)
    binary = probability >= threshold

    Image.fromarray(
        np.rint(probability * 255).astype(np.uint8)
    ).save(
        output / 'probability' / '{}_pre.png'.format(stem)
    )

    Image.fromarray(
        binary.astype(np.uint8) * 255
    ).save(
        output / 'binary' / '{}_binary.png'.format(stem)
    )

    rgb = np.array(
        image.convert('RGB').resize(
            (probability.shape[1], probability.shape[0])
        ),
        dtype=np.float32
    )

    rgb[binary] = (
        rgb[binary] * 0.45
        + np.array([255, 0, 0], dtype=np.float32) * 0.55
    )

    Image.fromarray(
        np.rint(rgb).astype(np.uint8)
    ).save(
        output / 'overlay' / '{}_overlay.png'.format(stem)
    )
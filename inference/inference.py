import os
import sys
import torch
import random
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '..')
from mask import Masker
from config import *
from modules import generate_model


def define_preprocess():
    input_size = DATA_CONFIG['input_size']
    mean = DATA_CONFIG['mean']
    std = DATA_CONFIG['std']
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return preprocess


def load_model(weights_path, device):
    device = BASIC_CONFIG['device']
    model, _ = generate_model(
        device,
        BASIC_CONFIG['pretrained'],
        checkpoint=weights_path,
    )
    return model


def predict(model, batch_size, preprocess, device, imgs, npy_folder, grid_size, patch_size):
    model.eval()
    torch.set_grad_enabled(False)

    masker = Masker(width=grid_size, pixel_size=patch_size, mode='random')
    progress = tqdm(imgs)
    total_diff = []

    img = None
    for num, path in enumerate(progress):
        image = Image.open(path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)

        if img is None:
            img = image
            if num != len(imgs) - 1:
                continue
        if img.shape[0] <= batch_size:
            img = torch.cat([img, image], dim=0)
            if not(num == len(imgs) - 1 or img.shape[0] == batch_size):
                continue

        diffs = []
        for step in range(0, masker.n_masks):
            if not (step % (patch_size / 2) == 0 and (step // grid_size) % (patch_size / 2) == 0):
                continue
            noisy, mask = masker.mask(img, step)
            recon_img = model(noisy)

            diffs.append(((img - recon_img) * mask).unsqueeze(1).abs())
        diff = torch.cat(diffs, 1).mean(1).cpu().numpy()

        total_diff += list(diff)
        img = None
    np.save(npy_folder, total_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '--weights-path',
        type=str,
        help='Path to the model file.'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        help='Path to the EyeQ test folder.'
    )
    parser.add_argument(
        '--diff-dir',
        type=str,
        help='Path to save residual map.'
    )
    args = parser.parse_args()

    random.seed(0)
    weights_path = args.weights_path
    eyeQ_test_folder = args.test_dir
    diffs_save_folder = args.diff_dir

    device = 'cuda'
    model = load_model(weights_path, device)
    preprocess = define_preprocess()
    grid_size = TRAIN_CONFIG['grid_size']
    patch_size = TRAIN_CONFIG['patch_size']
    batch_size = TRAIN_CONFIG['batch_size']
    for i in range(5):
        src = '{}/{}'.format(eyeQ_test_folder, i)
        npy_folder = '{}/grade_{}.npy'.format(diffs_save_folder, i)
        imgs = [os.path.join(src, img) for img in os.listdir(src)]

        result = predict(model, batch_size, preprocess, device, imgs, npy_folder, grid_size, patch_size)

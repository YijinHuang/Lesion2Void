import os
import random
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

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


def predict(model, batch_size, preprocess, device, imgs, npy_folder, grid_size, pixel_size):
    model.eval()
    torch.set_grad_enabled(False)

    masker = Masker(width=grid_size, pixel_size=pixel_size, mode='random')
    progress = tqdm(imgs)
    total_diff = []

    # mask
    circle = np.zeros((224, 224), dtype=np.uint8)
    circle = cv.circle(circle, (112, 112), 100, 1, -1)

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
            if not (step % (pixel_size / 2) == 0 and (step // grid_size) % (pixel_size / 2) == 0):
                continue
            noisy, mask = masker.mask(img, step)
            recon_img = model(noisy)

            diffs.append(((img - recon_img) * mask).unsqueeze(1).abs())
        diff = torch.cat(diffs, 1).mean(1).cpu().numpy()

        total_diff += list(diff)
        img = None
    np.save(npy_folder, total_diff)


if __name__ == "__main__":
    random.seed(0)
    weights_path = '/data1/yijin/workspace/result/anomaly_detection/l2v_test/best_validation_weights.pt'
    device = 'cuda'
    model = load_model(weights_path, device)
    preprocess = define_preprocess()
    gride_size = 48
    pixel_size = 16
    batch_size = 64
    for i in range(5):
        src = '/data1/yijin/workspace/dataset/EyeQ/split_EyeQ/test/{}'.format(i)
        npy_folder = './diffs/test_l2v_{}.npy'.format(i)
        imgs = [os.path.join(src, img) for img in os.listdir(src)]

        result = predict(model, batch_size, preprocess, device, imgs, npy_folder, gride_size, pixel_size)

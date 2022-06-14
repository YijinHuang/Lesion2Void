# from https://github.com/czbiohub/noise2self/blob/master/mask.py
import numpy as np
import torch
import random

class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, pixel_size=1, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.n_masks = width ** 2
        self.pixel_size = pixel_size

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i, mode='test'):
        if mode == 'training':
            i = random.randrange(0, self.n_masks)

        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, self.pixel_size, phasex, phasey)
        mask = mask.to(X.device)

        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        elif self.mode == 'random':
            masked = X * mask_inv + (torch.rand(mask.shape).to(X.device) * 2 - 1) * mask
        else:
            raise NotImplementedError

        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):

        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


def pixel_grid_mask(shape, patch_size, pixel_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2] - pixel_size):
        for j in range(shape[-1] - pixel_size):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                for m in range(pixel_size):
                    for n in range(pixel_size):
                        A[i + m, j + n] = 1
    return torch.Tensor(A)


def interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv

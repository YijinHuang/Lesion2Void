import os
import cv2 as cv
import numpy as np
from tqdm import tqdm


def preprocess(path, new_path):
    scale = 512
    try:
        a = cv.imread(path)
        b = np.zeros(a.shape)
        cv.circle(b, (int(a.shape[1]/2), int(a.shape[0]/2)),
                  int(scale/2*0.98), (1, 1, 1), -1, 8, 0)
        aa = cv.addWeighted(a, 4, cv.GaussianBlur(
            a, (0, 0), scale/30), -4, 128)*b+128*(1-b)
        aa = aa.astype(np.int)
        cv.imwrite(new_path, aa)
    except Exception as e:
        print(path)


if __name__ == '__main__':
    src_folder = ''
    dst_folder = ''
    for root, _, files in os.walk(src_folder):
        for f in tqdm(files):
            src_path = os.path.join(root, f)
            dst_path = src_path.replace(src_folder, dst_folder)
            dst_root = os.path.split(dst_folder)[0]
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)
            preprocess(src_path, dst_path)

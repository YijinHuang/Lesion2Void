import os
import shutil

crop_path = '/mnt/yijin/workspace/dataset/EyeQ/split_EyeQ'
mask_path = '/mnt/yijin/workspace/dataset/EyeQ/split_mask'

data_dict = {}
for root, dirs, imgs in os.walk('/mnt/yijin/workspace/dataset/split_eyepacs_hq_512_preprocessed'):
    cat = os.path.split(root)[-1]
    for img in imgs:
        path = os.path.join(root, img)
        data_dict[img] = (cat, path)

for root, dirs, imgs in os.walk('/mnt/yijin/workspace/dataset/EyeQ/EyeQ_Good'):
    path, typ = os.path.split(root)
    _, split = os.path.split(path)
    for img in imgs:
        cat, src_path = data_dict[img]
        if typ == 'crop_good':
            dst_path = os.path.join(crop_path, split, cat)
        else:
            src_path = os.path.join(root, img)
            dst_path = os.path.join(mask_path, split, cat)
        shutil.copy(src_path, dst_path)

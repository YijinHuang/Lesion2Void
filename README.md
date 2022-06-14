# Lesion2Void

This is the pytorch implementation of the paper:

> Y. Huang, W. Huang, W. Luo and X. Tang, "Lesion2Void: Unsupervised Anomaly Detection in Fundus Images", Proceedings of the IEEE 19th International Symposium on Biomedical Imaging (ISBI), Kolkata, India, March 2022.

![](./imgs/framework.png)

We propose an unsupervised anomaly detection framework for diabetic retinopathy (DR) identification from fundus images, named Lesion2Void. Lesion2Void is capable of identifying anomalies in fundus images by only leveraging normal data without any additional annotation during training. We first randomly mask out multiple patches in normal fundus images. Then, a convolutional neural network is trained to reconstruct the corresponding complete images. We make a simple assumption that in a fundus image, lesion patches, if present, are independent of each other and are also independent of their neighboring pixels, whereas normal patches can be predicted based on the information from the neighborhood. Therefore, in the testing phase, an image can be identified as normal or abnormal by measuring the reconstruction errors of the erased patches.



## Usage

### Installation
Required environment:

- python 3.8+
- pytorch 1.5.1
- torchvision 0.6.1
- tensorboard 2.2.1
- sklearn
- tqdm



### Dataset
1. Download [EyeQ](https://github.com/HzFu/EyeQ) dataset. Then use `dataset/crop.py` to remove the black border of images and resize them to 512 x 512.
2. Organize the EyeQ dataset as follow. Here, `val` and `test` directory have the same structure of train.
```
├── your_data_dir
    ├── train
        ├── grade 0
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
        ├── grade 1
            ├── image3.jpg
            ├── image4.jpg
            ├── ...
        ├── grade 2
        ├── ...
    ├── val
    ├── test
```

3. Generate FOV masks for each image using `dataset/fov_mask.py`. (will be updated soon)
4. Define a dict that only contains normal samples as follow. Then save it using pickle.
```
your_data_dict = {
    # all images of grade 0 in train set of EyePACS
    'train': [
        ('path/to/image1', 'path/to/fov_mask1')
        ('path/to/image2', 'path/to/fov_mask2')
        ('path/to/image3', 'path/to/fov_mask3')
        ...
    ],
    # all images of grade 0 in val set of EyePACS
    'val': [
        ('path/to/image4', 'path/to/fov_mask4')
        ('path/to/image5', 'path/to/fov_mask5')
        ...
    ]
}

import pickle
pickle.dump(your_data_dict, open('path/to/pickle/file', 'wb'))
```


### Training
1. Replace the value of 'data_index' in configs.py with 'path/to/pickle/file' and set 'data_path' as null. Update 'save_path' and 'log_path' in config.py. You can update other training configurations and hyper-parameters in config.py for your customized dataset.
2. Run to train:
```
$ CUDA_VISIBLE_DEVICES=x python main.py
```
where 'x' is the id of your GPU.



### Inference
1. Get residual maps of test set using trained reconstruction model:
```
$ cd inference
$ CUDA_VISIBLE_DEVICES=x python inference.py --weights-path /path/to/model_save_folder/best_validation_weights.pt --test-dir /path/to/EyeQ/test/folder --diff-dir /path/to/save/difference/result
```
2. Post-processing and calculate AUC:
```
$ python auc.py --diff-dir /path/to/save/difference/result
```

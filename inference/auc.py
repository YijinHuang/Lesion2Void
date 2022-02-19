import cv2 as cv
import numpy as np
from sklearn import metrics


circle = np.zeros((224, 224))
circle = cv.circle(circle, (112, 112), 100, 1, -1)


def postprocess(diff):
    diff = diff * circle

    img = np.uint8(diff * 255)
    img = cv.threshold(img, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    res, labels = cv.connectedComponents(img)
    mask = np.zeros_like(labels, dtype=np.uint8)

    mask_idx = []
    for idx in range(1, res):
        scores = (labels == idx).sum()
        mask_idx.append((idx, scores))

    mask_idx.sort(key=lambda x: x[1], reverse=True)

    for idx, _ in mask_idx[30:100]:
        mask[labels == idx] = 1
    diff = diff * mask
    return diff[mask == 1].sum()


def main(folder, eval_channel, eval_grade):
    print('Evaluating {}'.format(folder))
    negs = np.load('./diffs/{}_0.npy'.format(folder))
    posses = [np.load('./diffs/{}_{}.npy'.format(folder, i)) for i in eval_grade]

    # Normalization
    for idx in range(len(negs)):
        negs[idx] = (negs[idx] - negs[idx].min()) / (negs[idx].max() - negs[idx].min())
    for j in range(len(posses)):
        posss = posses[j]
        for idx in range(len(posss)):
            posss[idx] = (posss[idx] - posss[idx].min()) / (posss[idx].max() - posss[idx].min())

    # post process
    processed_negs = []
    processed_poss = []
    for sample in negs:
        processed_negs.append(postprocess(sample[eval_channel]).sum())
    for posss in posses:
        processed_poss.append([])
        for sample in posss:
            processed_poss[-1].append(postprocess(sample[eval_channel]).sum())

    # calcuate AUC
    for pos in processed_poss:
        score = auc(processed_negs, pos)
        print('{:.6f}'.format(score.item()))

    score = auc(processed_negs, np.concatenate(processed_poss, axis=0))
    print('{:.6f}'.format(score.item()))


def auc(neg, pos):
    test_0 = neg
    label_0 = np.zeros_like(test_0)
    test_1 = pos
    label_1 = np.ones_like(test_1)

    y = np.concatenate([label_0, label_1], axis=0)
    pred = np.concatenate([test_0, test_1], axis=0)
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    re = metrics.auc(fpr, tpr)
    return re


if __name__ == '__main__':
    folder = 'test_sm'
    eval_channel = 1
    eval_grade = range(1, 5)
    main(folder, eval_channel, eval_grade)


# 0.558523
# 0.586104
# 0.838532
# 0.863557
# 0.603609

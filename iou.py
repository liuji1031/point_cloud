import numba
import numpy as np

# modified from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
@numba.jit(nopython=True)
def iou_two_boxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(xB-xA, 0.)*max(yB-yA, 0.)
    if interArea == 0:
        return 0.
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = np.abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

@numba.jit(nopython=True)
def iou_box_array(iou_matrix, boxes1, boxes2):
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    for i in range(n):
        for j in range(m):
            boxA = boxes1[i]
            boxB = boxes2[j]

            iou_matrix[i, j] = iou_two_boxes(boxA, boxB)
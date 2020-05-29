import os
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

CASE_NUM = 50


def show_image_with_boxes(dt_path, kaist_path, threshold, img_index=-1, img_id=""):
    if img_id == "":
        img_paths = os.listdir(dt_path)
        if img_index == -1:
            img_index = random.randrange(len(img_paths))
        img_id = img_paths[img_index]
    img_set, img_vid, img_name = img_id.split("_", 2)
    img_name = img_name.replace("txt", "jpg")

    with open(os.path.join(dt_path, img_id).replace("\\", "/")) as dt_file:
        dt_data = [line.rstrip('\n') for line in dt_file]

    gt_path = os.path.join(kaist_path, "annotations", "test_improved").replace("\\", "/")
    with open(os.path.join(gt_path, img_id).replace("\\", "/")) as gt_file:
        gt_data = [line.rstrip('\n') for line in gt_file][1:]

    img_rgb_path = os.path.join(kaist_path, "images", img_set, img_vid, "visible", img_name).replace("\\", "/")
    img_ir_path = os.path.join(kaist_path, "images", img_set, img_vid, "lwir", img_name).replace("\\", "/")
    img_rgb = cv2.cvtColor(cv2.imread(img_rgb_path), cv2.COLOR_RGB2BGR)
    img_ir = cv2.cvtColor(cv2.imread(img_ir_path), cv2.COLOR_RGB2BGR)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_rgb)
    axs[1].imshow(img_ir)
    for line in dt_data:
        name, x1, y1, x2, y2, score = line.split()
        x = round(float(x1))
        y = round(float(y1))
        w = round(float(x2) - float(x1))
        h = round(float(y2) - float(y1))
        if name == "person" and float(score) > threshold:
            axs[0].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none'))
            axs[1].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none'))

    for line in gt_data:
        name, x, y, w, h, _ = line.split(maxsplit=5)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if name == "person":
            axs[0].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='green', facecolor='none'))
            axs[1].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='green', facecolor='none'))
        if name == "person?":
            axs[0].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='yellow', facecolor='none'))
            axs[1].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='yellow', facecolor='none'))
        if name == "people":
            axs[0].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', facecolor='none'))
            axs[1].add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', facecolor='none'))

    plt.show()


def bb_iou(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)
    # return the intersection over union value
    return iou


def find_failure_cases(dt_path, gt_path, threshold, save_dir, non_zero=False):
    false_pos = [("temp", float("inf"))]
    false_neg = [("temp", float("inf"))]

    img_paths = os.listdir(dt_path)
    for path in img_paths:
        dt_bboxes = []
        gt_bboxes = []

        with open(os.path.join(dt_path, path).replace("\\", "/")) as dt_file:
            dt_data = [line.rstrip('\n') for line in dt_file]
            for line in dt_data:
                name, x1, y1, x2, y2, score = line.split()
                if name == "person" and float(score) > threshold:
                    dt_bboxes.append([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])

        with open(os.path.join(gt_path, path).replace("\\", "/")) as gt_file:
            gt_data = [line.rstrip('\n') for line in gt_file][1:]
            for line in gt_data:
                name, x, y, w, h, _ = line.split(maxsplit=5)
                if name == "person":
                    gt_bboxes.append([int(x), int(y), int(x) + int(w), int(y) + int(h)])

        if len(dt_bboxes) == 0 or len(gt_bboxes) == 0:
            if non_zero:
                continue
            if len(gt_bboxes) != 0:
                false_neg.insert(0, (path, 0.0))
            elif len(gt_bboxes) == 0:
                false_pos.insert(0, (path, 0.0))
            continue

        ious = np.zeros((len(dt_bboxes), len(gt_bboxes)))
        for i in range(len(dt_bboxes)):
            for j in range(len(gt_bboxes)):
                ious[i, j] = bb_iou(dt_bboxes[i], gt_bboxes[j])
        dt_ious = np.max(ious, axis=1)
        gt_ious = np.max(ious, axis=0)
        dt_worst = np.min(dt_ious)
        gt_worst = np.min(gt_ious)

        if not (dt_worst == 0 and non_zero):
            for i in range(min(CASE_NUM, len(false_pos))):
                if dt_worst < false_pos[i][1]:
                    false_pos.insert(i, (path, dt_worst))
                    break

        if not (gt_worst == 0 and non_zero):
            for i in range(min(CASE_NUM, len(false_neg))):
                if gt_worst < false_neg[i][1] and gt_worst != 0:
                    false_neg.insert(i, (path, gt_worst))
                    break

    with open(os.path.join(save_dir, "false_positives.txt").replace("\\", "/"), "w") as file:
        for i in range(min(CASE_NUM, len(false_pos))):
            if false_pos[i][0] != "temp":
                file.write("{}\n".format(false_pos[i][0]))

    with open(os.path.join(save_dir, "false_negatives.txt").replace("\\", "/"), "w") as file:
        for i in range(min(CASE_NUM, len(false_neg))):
            if false_neg[i][0] != "temp":
                file.write("{}\n".format(false_neg[i][0]))


if __name__ == "__main__":
    dt = "C:/Users/batuy/Desktop/out"
    gt = "C:/Users/batuy/Desktop/datasets/kaist/annotations/test_improved"
    save = "C:/Users/batuy/Desktop/save"
    kaist = "C:/Users/batuy/Desktop/datasets/kaist"
    img = "set07_V000_I02399.txt"
    # find_failure_cases(dt, gt, 0.5, save, True)
    show_image_with_boxes(dt, kaist, 0.5, img_id=img)

import math
import os
import subprocess

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from utils.image import crop_image
from utils.image import draw_gaussian, gaussian_radius

KAIST_NAMES = ["__background__", "person"]
KAIST_MEAN = [0.37869821907139517, 0.4317424733436831, 0.4633410317553122, 0.21049286781343599]
KAIST_STD = [0.29429125096397235, 0.29691923533376036, 0.3112634491120771, 0.1285024211427516]


class KAIST(Dataset):
    def __init__(self, data_dir, split, gaussian=True, img_size=512):
        super(KAIST, self).__init__()
        self.num_classes = 1
        self.class_names = KAIST_NAMES

        self.split = split
        self.data_dir = os.path.join(data_dir, "kaist")
        self.img_dir = os.path.join(self.data_dir, "images")
        _ann_name = {"train": "train_sanitized", "test": "test_improved"}
        self.annot_path = os.path.join(self.data_dir, "annotations", _ann_name[split])
        self.img_paths = os.listdir(self.annot_path)

        self.max_objs = 64
        self.padding = 128
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}

        self.mean = np.array(KAIST_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(KAIST_STD, dtype=np.float32)[None, None, :]

        self.gaussian = gaussian
        self.gaussian_iou = 0.3

        self.num_samples = len(self.img_paths)
        print('Loaded %d %s samples' % (self.num_samples, split))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.img_paths[index]
        img_set, img_vid, img_name = img_id.split("_", 2)
        img_name = img_name.replace("txt", "jpg")
        img_path = os.path.join(self.img_dir, img_set, img_vid)
        img_rgb = cv2.imread(os.path.join(img_path, "visible", img_name), cv2.IMREAD_COLOR)
        img_ir = cv2.imread(os.path.join(img_path, "lwir", img_name), cv2.IMREAD_GRAYSCALE)

        with open(os.path.join(self.annot_path, self.img_paths[index])) as annot_file:
            annot_data = [line.rstrip('\n') for line in annot_file][1:]

        bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
        if len(annot_data) != 0:
            bboxes = bboxes.repeat(len(annot_data), axis=0)
            for i in range(len(annot_data)):
                line_data = annot_data[i].split()
                label = line_data[0]
                if self.split == "train":
                    if label not in ["person", "person?", "people"]:
                        continue
                elif label != "person":
                    continue
                bboxes[i, :] = list(map(int, line_data[1:5]))

        bboxes[:, 2:] += bboxes[:, :2]

        # resize image and bbox
        height, width = img_rgb.shape[:2]
        img_rgb = cv2.resize(img_rgb, (self.img_size['w'], self.img_size['h']))
        img_ir = cv2.resize(img_ir, (self.img_size['w'], self.img_size['h']))
        img_ir = np.expand_dims(img_ir, axis=2)
        bboxes[:, 0::2] *= self.img_size['w'] / width
        bboxes[:, 1::2] *= self.img_size['h'] / height

        # discard non-valid bboxes
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.img_size['w'] - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.img_size['h'] - 1)
        keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0, (bboxes[:, 3] - bboxes[:, 1]) > 0)
        bboxes = bboxes[keep_inds]

        # randomly flip image and bboxes
        if self.split == 'train' and np.random.uniform() > 0.5:
            img_rgb[:] = img_rgb[:, ::-1, :]
            img_ir[:] = img_ir[:, ::-1, :]
            bboxes[:, [0, 2]] = img_rgb.shape[1] - bboxes[:, [2, 0]] - 1

        img_rgb = img_rgb.astype(np.float32) / 255.
        img_ir = img_ir.astype(np.float32) / 255.

        img_rgb -= self.mean[0, 0, :3]
        img_rgb /= self.std[0, 0, :3]
        img_ir -= self.mean[0, 0, 3]
        img_ir /= self.std[0, 0, 3]
        img_rgb = img_rgb.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        img_ir = img_ir.transpose((2, 0, 1))

        hmap_tl = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)
        hmap_br = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)

        regs_tl = np.zeros((self.max_objs, 2), dtype=np.float32)
        regs_br = np.zeros((self.max_objs, 2), dtype=np.float32)

        inds_tl = np.zeros((self.max_objs,), dtype=np.int64)
        inds_br = np.zeros((self.max_objs,), dtype=np.int64)

        num_objs = np.array(min(bboxes.shape[0], self.max_objs))
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        ind_masks[:num_objs] = 1

        for i, (xtl, ytl, xbr, ybr) in enumerate(bboxes):
            fxtl = (xtl * self.fmap_size['w'] / self.img_size['w'])
            fytl = (ytl * self.fmap_size['h'] / self.img_size['h'])
            fxbr = (xbr * self.fmap_size['w'] / self.img_size['w'])
            fybr = (ybr * self.fmap_size['h'] / self.img_size['h'])

            ixtl = int(fxtl)
            iytl = int(fytl)
            ixbr = int(fxbr)
            iybr = int(fybr)

            if self.gaussian:
                width = xbr - xtl
                height = ybr - ytl

                width = math.ceil(width * self.fmap_size['w'] / self.img_size['w'])
                height = math.ceil(height * self.fmap_size['h'] / self.img_size['h'])

                radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))

                draw_gaussian(hmap_tl[0], [ixtl, iytl], radius)
                draw_gaussian(hmap_br[0], [ixbr, iybr], radius)
            else:
                hmap_tl[0, iytl, ixtl] = 1
                hmap_br[0, iybr, ixbr] = 1

            regs_tl[i, :] = [fxtl - ixtl, fytl - iytl]
            regs_br[i, :] = [fxbr - ixbr, fybr - iybr]
            inds_tl[i] = iytl * self.fmap_size['w'] + ixtl
            inds_br[i] = iybr * self.fmap_size['w'] + ixbr

        return {'img_rgb': img_rgb, 'img_ir': img_ir,
                'hmap_tl': hmap_tl, 'hmap_br': hmap_br,
                'regs_tl': regs_tl, 'regs_br': regs_br,
                'inds_tl': inds_tl, 'inds_br': inds_br,
                'ind_masks': ind_masks}


class KAIST_eval(KAIST):
    def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=True):
        super(KAIST_eval, self).__init__(data_dir, split)
        self.test_flip = test_flip
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, index):
        img_id = self.img_paths[index]
        img_set, img_vid, img_name = img_id.split("_", 2)
        img_name = img_name.replace("txt", "jpg")
        img_path = os.path.join(self.img_dir, img_set, img_vid)
        img_rgb = cv2.imread(os.path.join(img_path, "visible", img_name), cv2.IMREAD_COLOR)
        img_ir = cv2.imread(os.path.join(img_path, "lwir", img_name), cv2.IMREAD_GRAYSCALE)
        height, width = img_rgb.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            in_height = new_height | 127
            in_width = new_width | 127

            fmap_height, fmap_width = (in_height + 1) // self.down_ratio, (in_width + 1) // self.down_ratio
            height_ratio = fmap_height / in_height
            width_ratio = fmap_width / in_width

            resized_img_rgb = cv2.resize(img_rgb, (new_width, new_height))
            resized_img_rgb, border, offset = crop_image(image=resized_img_rgb,
                                                         center=[new_height // 2, new_width // 2],
                                                         channel = 3,
                                                         new_size=[in_height, in_width])

            resized_img_rgb = resized_img_rgb / 255.
            resized_img_rgb -= self.mean[0, 0, :3]
            resized_img_rgb /= self.std[0, 0, :3]
            resized_img_rgb = resized_img_rgb.transpose((2, 0, 1))[None, :, :, :]  # [H, W, C] to [C, H, W]

            resized_img_ir = cv2.resize(img_ir, (new_width, new_height))
            resized_img_ir = np.expand_dims(resized_img_ir, axis=2)
            resized_img_ir, border, offset = crop_image(image=resized_img_ir,
                                                        center=[new_height // 2, new_width // 2],
                                                        channel = 1,
                                                        new_size=[in_height, in_width])
            resized_img_ir = resized_img_ir / 255.
            resized_img_ir -= self.mean[0, 0, 3]
            resized_img_ir /= self.std[0, 0, 3]
            resized_img_ir = resized_img_ir.transpose((2, 0, 1))[None, :, :, :]  # [H, W, C] to [C, H, W]
            
            if self.test_flip:
                resized_img_rgb = np.concatenate((resized_img_rgb, resized_img_rgb[..., ::-1].copy()), axis=0)
                resized_img_ir = np.concatenate((resized_img_ir, resized_img_ir[..., ::-1].copy()), axis=0)

            out[scale] = {'img_rgb': resized_img_rgb,
                          'img_ir': resized_img_ir,
                          'border': border,
                          'size': [new_height, new_width],
                          'fmap_size': [fmap_height, fmap_width],
                          'ratio': [height_ratio, width_ratio]}

        return img_id, out

    def convert_eval_format(self, all_bboxes, det_dir):
        for i in range(self.num_samples):
            img_id = self.img_paths[i]
            with open(os.path.join(det_dir, img_id), "w") as file:
                for bbox in all_bboxes[img_id][1]:
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = float(bbox[2])
                    y2 = float(bbox[3])
                    score = float(bbox[4])
                    file.write("person {:.4f} {:.4f} {:.4f} {:.4f} {:.8f}\n".format(x1, y1, x2, y2, score))

    def run_eval(self, results, run_dir):
        det_dir = os.path.join(run_dir, "detections")
        if not os.path.exists(det_dir):
            os.mkdir(det_dir)
        det_dir = os.path.join(det_dir, "det")
        if not os.path.exists(det_dir):
            os.mkdir(det_dir)
        self.convert_eval_format(results, det_dir)
        lamr = -1
        returncode = self._do_matlab_eval(run_dir)
        if returncode == 0:
            mat = scipy.io.loadmat(os.path.join(run_dir, 'results.mat'))
            lamr = mat['imp_mr'][0]
        return lamr

    def _do_matlab_eval(self, run_dir):
        path = os.path.join(os.getcwd(), 'datasets', 'KAISTdevkit-matlab-wrapper')
        dt_dir = os.path.join(run_dir, 'detections', 'det')
        gt_dir = self.annot_path
        save_path = os.path.join(run_dir, 'results')
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; '
        cmd += 'kaist_eval_full(\'{:s}\', \'{:s}\', false, true, \'{:s}\'); quit;"'.format(dt_dir, gt_dir, save_path)
        process = subprocess.run(cmd, shell=True)
        return process.returncode

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
            if k in ["img_rgb", "img_ir"] else np.array(sample[s][k])[None, ...] for k in sample[s]} for s in sample}))
        return out

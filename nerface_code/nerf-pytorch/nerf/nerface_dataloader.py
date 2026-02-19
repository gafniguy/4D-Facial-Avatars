import cv2
import imageio
import torch
from torch.utils import data
import json
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm

class NerfaceDataset(Dataset):
    def __init__(self, mode, cfg, debug=False,N_max=20000):
        self.cfg = cfg
        print("initializing NerfaceDataset with mode %s" % mode)
        self.mode = mode
        self.N_max=N_max
        basedir = cfg.dataset.basedir
        load_bbox = True
        self.debug = debug

        with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
            self.metas = json.load(fp)

        # get size
        frame = self.metas["frames"][0]
        fname = os.path.join(basedir, frame["file_path"] + ".png")
        im = imageio.imread(fname)
        self.H, self.W = im.shape[:2]

        poses = []
        expressions = []

        bboxs = []
        fnames = []
        probs = np.zeros((min(self.N_max,len(self.metas["frames"])),self.H, self.W))
        # Prepare importance sampling maps
        p = 0.9
        probs.fill(1 - p)

        ray_importance_sampling_maps = []

        #print("computing bounding boxes probability maps")

        for i,frame in enumerate(tqdm(self.metas["frames"])):
            if i>=self.N_max:
                break
            #imageio.imread(fname).append(os.path.join(basedir, frame["file_path"] + ".png"))
            #imgs.append(imageio.imread(fname))
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            fnames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))

            if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0, 1.0, 0.0, 1.0]))
                else:
                    if mode == 'train':
                        bbox = np.array(frame["bbox"])
                        bbox[0:2] *= self.H
                        bbox[2:4] *= self.W
                        bbox = np.floor(bbox).astype(int)
                        probs[i,bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
                        probs[i] = (1 / probs[i].sum()) * probs[i]
                        bboxs.append(bbox)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)

        #counts.append(counts[-1] + imgs.shape[0])
        #all_imgs.append(imgs)
        #all_frontal_imgs.append(frontal_imgs)
        #all_poses.append(poses)
        #all_expressions.append(expressions)
        #all_bboxs.append(bboxs)

        #i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

        #imgs = np.concatenate(all_imgs, 0)

        #poses = np.concatenate(all_poses, 0)
        #expressions = np.concatenate(all_expressions, 0)
        #bboxs = np.concatenate(all_bboxs, 0)

        camera_angle_x = float(self.metas["camera_angle_x"])
        focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

        # focals = (meta["focals"])
        #intrinsics = self.metas["intrinsics"] if self.metas["intrinsics"] else None
        if self.metas["intrinsics"]:
            self.intrinsics = np.array(self.metas["intrinsics"])
        else:
            self.intrinsics = np.array([focal, focal, 0.5, 0.5])  # fx fy cx cy
        # if type(focals) is list:
        #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
        # else:
        #     focal = np.array([focal, focal])

        # render_poses = torch.stack(
        #     [
        #         torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
        #         for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        #     ],
        #     0,
        # )

        # In debug mode, return extremely tiny images
        if debug:
            self.H = self.H // 32
            self.W = self.W // 32
            # focal = focal / 32.0
            self.intrinsics[:2] = self.intrinsics[:2] / 32.0
            # imgs = [
            #     torch.from_numpy(
            #         cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            #     )
            #     for i in range(imgs.shape[0])
            # ]
            # imgs = torch.stack(imgs, 0)

            #poses = torch.from_numpy(poses)

            #return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs

        if self.cfg.dataset.half_res:
            # TODO: resize images using INTER_AREA (cv2)
            self.H = self.H // 2
            self.W = self.W // 2
            # focal = focal / 2.0
            self.intrinsics[:2] = self.intrinsics[:2] * 0.5
            # imgs = [
            #     torch.from_numpy(
            #         # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
            #         cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            #     )
            #     for i in range(imgs.shape[0])
            # ]
           # imgs = torch.stack(imgs, 0)


        #else:
            # imgs = [
            #     torch.from_numpy(imgs[i]
            #                      # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
            #                      # cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            #                      )
            #     for i in range(imgs.shape[0])
            # ]
            # imgs = torch.stack(imgs, 0)

        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        #bboxs[:, 0:2] *= self.H
        #bboxs[:, 2:4] *= self.W
        #bboxs = np.floor(bboxs)
        bboxs = torch.from_numpy(bboxs).int()
        print("Done with data loading")

        self.bboxs = bboxs
        self.poses = poses
        self.expressions = expressions
        self.fnames = fnames
        self.probs = probs

    def __getitem__(self, idx):
        #bbox = self.bboxs[idx]
        pose = self.poses[idx]
        expression = self.expressions[idx]
        fname = self.fnames[idx]
        img = imageio.imread(fname)
        img = (np.array(img) / 255.0).astype(np.float32)
        img = torch.from_numpy(cv2.resize(img, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA))
        if self.cfg.nerf.train.white_background:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])

        # return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs
        return img, pose, [self.H, self.W, self.intrinsics], expression, self.probs[idx].reshape(-1)

    def __len__(self):
        return min(self.N_max,self.poses.shape[0])
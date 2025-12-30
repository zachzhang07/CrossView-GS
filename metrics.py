#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import numpy as np

import subprocess
import glob

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir, suffix=None):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(gt_dir):
        if suffix is not None and suffix not in fname:
            continue
        gt = Image.open(gt_dir / fname)
        render_path = glob.glob(str(renders_dir / (fname[:-4] + "*")))[0]
        render = Image.open(render_path)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

    return renders, gts, image_names


def evaluate(model_paths, suffix=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    if suffix is not None:
        print(f"Evaluating {suffix}...")
    else:
        print(f"Evaluating all...")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dirs = glob.glob(os.path.join(scene_dir, "test*"))
        for test_dir in test_dirs:
            for method in os.listdir(test_dir):
                print("Method:", method)
                print('test_dir: ', test_dir)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = Path(test_dir) / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                json_path = method_dir / "per_view_count.json"
                renders, gts, image_names = readImages(renders_dir, gt_dir, suffix)

                # json_file = open(json_path)
                # gs_data = json.load(json_file)
                # json_file.close()
                ssims = []
                psnrs = []
                lpipss = []
                gss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # image_name = "{:05d}.png".format(idx)
                    # gss.append(gs_data[image_names[idx]])
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                # print("  GS_NUMS: {:>12.7f}".format(torch.tensor(gss).float().mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "GS_NUMS": torch.tensor(gss).float().mean().item()
                })
                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    # "GS_NUMS": {name: gs for gs, name in zip(torch.tensor(gss).tolist(), image_names)}
                })

                print("{:>12.7f}".format(torch.tensor(psnrs).mean(), ".5") + " " +
                      "{:>12.7f}".format(torch.tensor(ssims).mean(), ".5") + " " +
                      "{:>12.7f}".format(torch.tensor(lpipss).mean(), ".5") + " ")
                print("")

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--suffix', nargs='+', default=['aerial', 'street'])

    args = parser.parse_args()
    print(args)
    for suffix in args.suffix:
        evaluate(args.model_paths, suffix)
    evaluate(args.model_paths)
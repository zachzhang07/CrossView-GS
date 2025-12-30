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
import os
import sys
import yaml
from os import makedirs
import torch
import numpy as np

import subprocess

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state, parse_cfg, get_render_func
from argparse import ArgumentParser
from utils.image_utils import psnr


def render_set(base_model, model_path, name, iteration, views, gaussians, pipe, background, render_mode, ape_code,
               suffix='', rename=True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    os.makedirs(error_path, exist_ok=True)

    t_list = []
    per_view_dict = {}
    per_view_level_dict = {}
    modules = __import__('gaussian_renderer')
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize()
        t0 = time.time()

        if ape_code == -1:
            render_pkg = getattr(modules, get_render_func(base_model))(view, gaussians, pipe, background, iteration,
                                                                       render_mode)
        else:
            render_pkg = getattr(modules, get_render_func(base_model))(view, gaussians, pipe, background, iteration,
                                                                       render_mode, ape_code)

        torch.cuda.synchronize()
        t1 = time.time()
        t_list.append(t1 - t0)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        per_view_dict['{0:05d}'.format(idx) + suffix + ".png"] = visible_count.item()
        
        gt = view.original_image[0:3, :, :]
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + suffix + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + suffix + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + suffix + ".png"))
        
        if rename:
            psnr_value = psnr(rendering[None, ::].contiguous(), gt[None, ::].contiguous()).item()
            os.rename(os.path.join(render_path, '{0:05d}'.format(idx) + suffix + ".png"),
                      os.path.join(render_path, '{0:05d}{1}_{2:.2f}.png'.format(idx, suffix, psnr_value)))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: {fps:.5f}')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), f"per_view_count_{suffix}.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)


def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, ape_code, new_testset):
    with torch.no_grad():
        modules = __import__('scene.gs_model_' + dataset.base_model, fromlist=[''])
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                      resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.set_coarse_interval(opt)
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            render_set(dataset.base_model, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipe, scene.background, dataset.render_mode, ape_code)

        if not skip_test:
            render_set(dataset.base_model, dataset.model_path,
                       "test" if new_testset == '-1' else new_testset.split('/')[-1], scene.loaded_iter,
                       scene.getTestCameras(data_type='aerial'),
                       gaussians, pipe, scene.background, dataset.render_mode, ape_code, suffix="_aerial")
            render_set(dataset.base_model, dataset.model_path,
                       "test" if new_testset == '-1' else new_testset.split('/')[-1], scene.loaded_iter,
                       scene.getTestCameras(data_type='street'),
                       gaussians, pipe, scene.background, dataset.render_mode, ape_code, suffix="_street")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default='-1', type=str)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--new_testset", default='-1', type=str)
    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path
        if args.new_testset != '-1':
            lp.source_path = args.new_testset
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # args.skip_train = True

    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, args.ape, args.new_testset)

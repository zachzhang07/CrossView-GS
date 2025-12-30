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
import numpy as np

from utils.system_utils import searchForMaxIteration

import subprocess

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

os.environ["MKL_NUM_THREADS"] = "12"

os.environ["NUMEXPR_NUM_THREADS"] = "12"

os.environ["OMP_NUM_THREADS"] = "12"

import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from gaussian_renderer import network_gui
from scene import Scene
from utils.general_utils import safe_state, parse_cfg, get_render_func, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import yaml
import warnings
import glob

warnings.filterwarnings('ignore')

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def supply_from_split(aerial_gaussians, street_gaussians, gaussians, supply_method, logger, dataset=None):
    from utils.system_utils import mkdir_p
    from plyfile import PlyData, PlyElement

    def export_pc(anchor, offsets, scales, path, downsample_rate=10):
        mkdir_p(path)

        points = (anchor[:, None, :] + offsets * 1.0 * torch.exp(scales)[:, :3][:, None, :]).reshape((-1, 3))
        points = points[::downsample_rate].cpu().detach().numpy()

        # 将点云数据转换为适合 PLY 文件格式的结构
        vertices = np.array([(point[0], point[1], point[2]) for point in points],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
                                             'vertex')
        # 保存为 PLY 文件
        PlyData([vertex_element]).write(os.path.join(path, "raw_points.ply"))
        print(f"Saved pruned point cloud to {path}")

    def export_ply_and_mlp(model, idx, path, ckpt_path):
        mkdir_p(path)
        anchor = model._anchor[idx].detach().cpu().numpy()
        anchor_feat = model._anchor_feat[idx].detach().cpu().numpy()
        offset = model._offset[idx].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = model._scaling[idx].detach().cpu().numpy()
        rotation = model._rotation[idx].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in model.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offset, anchor_feat, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(path, f"point_cloud.ply"))
        print(f'Saved point cloud to {os.path.join(path, f"point_cloud.ply")}')

        # copy mlps from aerial_ckpt and street_ckpt
        import shutil
        for mlp_name in ['color_mlp', 'cov_mlp', 'opacity_mlp']:
            shutil.copyfile(os.path.join(ckpt_path, "point_cloud",
                                         "iteration_30000", mlp_name + ".pt"), os.path.join(path, mlp_name + ".pt"))

    assert aerial_gaussians.voxel_size == street_gaussians.voxel_size == gaussians.voxel_size, \
        "Voxel size is not the same"
    cur_size = gaussians.voxel_size

    # get the grid coordinates of the alive anchors
    aerial_anchor, aerial_idx = torch.unique(aerial_gaussians.get_anchor, dim=0, return_inverse=True)
    aerial_grid_coords = torch.round(aerial_anchor / cur_size).int()
    fusion_anchor, fusion_idx = torch.unique(gaussians.get_anchor, dim=0, return_inverse=True)
    fusion_grid_coords = torch.round(fusion_anchor / cur_size).int()

    combined = torch.cat([fusion_grid_coords, aerial_grid_coords], dim=0)
    unique_elements, idx, cnt = torch.unique(combined, dim=0, return_inverse=True, return_counts=True)
    aerial_unique_mask = (cnt.index_select(0, idx) == 1)[fusion_grid_coords.shape[0]:]

    logger.info(f"Aerial unique anchor count: {aerial_unique_mask.sum()}, "
                f"common anchor count: {(aerial_unique_mask == False).sum()}")

    supply_factor_define = 1

    if supply_factor_define==0 and aerial_unique_mask.sum() / (aerial_unique_mask == False).sum() > 10:
        supply_factor = 4
    else:
        supply_factor = supply_factor_define

    if supply_method and supply_method in ['aerial', 'both']:
        if 'visibility' in supply_method:
            aerial_visible_mask = aerial_gaussians.get_visibility_mask(dataset.aerial_visibility_path)
            aerial_unique_mask = aerial_unique_mask & aerial_visible_mask
            logger.info(f"Add {aerial_unique_mask.sum() // supply_factor} anchors from aerial model to fusion model "
                        f"with factor {supply_factor} and visibility!")

        else:
            logger.info(f"Add {aerial_unique_mask.sum() // supply_factor} anchors from aerial model to fusion model "
                        f"with factor {supply_factor}")
        gaussians.add_anchor_mask(aerial_gaussians, aerial_unique_mask, supply_factor)
    logger.info(f"Fusion model's anchors {gaussians.get_anchor.shape[0]} now")

    street_anchor, street_idx = torch.unique(street_gaussians.get_anchor, dim=0, return_inverse=True)
    street_grid_coords = torch.round(street_anchor / cur_size).int()
    fusion_anchor, fusion_idx = torch.unique(gaussians.get_anchor, dim=0, return_inverse=True)
    fusion_grid_coords = torch.round(fusion_anchor / cur_size).int()

    combined = torch.cat([fusion_grid_coords, street_grid_coords], dim=0)
    unique_elements, idx, cnt = torch.unique(combined, dim=0, return_inverse=True, return_counts=True)
    street_unique_mask = (cnt.index_select(0, idx) == 1)[fusion_grid_coords.shape[0]:]

    logger.info(f"Street unique anchor count: {street_unique_mask.sum()}, "
                f"common anchor count: {(street_unique_mask == False).sum()}")

    if supply_factor_define==0 and street_unique_mask.sum() / (street_unique_mask == False).sum() > 10:
        supply_factor = 4
    else:
        supply_factor = supply_factor_define

    if supply_method and supply_method in ['street', 'both']:
        if 'visibility' in supply_method:
            street_visible_mask = get_visibility_mask(dataset)
            street_unique_mask = street_unique_mask & street_visible_mask
            logger.info(f"Add {street_unique_mask.sum() // supply_factor} anchors from street model to fusion model "
                        f"with factor {supply_factor} and visibility!")

        else:
            logger.info(f"Add {street_unique_mask.sum() // supply_factor} anchors from street model to fusion model "
                        f"with factor {supply_factor}")
        gaussians.add_anchor_mask(street_gaussians, street_unique_mask, supply_factor)
    logger.info(f"Fusion model's anchors {gaussians.get_anchor.shape[0]} now")

    return gaussians


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    modules = __import__('scene.gs_model_' + dataset.base_model, fromlist=[''])
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger,
                  resolution_scales=dataset.resolution_scales)
    gaussians.set_coarse_interval(opt)

    aerial_gaussians, street_gaussians = None, None

    if (hasattr(dataset, 'supply_voxels') and dataset.supply_voxels) or (
            hasattr(dataset, 'fuse_voxels') and dataset.fuse_voxels):
        aerial_gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        aerial_iter = searchForMaxIteration(os.path.join(dataset.aerial_checkpoint, "point_cloud"))
        aerial_gaussians.load_ply(os.path.join(dataset.aerial_checkpoint, "point_cloud",
                                               "iteration_" + str(aerial_iter), "point_cloud.ply"))
        aerial_gaussians.load_mlp_checkpoints(os.path.join(dataset.aerial_checkpoint, "point_cloud",
                                                           "iteration_" + str(aerial_iter)))
        aerial_gaussians.eval()
        logger.info(f"Aerial Gaussian Loaded with {aerial_gaussians.get_anchor.shape[0]} anchors "
                    f"from {os.path.join(dataset.aerial_checkpoint, 'point_cloud', 'iteration_' + str(aerial_iter))}")

        street_gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        street_iter = searchForMaxIteration(os.path.join(dataset.street_checkpoint, "point_cloud"))
        street_gaussians.load_ply(os.path.join(dataset.street_checkpoint, "point_cloud",
                                               "iteration_" + str(street_iter), "point_cloud.ply"))
        street_gaussians.load_mlp_checkpoints(os.path.join(dataset.street_checkpoint, "point_cloud",
                                                           "iteration_" + str(street_iter)))
        street_gaussians.eval()
        logger.info(f"Street Gaussian Loaded with {street_gaussians.get_anchor.shape[0]} anchors "
                    f"from {os.path.join(dataset.street_checkpoint, 'point_cloud', 'iteration_' + str(street_iter))}")

        supply_voxels = dataset.supply_voxels if hasattr(dataset, 'supply_voxels') and dataset.supply_voxels else None

        if supply_voxels:
            gaussians = supply_from_split(aerial_gaussians, street_gaussians, gaussians, supply_voxels, logger,
                                          dataset)

            point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(0))
            logger.info(
                f"Saving point cloud to {point_cloud_path} with {gaussians.get_anchor.shape[0]} anchors and reload it")
            gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), 0)
            gaussians.load_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = 0
    if hasattr(opt, 'depth_l1_weight_init') and opt.depth_l1_weight_init > 0:
        depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final,
                                            max_steps=opt.iterations)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Ll1fuse_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules = __import__('gaussian_renderer')

    if hasattr(pipe, 'cross_view') and pipe.cross_mv > 0:
        raise NotImplementedError

    len_dict = None

    if hasattr(opt, 'densify_split') and opt.densify_split:
        logger.info(f"Densifying method: {opt.densify_split}")

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        total_loss = 0
        camera_t = []
        imgs = []
        cams = []

        if hasattr(pipe, 'cross_view') and pipe.cross_mv > 0:
            raise NotImplementedError
        else:
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_pkg = getattr(modules, get_render_func(dataset.base_model))(viewpoint_cam, gaussians, pipe,
                                                                               scene.background, iteration,
                                                                               dataset.render_mode)
            image, scaling = render_pkg["render"], render_pkg["scaling"]

            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

            # Depth regularization
            Ll1depth_pure = 0.0
            if hasattr(opt, 'depth_l1_weight_init') and opt.depth_l1_weight_init > 0 and depth_l1_weight(
                    iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["render_depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            Ll1fuse_pure, Ll1fuse = 0.0, 0.0
            if hasattr(dataset, 'fuse_voxels') and dataset.fuse_voxels and opt.lambda_fuse > 0:
                if viewpoint_cam.data_type in ["aerial", 'v1']:
                    reg_model = aerial_gaussians
                elif viewpoint_cam.data_type in ["street", 'v2']:
                    reg_model = street_gaussians
                else:
                    raise NotImplementedError
                # reg_model = aerial_gaussians if viewpoint_cam.data_type == "aerial" else street_gaussians
                reg_render_pkg = getattr(modules, get_render_func(dataset.base_model))(viewpoint_cam, reg_model, pipe,
                                                                                       scene.background, iteration,
                                                                                       dataset.render_mode)
                reg_image = reg_render_pkg["render"]
                if dataset.fuse_method == 'img':
                    Ll1fuse_pure = max(torch.tensor(0.0, device="cuda"),
                                       l1_loss(image, gt_image) - l1_loss(reg_image,
                                                                          gt_image).detach() + opt.alpha_fuse)
                    if len_dict:
                        Ll1fuse = opt.lambda_fuse * Ll1fuse_pure * len_dict[viewpoint_cam.data_type] / len_dict[
                            'fusion']
                    else:
                        Ll1fuse = opt.lambda_fuse * Ll1fuse_pure
                    loss += Ll1fuse_pure
                    Ll1fuse = Ll1fuse.item()
                else:
                    raise NotImplementedError(f"Fusion method {dataset.fuse_method} not implemented")

            if opt.lambda_dreg > 0:
                if scaling.shape[0] > 0:
                    scaling_reg = scaling.prod(dim=1).mean()
                else:
                    scaling_reg = torch.tensor(0.0, device="cuda")
                loss += opt.lambda_dreg * scaling_reg

            if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
                # normal consistency loss
                normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
                normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].squeeze(
                    0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                loss += opt.lambda_normal * normal_error.mean()

            if opt.lambda_dist and iteration > opt.dist_start_iter:
                loss += opt.lambda_dist * render_pkg["render_distort"].mean()

            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_Ll1fuse_for_log = 0.4 * Ll1fuse + 0.6 * ema_Ll1fuse_for_log

            if iteration % 10 == 0:
                if hasattr(opt, 'depth_l1_weight_init') and opt.depth_l1_weight_init > 0:
                    progress_bar.set_postfix(
                        {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                elif hasattr(dataset, 'fuse_voxels') and dataset.fuse_voxels:
                    progress_bar.set_postfix(
                        {"Loss": f"{ema_loss_for_log:.{7}f}", "Fusion Loss": f"{ema_Ll1fuse_for_log:.{7}f}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, getattr(modules, get_render_func(dataset.base_model)),
                            (pipe, scene.background, iteration, dataset.render_mode), wandb, logger)
            if iteration in saving_iterations:
                logger.info(f"\n[ITER {iteration}] Saving Gaussians with {gaussians.get_anchor.shape[0]} anchors...")
                scene.save(iteration)

                # scene.save_statis(iteration)

            # densification
            if opt.update_until > iteration > opt.start_stat:
                # add statis
                # print(f"Adding statis for {viewpoint_cam.data_type}...")
                gaussians.training_statis(render_pkg, image.shape[2], image.shape[1], viewpoint_cam.data_type,
                                          opt.densify_split if hasattr(opt, 'densify_split') else False)

                # densification
                if opt.densification and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.run_densify(iteration, opt, lp)

            elif iteration == opt.update_until:
                gaussians.clean()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration in checkpoint_iterations:
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene,
                    renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, 'train_total_loss': loss, })

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(
                            f'{dataset_name}/' + config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None], global_step=iteration)
                        tb_writer.add_images(
                            f'{dataset_name}/' + config['name'] + "_view_{}/errormap".format(viewpoint.image_name),
                            (gt_image[None] - image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None] - image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/' + config['name'] + "_view_{}/ground_truth".format(
                                viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - l1_loss', l1_test,
                                         iteration)
                    tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - psnr', psnr_test,
                                         iteration)
                if wandb is not None:
                    wandb.log(
                        {f"{config['name']}_loss_viewpoint_l1_loss": l1_test, f"{config['name']}_PSNR": psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/' + 'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()


def render_set(base_model, model_path, name, iteration, views, gaussians, pipe, background, render_mode,
               suffix='', rename=True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    per_view_dict = {}
    modules = __import__('gaussian_renderer')
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize();
        t_start = time.time()

        render_pkg = getattr(modules, get_render_func(base_model))(view, gaussians, pipe, background, iteration,
                                                                   render_mode)
        torch.cuda.synchronize();
        t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + suffix + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + suffix + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + suffix + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

        if rename:
            psnr_value = psnr(rendering[None, ::].contiguous(), gt[None, ::].contiguous()).item()
            os.rename(os.path.join(render_path, '{0:05d}'.format(idx) + suffix + ".png"),
                      os.path.join(render_path, '{0:05d}{1}_{2:.2f}.png'.format(idx, suffix, psnr_value)))

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list


def render_sets(dataset, opt, pipe, iteration, skip_train=False, skip_test=False, wandb=None, tb_writer=None,
                dataset_name=None, logger=None):
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
            t_train_list, visible_count = render_set(dataset.base_model, dataset.model_path, "train", scene.loaded_iter,
                                                     scene.getTrainCameras(), gaussians, pipe, scene.background,
                                                     dataset.render_mode)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: {train_fps.item():.5f}')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/train_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"train_fps": train_fps.item(), })

        if not skip_test:
            if hasattr(lp, 'data_type_list'):
                t_test_list, visible_count_list = [], []
                for s in lp.data_type_list:
                    t_test, visible_count = render_set(dataset.base_model, dataset.model_path, "test", scene.loaded_iter,
                                                       scene.getTestCameras(data_type=s),
                                                       gaussians, pipe, scene.background, dataset.render_mode,
                                                       suffix='_' + s)
                    t_test_list = t_test_list + t_test
                    visible_count_list = visible_count_list + visible_count
                test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            else:
                t_test_list1, visible_count1 = render_set(dataset.base_model, dataset.model_path, "test", scene.loaded_iter,
                                                          scene.getTestCameras(data_type='aerial'),
                                                          gaussians, pipe, scene.background,
                                                          dataset.render_mode, suffix='_aerial')
                t_test_list2, visible_count2 = render_set(dataset.base_model, dataset.model_path, "test", scene.loaded_iter,
                                                          scene.getTestCameras(data_type='street'),
                                                          gaussians, pipe, scene.background,
                                                          dataset.render_mode, suffix='_street')
                t_test_list = t_test_list1 + t_test_list2
                visible_count = visible_count1 + visible_count2
                test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: {test_fps.item():.5f}')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps": test_fps, })

    return visible_count


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


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None,
             logger=None, suffix=None):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    print("")
    if suffix is not None:
        print(f"Evaluating {suffix}...")
    else:
        print(f"Evaluating all...")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir, suffix)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        logger.info(f"model_paths: {model_paths}")
        logger.info("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  GS_NUMS: {:>12.7f}".format(torch.tensor(visible_count).float().mean(), ".5"))
        logger.info("{:>12.7f}".format(torch.tensor(psnrs).mean(), ".5") + " " +
                    "{:>12.7f}".format(torch.tensor(ssims).mean(), ".5") + " " +
                    "{:>12.7f}".format(torch.tensor(lpipss).mean(), ".5") + " ")
        print("")

        if wandb is not None:
            wandb.log({"test_PSNR": torch.stack(psnrs).mean().item(), })
            wandb.log({"test_SSIM": torch.stack(ssims).mean().item(), })
            wandb.log({"test_LPIPS": torch.stack(lpipss).mean().item(), })
            wandb.log({"test_GS_NUMS": torch.stack(visible_count).float().mean().item(), })

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/GS_NUMS', torch.tensor(visible_count).float().mean().item(), 0)

        full_dict[scene_dir][method].update({
            "PSNR": torch.tensor(psnrs).mean().item(),
            "SSIM": torch.tensor(ssims).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
            "GS_NUMS": torch.tensor(visible_count).float().mean().item(),
        })

        per_view_dict[scene_dir][method].update({
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
        })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default='-1')
    parser.add_argument('--suffix', default=['aerial', 'street'])
    parser.add_argument("--no_ts", action="store_true", help="no timestamp in output path")

    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        args.save_iterations.append(op.iterations)

    # enable logging
    if not args.no_ts:
        cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        lp.model_path = os.path.join("outputs", lp.dataset_name, lp.data_type, lp.scene_name, cur_time)
    else:
        lp.model_path = os.path.join("outputs", lp.dataset_name, lp.data_type, lp.scene_name)
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [op.iterations]

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    saveRuntimeCode(os.path.join(lp.model_path, 'backup'))

    logger.info(f'args: {args}')
    logger.info(f'lp: {lp}')
    logger.info(f'op: {op}')
    logger.info(f'pp: {pp}')

    exp_name = lp.scene_name if lp.dataset_name == "" else lp.dataset_name + "_" + lp.data_type + "_" + lp.scene_name
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Octree-GS",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + lp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                 args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(op.model_path, f'point_cloud/iteration_{op.iterations}', 'point_cloud.ply')
        training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                 args.start_checkpoint, args.debug_from, wandb, logger, new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    if hasattr(lp, 'data_type_list'):
        suffix = lp.data_type_list
    else:
        suffix = ['aerial', 'street']
    for s in suffix:
        evaluate(lp.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger, suffix=s)
    evaluate(lp.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
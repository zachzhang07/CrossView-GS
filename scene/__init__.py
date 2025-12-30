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
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
import numpy as np


class Scene:

    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None,
                 logger=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales
        self.args = args

        if args.random_background:
            self.background = torch.rand(3, dtype=torch.float32, device="cuda")
        elif args.white_background:
            self.background = torch.ones(3, dtype=torch.float32, device="cuda")
        else:
            self.background = torch.zeros(3, dtype=torch.float32, device="cuda")

        if load_iteration and isinstance(load_iteration, str) and load_iteration.isdigit() and load_iteration != '0':
            load_iteration = int(load_iteration)

        if load_iteration:
            if load_iteration == '-1' or -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            elif load_iteration == '0':
                self.loaded_iter = 0
            else:
                self.loaded_iter = load_iteration

            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if 'uc_gs' in args.source_path:
                print("Found uc_gs in source path, assuming UCGS data set!")
                depth_dir = args.depth_dir if hasattr(args, 'depth_dir') else ""
                scene_info = sceneLoadTypeCallbacks["UCGS"](args.source_path, args.images, args.eval,
                                                            depth_dir=depth_dir,
                                                            warmup_ply_path=ply_path)
            elif 'road' in args.source_path:
                print("Found road in source path, assuming Road data set!")
                ply_path = os.path.join(args.source_path, "sparse/0/fused_point_cloud.ply")
                ply_path = None if not os.path.exists(ply_path) else ply_path
                scene_info = sceneLoadTypeCallbacks["Road"](args.source_path, args.images, args.eval, llffhold=32,
                                                            warmup_ply_path=ply_path)
            elif 'tower' in args.source_path:
                print("Found tower in source path, assuming Tower data set!")
                scene_info = sceneLoadTypeCallbacks["Tower"](args.source_path, args.images, args.eval,
                                                             warmup_ply_path=ply_path)
            elif 'temple' in args.source_path:
                print("Found temple in source path, assuming Temple data set!")
                scene_info = sceneLoadTypeCallbacks["Temple"](args.source_path, args.images, args.eval,
                                                              llffhold=args.llff_hold if hasattr(args, 'llff_hold') else 4,
                                                              warmup_ply_path=ply_path)
            elif 'monument' in args.source_path:
                print("Found monument in source path, assuming Monument data set!")
                scene_info = sceneLoadTypeCallbacks["Monument"](args.source_path, args.images, args.eval, llffhold=4,
                                                              warmup_ply_path=ply_path)
            elif 'Zeche' in args.source_path:
                print("Found Zeche in source path, assuming Zeche data set!")
                scene_info = sceneLoadTypeCallbacks["Zeche"](args.source_path, args.images, args.eval,
                                                             warmup_ply_path=ply_path)
            elif 'Dortmund' in args.source_path:
                print("Found Dortmund in source path, assuming Dortmund data set!")
                scene_info = sceneLoadTypeCallbacks["Dortmund"](args.source_path, args.images, args.eval,
                                                              warmup_ply_path=ply_path)
            elif 'SWJTU' in args.source_path:
                print("Found SWJTU in source path, assuming SWJTU data set!")
                scene_info = sceneLoadTypeCallbacks["SWJTU"](args.source_path, args.images, args.eval,
                                                             warmup_ply_path=ply_path)
            elif 'wild' in args.source_path:
                print("Found wild in source path, assuming Wild data set!")
                scene_info = sceneLoadTypeCallbacks["Wild"](args.source_path, args.images, args.eval,
                                                              llffhold=args.llff_hold if hasattr(args, 'llff_hold') else 4,
                                                              warmup_ply_path=ply_path)
            else:
                print("Found transforms_train.json file, assuming Colmap data set!")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                              warmup_ply_path=ply_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if 'MatrixCity' in args.source_path:
                print("Found transforms_train.json file in MatrixCity path, assuming MatrixCity data set!")
                scene_info = sceneLoadTypeCallbacks["matrixcity"](args.source_path, args.eval, warmup_ply_path=ply_path)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.eval, warmup_ply_path=ply_path)
        else:
            assert False, f"Could not recognize scene type! args.source_path: {args.source_path}"

        self.gaussians.set_appearance(len(scene_info.train_cameras))

        if (not self.loaded_iter and not (hasattr(args, 'supply_voxels') and args.supply_voxels)
                and not (hasattr(args, 'fuse_voxels') and 'fromInit' not in args.fuse_voxels)):
            pcd = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        assert len(self.resolution_scales) == 1 and self.resolution_scales[0] == 1.0, "Multi-res not supported yet!"
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args, self.background)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args, self.background)
            # print(f'Test cams len: {len(self.test_cameras[resolution_scale])}')
            # for cam in self.test_cameras[resolution_scale]:
            #     print(f'Test cam name: {cam.image_name}, type: {cam.data_type}')
            # 1 / 0

        if args.data_type != "" :
            if hasattr(args, 'data_type_list'):
                self.train_data_type_idx, self.test_data_type_idx = {}, {}
                for i in args.data_type_list:
                    self.train_data_type_idx[i] = []
                    self.test_data_type_idx[i] = []
                for i, cam in enumerate(self.train_cameras[1.0]):
                    if cam.data_type in self.train_data_type_idx.keys():
                        self.train_data_type_idx[cam.data_type].append(i)
                for i, cam in enumerate(self.test_cameras[1.0]):
                    if cam.data_type in self.test_data_type_idx.keys():
                        self.test_data_type_idx[cam.data_type].append(i)
                print(f"Using data types: {args.data_type} in {args.data_type_list}")
                str_train = "Train cameras: "
                for i in self.train_data_type_idx.keys():
                    str_train += f"{i}: {len(self.train_data_type_idx[i])}, "
                print(str_train)
                str_test = "Test cameras: "
                for i in self.test_data_type_idx.keys():
                    str_test += f"{i}: {len(self.test_data_type_idx[i])}, "
                print(str_test)
            else:
                self.train_data_type_idx = {'aerial': [], 'street': []}
                self.test_data_type_idx = {'aerial': [], 'street': []}
                for i, cam in enumerate(self.train_cameras[1.0]):
                    if cam.data_type in self.train_data_type_idx.keys():
                        self.train_data_type_idx[cam.data_type].append(i)
                for i, cam in enumerate(self.test_cameras[1.0]):
                    if cam.data_type in self.test_data_type_idx.keys():
                        self.test_data_type_idx[cam.data_type].append(i)
                print(f"Using data types: {args.data_type}")
                print(f"Train cameras: aerial: {len(self.train_data_type_idx['aerial'])}, "
                      f"street: {len(self.train_data_type_idx['street'])}")
                print(f"Test cameras: aerial: {len(self.test_data_type_idx['aerial'])}, "
                      f"street: {len(self.test_data_type_idx['street'])}")

            # print(f'scene init aerial')
            # for idx in self.test_data_type_idx['aerial']:
            #     print(f'test camera: {idx}, {self.test_cameras[1.0][idx].image_name}, {self.test_cameras[1.0][idx].data_type}')
            # print(f'scene init street')
            # for idx in self.test_data_type_idx['street']:
            #     print(f'test camera: {idx}, {self.test_cameras[1.0][idx].image_name}, {self.test_cameras[1.0][idx].data_type}')
            # 1 / 0

        if self.loaded_iter is not None:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                             "point_cloud",
                                                             "iteration_" + str(self.loaded_iter)))
            print(f"Gaussian Loaded with {self.gaussians.get_anchor.shape[0]} anchors "
                  f"from {os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter))}")
        elif (hasattr(args, 'supply_voxels') and args.supply_voxels) or (
                hasattr(args, 'fuse_voxels') and 'fromInit' not in args.fuse_voxels):
            fusion_iter = searchForMaxIteration(os.path.join(args.fusion_checkpoint, "point_cloud"))
            self.gaussians.load_ply(os.path.join(args.fusion_checkpoint,
                                                 "point_cloud", "iteration_" + str(fusion_iter),
                                                 "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(args.fusion_checkpoint,
                                                             "point_cloud", "iteration_" + str(fusion_iter)))
            # print(
            #     f'Firstly loading fusion model in {os.path.join(args.fusion_checkpoint, "point_cloud", "iteration_" + str(fusion_iter))}')

            logger.info(f"Fusion Gaussian Loaded with {self.gaussians.get_anchor.shape[0]} anchors "
                        f"from {os.path.join(args.fusion_checkpoint, 'point_cloud', 'iteration_' + str(fusion_iter))}")
        # elif hasattr(args, 'vis_checkpoint') and args.vis_checkpoint is not None:
        #     self.gaussians.load_ply(os.path.join(args.vis_checkpoint, "point_cloud.ply"))
        #     self.gaussians.load_mlp_checkpoints(args.vis_checkpoint)
        #     logger.info(f"Visualization Loaded with {args.vis_checkpoint} model ")
        else:
            self.gaussians.create_from_pcd(pcd, self.cameras_extent, logger, args)

    def save_ply(self, pcd, ratio, path):
        new_points = pcd.points[::ratio]
        new_colors = pcd.colors[::ratio]
        new_normals = pcd.normals[::ratio]
        new_pcd = BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals)
        storePly(path, new_points, new_colors)
        return new_pcd

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), iteration)
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def save_statis(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_statis(os.path.join(point_cloud_path, "opacities.npy"))

    # def getTrainCamerasLen(self, scale=1.0):
    #     len_dict = {'aerial': len(self.train_data_type_idx['aerial']),
    #                 'street': len(self.train_data_type_idx['street']),
    #                 'fusion': len(self.train_cameras[scale])}
    #     return len_dict

    def getTrainCameras(self, data_type=None, scale=1.0):
        if data_type is None:
            data_type = self.args.data_type
        all_cams = []
        if data_type is None or data_type == "fusion":
            all_cams.extend(self.train_cameras[scale])
        else:
            # print(f"getTrainCameras using data type: {data_type}")
            for idx in self.train_data_type_idx[data_type]:
                all_cams.append(self.train_cameras[scale][idx])
                # print(f'train camera: {idx}, {self.train_cameras[scale][idx].image_name}, {self.train_cameras[scale][idx].data_type}')
            # 1 / 0
        return all_cams

    def getTestCameras(self, data_type=None, scale=1.0):
        if data_type is None:
            data_type = self.args.data_type
        all_cams = []
        if data_type is None or data_type == "fusion":
            all_cams.extend(self.test_cameras[scale])
        else:
            # print(f"getTestCameras using data type: {data_type}")
            for idx in self.test_data_type_idx[data_type]:
                all_cams.append(self.test_cameras[scale][idx])
                # print(f'test camera: {idx}, {self.test_cameras[scale][idx].image_name}, {self.test_cameras[scale][idx].data_type}')
            # 1 / 0
        return all_cams

    def getCameras(self, scale=1.0):
        all_cams = []
        all_cams.extend(self.train_cameras[scale])
        all_cams.extend(self.test_cameras[scale])

        return all_cams

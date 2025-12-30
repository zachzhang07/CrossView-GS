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
import glob
import sys
import cv2
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement

try:
    import laspy
except:
    print("No laspy")
from utils.graphics_utils import BasicPointCloud
import concurrent.futures


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    data_type: str
    depth_params: dict
    depth_path: str


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    # ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    def process_frame(idx, key):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        return CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                          image_path=image_path, image_name=image_name, width=width, height=height)

    ct = 0
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交每个帧到执行器进行处理
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        # 使用as_completed()获取已完成的任务
        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            cam_infos.append(cam_info)

            ct += 1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW + f"{ct}/{len(cam_extrinsics)}" + Style.RESET_ALL})
                progress_bar.update(10)

        progress_bar.close()

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    return cam_infos


def readColmapCamerasForTower(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        idx = int(image_name[-3:])

        data_type = "street" if 'boat' in image_name else "aerial"
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForUCGS(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        idx = int(image_name[-3:])

        data_type = "street" if idx <= len(cam_extrinsics) // 2 else "aerial"
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForRoad(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name.split(".")[0]
        try:
            image = Image.open(image_path)
        except:
            continue

        data_type = "street" if 'TIMELAPSE' in image_name else "aerial"
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForTemple(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder,
                               load_v3=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if image_name[:3] in ['179', '272', '254']:
            data_type = "v1"
        elif image_name[:3] in ['695']:
            data_type = "v2"
        elif load_v3 and image_name[:3] in ['071', '224']:
            print(f'loading {image_name} as v3 data!')
            data_type = "v3"
        else:
            # print(f"Skipping image {image_name} with video suffix {image_name[:3]}!")
            continue

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForMonument(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if image_name[:3] in ['076', '173', '177', '908', '954', '995']:
            data_type = "v1"
        elif image_name[:3] in ['089', '479', '996']:
            data_type = "v2"
        else:
            # print(f"Skipping image {image_name} with video suffix {image_name[:3]}!")
            continue

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForWild(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if image_name[:2] == 'd1':
            data_type = "v1"
        elif image_name[:2] == 'd2':
            data_type = "v2"
        else:
            # print(f"Skipping image {image_name} with video suffix {image_name[:3]}!")
            continue

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForZeche(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        idx = int(image_name[-4:])

        data_type = "street" if idx <= 6000 else "aerial"
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasForSWJTU(cam_extrinsics, cam_intrinsics, images_folder, depths_params, depths_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # idx = int(image_name[-4:])

        data_type = "street" if 'IMG' in image_name else "aerial"
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove].split('/')[-1]]
            except:
                print("\n", key, "not found in depths_params")
        depth_path = os.path.join(depths_folder,
                                  f"{extr.name[:-n_remove].split('/')[-1]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, data_type=data_type,
                              depth_params=depth_params, depth_path=depth_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readCamerasFromTransforms(path, transformsfile, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        def process_frame(idx, frame):
            # Process each frame and append cam_info to cam_infos list
            cam_name = frame["file_path"] + extension
            image_path = os.path.join(path, cam_name)
            if not os.path.exists(image_path):
                raise ValueError(f"Image {image_path} does not exist!")
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if (
                    "k1" in frame
                    and "k2" in frame
                    and "p1" in frame
                    and "p2" in frame
                    and "k3" in frame
            ):
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array(
                    [frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]],
                    dtype=np.float32,
                )
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )

        ct = 0
        progress_bar = tqdm(frames, desc="Loading dataset")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交每个帧到执行器进行处理
            futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]

            # 使用as_completed()获取已完成的任务
            for future in concurrent.futures.as_completed(futures):
                cam_info = future.result()
                cam_infos.append(cam_info)

                ct += 1
                if ct % 10 == 0:
                    progress_bar.set_postfix({"num": Fore.YELLOW + f"{ct}/{len(frames)}" + Style.RESET_ALL})
                    progress_bar.update(10)

            progress_bar.close()

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    return cam_infos


def readColmapSceneInfo(path, images, eval, llffhold=8, warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = images
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir))

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from {ply_path}')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readTowerSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForTower(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                          depths_params=depths_params,
                                          depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    for c in test_cam_infos:
        print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readUCGSSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForUCGS(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                         depths_params=depths_params,
                                         depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name[-3:])

    if eval:
        test_in_ucgs_setting = False
        if test_in_ucgs_setting:
            print("Test in UCGS setting!!")
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if 'train' in c.image_name]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if 'eval' in c.image_name]

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readRoadSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForRoad(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                         depths_params=depths_params,
                                         depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name.split('/')[-1])

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readTempleSceneInfo(path, images, eval, llffhold=4, depth_dir="", load_v3=False, warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForTemple(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                           depths_params=depths_params,
                                           depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "",
                                           load_v3=load_v3)
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    # print('test cameras:')
    # for c in test_cam_infos:
    #     print(f'{c.image_name}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readMonumentSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForMonument(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                             depths_params=depths_params,
                                             depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    print('test cameras:')
    for c in test_cam_infos:
        print(f'{c.image_name}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readWildSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForWild(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                         depths_params=depths_params,
                                         depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    print('test cameras:')
    for c in test_cam_infos:
        print(f'{c.image_name}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readZecheSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForZeche(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                          depths_params=depths_params,
                                          depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name[-4:])

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    for c in test_cam_infos:
        print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readDortmundSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    # index is same as zeche
    cam_infos = readColmapCamerasForZeche(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                          depths_params=depths_params,
                                          depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readSWJTUSceneInfo(path, images, eval, llffhold=8, depth_dir="", warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isn't there AND depths file is here -> throw error
    depths_params = None
    if depth_dir != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
            print(f"Depths params loaded from {depth_params_file}.")

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = images
    cam_infos = readColmapCamerasForSWJTU(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir),
                                          depths_params=depths_params,
                                          depths_folder=os.path.join(path, depth_dir) if depth_dir != "" else "")
    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for c in test_cam_infos:
    #     print(f'test camera: {c.image_name}, data_type: {c.data_type}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readNerfSyntheticInfo(path, eval, extension=".png", warmup_ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if warmup_ply_path is not None:
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_paths = glob.glob(os.path.join(path, "*.ply"))
        if len(ply_paths) == 0:
            ply_path = os.path.join(path, "points3d.ply")
            # Since this data set has no colmap data, we start with random points
            num_pts = 10_000
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            colors = np.random.random((num_pts, 3))
            normals = np.zeros((num_pts, 3))
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

            storePly(ply_path, xyz, colors * 255)
        else:
            ply_path = ply_paths[0]
            pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


def readCamerasFromMatrixCityTransforms(path, transformsfile, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        def process_frame(idx, frame):
            # Process each frame and append cam_info to cam_infos list
            cam_name = frame["file_path"] + extension
            image_path = os.path.join(path, cam_name)

            if 'aerial' in frame['file_path']:
                img_data_type = 'aerial'
            elif 'street' in frame['file_path']:
                img_data_type = 'street'
            else:
                if 'front_back' in frame['file_path']:
                    img_data_type = 'v1'
                elif 'top' in frame['file_path']:
                    img_data_type = 'v2'
                else:
                    # print(f'skipping {frame["file_path"]}')
                    # img_data_type = 'unknown'
                    return None

            if not os.path.exists(image_path):
                raise ValueError(f"Image {image_path} does not exist!")

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            # new fake image
            # print('using fake img for speed up!')
            # image = Image.new(size=(int(frame["w"]), int(frame["h"])), mode="RGB")

            assert image.size[0] == frame["w"] and image.size[1] == frame[
                "h"], f"image size {image.size} does not match frame size {frame['w']}x{frame['h']}"

            if "k1" in frame and "k2" in frame and "p1" in frame and "p2" in frame and "k3" in frame:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array(
                    [frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]],
                    dtype=np.float32,
                )
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
                data_type=img_data_type,
                depth_params=None,
                depth_path="",
            )

        ct = 0
        progress_bar = tqdm(frames, desc="Loading dataset")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交每个帧到执行器进行处理
            futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]

            # 使用as_completed()获取已完成的任务
            for future in concurrent.futures.as_completed(futures):
                cam_info = future.result()
                # if '0001' in cam_info.image_name:
                #     print(f'cam_info: {cam_info}')
                #     1 / 0
                if cam_info is not None:
                    cam_infos.append(cam_info)

                ct += 1
                if ct % 10 == 0:
                    progress_bar.set_postfix({"num": Fore.YELLOW + f"{ct}/{len(frames)}" + Style.RESET_ALL})
                    progress_bar.update(10)

            progress_bar.close()

    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    return cam_infos


def readMatrixCityInfo(path, eval, extension=".png", warmup_ply_path=None):
    # if len(glob.glob(os.path.join(path, f"*.pt"))) > 0:
    #     print(f"Found pre-processed data *_cam_infos.pt in {path}, loading...")
    #     train_cam_infos = torch.load(os.path.join(path, f"train_cam_infos.pt"))
    #     test_cam_infos = torch.load(os.path.join(path, f"test_cam_infos.pt"))
    print(f"Reading Train Transforms from {path}/transforms_train.json")
    train_cam_infos = readCamerasFromMatrixCityTransforms(path, "transforms_train.json", extension)
    # torch.save(train_cam_infos, os.path.join(path, f"train_cam_infos.pt"))
    print(f"Reading Test Transforms from {path}/transforms_test.json")
    test_cam_infos = readCamerasFromMatrixCityTransforms(path, "transforms_test.json", extension)
    # torch.save(test_cam_infos, os.path.join(path, f"test_cam_infos.pt"))

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if warmup_ply_path is not None:
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_paths = glob.glob(os.path.join(path, "*.ply"))
        if len(ply_paths) == 0:
            ply_path = os.path.join(path, "points3d.ply")
            # Since this data set has no colmap data, we start with random points
            num_pts = 10_000
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            colors = np.random.random((num_pts, 3))
            normals = np.zeros((num_pts, 3))
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

            storePly(ply_path, xyz, colors * 255)
        else:
            print(f'Found point cloud file in {glob.glob(os.path.join(path, "*.ply"))[0]} , using it')
            ply_path = ply_paths[0]
            pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "matrixcity": readMatrixCityInfo,
    "UCGS": readUCGSSceneInfo,
    "Zeche": readZecheSceneInfo,
    "SWJTU": readSWJTUSceneInfo,
    "Tower": readTowerSceneInfo,
    "Temple": readTempleSceneInfo,
    "Monument": readMonumentSceneInfo,
    "Wild": readWildSceneInfo,
    "Road": readRoadSceneInfo,
    "Dortmund": readDortmundSceneInfo,
}

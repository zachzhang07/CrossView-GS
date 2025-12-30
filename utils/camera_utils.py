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

import torch
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import concurrent.futures
from tqdm import tqdm
from colorama import Fore, init, Style
import cv2

WARNED = False


def loadCam(args, id, cam_info, resolution_scale, background):
    orig_w, orig_h = cam_info.width, cam_info.height

    if cam_info.depth_path != "":
        try:
            is_nerf_synthetic = False
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2 ** 16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(
                f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    gt_image = PILtoTorch(cam_info.image, resolution, background)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, image_name=cam_info.image_name,
                  invdepthmap=invdepthmap, depth_params=cam_info.depth_params,
                  resolution_scale=resolution_scale,
                  uid=id, data_device=args.data_device, data_type=cam_info.data_type)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, background):
    camera_list = []

    init = torch.inverse(torch.ones((1, 1), device="cuda"))
    ct = 0
    progress_bar = tqdm(cam_infos, desc="Processing image")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(loadCam, args, cam_id, c, resolution_scale, background) for cam_id, c in
                   enumerate(cam_infos)]

        for future in concurrent.futures.as_completed(futures):
            camera = future.result()
            camera_list.append(camera)

            ct += 1
            if ct % 10 == 0 or ct == len(cam_infos):
                progress_bar.set_postfix({"num": Fore.YELLOW + f"{ct}/{len(cam_infos)}" + Style.RESET_ALL})
                progress_bar.update(10)

        progress_bar.close()

    camera_list = sorted(camera_list, key=lambda x: x.image_name)
    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

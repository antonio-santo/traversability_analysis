import sklearn.metrics
import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.path.abspath("/home/arvc/Desktop/Antonio/minkowski/scripts/examples"))
from minkunet import MinkUNet34C
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from plyfile import PlyData


class preprocessingDataset4f(Dataset):
    """
    Rutas de interes:
    - "/media/arvc/Extreme SSD/kitti_mine/valid/[4-10]
    """

    def __init__(self, mode, voxel, root_data="/home/arvc/DATASETS/kitti_mine/valid/9"):
        super(Dataset, self).__init__()
        self.root = root_data
        self.directories = sorted(glob.glob('{}/*'.format(self.root)))

        self.pcds = []
        for i in self.directories:
            self.pcds.append(i)
        self.voxel_size = voxel
        self.mode=mode
        self.optmimal=0

    def __len__(self):
        return len(self.pcds)
        #return 100

    def __getitem__(self, idx):
        #
        # def normalize_features(features):
        #     norm_arr = np.empty_like(features)
        #     for dim in range(features.shape[1]):
        #         minimo = min(features[:, dim])
        #         diff_arr = max(features[:, dim]) - minimo
        #         for n, l in enumerate(features[:, dim]):
        #             norm_arr[n, dim] = ((l - minimo) / diff_arr)
        #     return norm_arr.astype(np.float32)

        def normalize_features(features):
            norm_arr = np.empty_like(features)
            # for dim in range(features.shape[1]):
            minimo = min(features[:])
            diff_arr = max(features[:]) - minimo
            for n, l in enumerate(features[:]):
                norm_arr[n] = ((l - minimo) / diff_arr)
            return norm_arr.astype(np.float32)

        def compute_normals(pcd):
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=40))
            pcd.orient_normals_to_align_with_direction()
            normals = np.asarray(pcd.normals)
            ey = o3d.geometry.PointCloud()
            ey.points = o3d.utility.Vector3dVector(pcd.points)
            ey.normals = o3d.utility.Vector3dVector(normals)
            return ey

        pcd_raw = o3d.io.read_point_cloud(self.pcds[idx])
        points_raw = np.asarray(pcd_raw.points)
        # distances = np.linalg.norm(points_raw, axis=1)
        # umbral_dist = np.where(distances >= 15)
        # new_points = np.delete(points_raw, umbral_dist[0], axis=0)
        # pcd_def = o3d.geometry.PointCloud()
        # pcd_def.points = o3d.utility.Vector3dVector(new_points)
        self.coords_orig= np.asarray(points_raw.points)
        self.coords = self.coords_raw / self.voxel_size
        if self.optmimal==0: #calcula las normales de una voxelizacion y  luego traza hacia atras para darle normal a todos los puntos
            pcd = pcd_raw.voxel_down_sample_and_trace(0.03, pcd_raw.get_min_bound(), pcd_raw.get_max_bound(),approximate_class=True)
            cloud_with = compute_normals(pcd[0])
            final_normals = np.zeros((points_raw.shape))
            for k, t in enumerate(pcd[2]):  # points in voxels
                final_normals[np.array(t)] = np.asarray(cloud_with.normals)[k]

            self.features = np.stack((final_normals[:, 0], final_normals[:, 1], final_normals[:, 2], self.coords_raw[:, 2]), axis=1)
            self.features[:,3]=normalize_features(self.features[:,3])
            plydata = PlyData.read(self.pcds[idx])
            self.labels = np.array((plydata.elements[0].data['labels']))
            # self.labels = np.delete(self.labels, umbral_dist[0], axis=0)
            return self.coords, self.features, self.labels

        else: #calcula normales sobre los puntos sin mas con sus vecinos y radio
            final_normals = np.asarray(compute_normals(pcd_raw).normals)
            self.features = np.stack(
                (final_normals[:, 0], final_normals[:, 1], final_normals[:, 2], self.coords_raw[:, 2]), axis=1)
            self.features[:, 3] = normalize_features(self.features[:, 3])
            plydata = PlyData.read(self.pcds[idx])
            self.labels = np.array((plydata.elements[0].data['labels']))
            # self.labels = np.delete(self.labels, umbral_dist[0], axis=0)
            return self.coords, self.features, self.labels


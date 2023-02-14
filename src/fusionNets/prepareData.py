
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.path.abspath("/home/arvc/Desktop/Antonio/minkowski/scripts/examples"))
from plyfile import PlyData

class preprocessingDataset(Dataset):
    """
    Rutas de interes:
    - "/media/arvc/Extreme SSD/kitti_mine/valid/[4-10]
    """

    def __init__(self, root_data="/home/arvc/DATASETS/kitti_mine/valid/9"):
        super(Dataset, self).__init__()
        self.root = root_data
        self.directories = sorted(glob.glob('{}/*'.format(self.root)))
        self.pcds = []
        for i in self.directories:
            self.pcds.append(i)

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):

        def normalize_features(features):
            norm_arr = np.empty_like(features)
            for dim in range(features.shape[1]):
                minimo = min(features[:, dim])
                diff_arr = max(features[:, dim]) - minimo
                for n, l in enumerate(features[:, dim]):
                    norm_arr[n, dim] = ((l - minimo) / diff_arr)
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
        distances = np.linalg.norm(points_raw, axis=1)
        umbral_dist = np.where(distances >= 15)
        new_points = np.delete(points_raw, umbral_dist[0], axis=0)

        pcd_def = o3d.geometry.PointCloud()
        pcd_def.points = o3d.utility.Vector3dVector(new_points)
        self.coords_raw = new_points
        self.pcd_def = pcd_raw.voxel_down_sample_and_trace(0.2, pcd_def.get_min_bound(), pcd_def.get_max_bound(),approximate_class=True)
        self.coords = np.asarray(self.pcd_def[0].points)
        cloud_with = compute_normals(self.pcd_def[0])
        self.features = np.asarray(cloud_with.normals)
        plydata = PlyData.read(self.pcds[idx])
        self.labels = np.array((plydata.elements[0].data['labels']))
        self.labels = np.delete(self.labels, umbral_dist[0], axis=0)
        self.idx_voxel=[]
        # for k, t in enumerate(self.pcd_def[2]):  #points in voxels
        #     self.idx_voxel.append(np.array(t))
        return self.coords, self.features, self.labels, self.coords_raw
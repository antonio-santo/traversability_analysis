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
from dataloader2f import preprocessingDataset2f
from dataloader4f import preprocessingDataset4f

def visualize_each_cloud(pred, pcds, gradient):
    cloud = o3d.io.read_point_cloud(pcds)
    points = np.asarray(cloud.points)
    distances = np.linalg.norm(points, axis=1)
    umbral_dis_vis = np.where(distances >= 15)
    new_points = np.delete(points, umbral_dis_vis[0], axis=0)
    color = []
    if gradient == 0:
        # SOLUCION CATEGORICA DEL PROBLEMA
        for i in pred:
            if i == 1:
                color.append([95, 158, 160])
            if i == 0:
                color.append([106, 90, 205])
    else:
        # SOLUCION CONTINUA DEL PROBLEMA (GRADIENTE)
        color1 = np.array([178, 102, 128])
        color2 = np.array([0, 128, 0])
        color3 = np.array([255, 255, 0])
        divid = 20
        color_rate = (color3 - color2) / divid
        for i in pred:
            color.append(color1 + ((i * 20) * color_rate))

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(color).astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(new_points)
    o3d.visualization.draw_geometries([pcd])
    return True



if __name__ == '__main__':

    device = torch.device('cpu')
    datasets2evaluate=["/home/arvc/DATASETS/Rellis_3D_custom_def/valid",
                       "/home/arvc/DATASETS/kitti_mine/valid/4"]
    models2evaluate=[
        "Voxel0.05/BestModel0_th_0.4205591072142124voxel_size0.05_0.9269229441412247.pth",
        "Voxel0.1/BestModel0_th_0.477454120144248voxel_size0.1_0.9225966460906108.pth",
        "Voxel0.2/BestModel2_th_0.6042602644860744voxel_size0.2_0.9015045953476892.pth",
        "Voxel0.35/BestModel11_th_0.8108800373971462voxel_size0.35_0.8924573941853222.pth",
        "Voxel0.5/BestModel7_th_0.7333426631987094voxel_size0.5_0.8951478036268408.pth"]

    th = [0.4205591072142124,0.477454120144248,0.6042602644860744, 0.8108800373971462, 0.7333426631987094]

    for t in datasets2evaluate:
        test_dataset = preprocessingDataset2f(root_data=t, mode="test",voxel=0.1)  # este valid es un test en realidad
        test_data = DataLoader(test_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate, num_workers=1)

        for x,saved in enumerate(models2evaluate):

            model = MinkUNet34C(2, 1).to(device)
            model.load_state_dict(torch.load(saved,map_location=torch.device('cpu')))

            criterion = nn.BCELoss()
            optimizer = SGD(model.parameters(), lr=1e-1)

            all_accuracy = []
            all_f1score = []
            all_recall = []
            all_precision = []
            all_miou = []
            gradient = None
            for i, data in enumerate(tqdm(test_data)):
                optimizer.zero_grad()
                coords, features = data
                print(features.shape)
                test_in_field = ME.TensorField((features).to(dtype=torch.float32),
                                           coordinates=(coords),
                                           quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                           minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

                test_output = model(test_in_field.sparse())
                logit = test_output.slice(test_in_field)
                pred_raw=logit.F.detach().cpu().numpy()
                pred = np.where(pred_raw > th, 1, 0)
                # Pintar las nubes con las inferencias y sacar el accuracy junto con la matriz de confusion
                if gradient == 0:
                    visualize_each_cloud(pred, test_dataset.pcds[i], gradient)
                if gradient == 1:
                    visualize_each_cloud(pred_raw, test_dataset.pcds[i], gradient)
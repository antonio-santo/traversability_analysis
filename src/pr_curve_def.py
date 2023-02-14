#todo la curva precision y recall de datos de test y un nuevo calculo de umbral segun esos ejemplos de test
#basicamente quiero in testLabels pero guardandome todas las preds y labels de nubes sucesivas y hacer un prcurve, de esa pr curve te quedas con el
#umbral que maximice fscore y te la ploteas


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
from dataloader2f import preprocessingDataset2f

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

    # device = torch.device('cuda:x')
    device = torch.device('cpu')
    model = MinkUNet34C(2, 1).to(device)
    # model.load_state_dict(torch.load('',map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(''))

    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=1e-1)
    label_concat = []
    pred_raw_concat = []
    gradient = None
    optimal_th_list = []
    root = "/media/arvc/Extreme SSD/kitti_mine/valid/7"
    directories = sorted(glob.glob('{}/*'.format(root)))
    test_dataset = preprocessingDataset2f(root_data="",mode="valid", voxel=0.2)
    test_data = DataLoader(test_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate,num_workers=1)

    for i, data in enumerate(tqdm(test_data)):
        optimizer.zero_grad()
        coords, features, label = data
        test_in_field = ME.TensorField((features).to(dtype=torch.float32),
                                       coordinates=(coords),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                       minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                       device=device)
        test_label = label.to(device)
        test_output = model(test_in_field.sparse())
        logit = test_output.slice(test_in_field)
        val_loss = criterion(logit.F, test_label.unsqueeze(1).float())
        test_label_gt = test_label.cpu().numpy()
        pred_raw = logit.F.detach().cpu().numpy()
        label_concat.extend(test_label_gt)
        pred_raw_concat.extend(pred_raw)
        if i==1000:
            print("Calculando Precision-Recall curve sobre un conjunto de validation entero")
            precision, recall, thresholds = metrics.precision_recall_curve(np.asarray(label_concat), np.asarray(pred_raw_concat))
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
            pred = np.where(pred_raw_concat > thresholds[ix], 1, 0)
            plt.plot(recall, precision)
            plt.scatter(metrics.recall_score(label_concat, pred), metrics.precision_score(label_concat, pred),c=["orange"])  # optimo de esta nube
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()
            plt.savefig('pr_curve.png')

    #     pred = np.where(pred_raw > th, 1, 0)
    #
    #     # Pintar las nubes con las inferencias
    #     if gradient == 0:
    #         visualize_each_cloud(pred, test_dataset.pcds[i], gradient)
    #     if gradient == 1:
    #         visualize_each_cloud(pred_raw, test_dataset.pcds[i], gradient)

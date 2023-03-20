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
import matplotlib.ticker as ticker
from plyfile import PlyData
import matplotlib.pyplot as plot
import seaborn as sb
import pandas as pd
from dataloader2f import preprocessingDataset2f
from dataloader4f import preprocessingDataset4f


def visualize_each_cloud(pred, pcds):
    cloud = o3d.io.read_point_cloud(pcds)
    points = np.asarray(cloud.points)
    color = []
    for i in pred:
        if i == 1:
            color.append([95, 158, 160])
        if i == 0:
            color.append([106, 90, 205])

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(color).astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    return True

def histogram(euclidean):
    sb.histplot(data=euclidean, kde=True)
    plt.show()



def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = ((values - min(values)) / (1- min(values))) #1 cambiarlo por max(values)
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sb.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)
def plot_rate(df,ylim):

    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 5))
    plt.rcParams["xtick.labelsize"] = 3
    axes[0, 0].set_title('Voxel 0.05m')
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].set_title('Voxel 0.1m')
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].set_title('Voxel 0.2m')
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].set_title('Voxel 0.35m')
    axes[1, 1].set_ylim(0, 1)

    axes[2, 0].set_title('Voxel 0.5m')
    axes[2, 0].set_ylim(0, 1)

    sb.barplot(ax=axes[0, 0], x="distance", y="voxel005", data=df,palette=colors_from_values(df["voxel005"], "Spectral"))
    sb.barplot(ax=axes[0, 1], x="distance", y="voxel01", data=df,palette=colors_from_values(df["voxel01"], "Spectral"))
    sb.barplot(ax=axes[1, 0], x="distance", y="voxel02", data=df,palette=colors_from_values(df["voxel02"], "Spectral"))
    sb.barplot(ax=axes[1, 1], x="distance", y="voxel035", data=df,palette=colors_from_values(df["voxel035"], "Spectral"))
    sb.barplot(ax=axes[2, 0], x="distance", y="voxel05", data=df,palette=colors_from_values(df["voxel05"], "Spectral"))
    axes[2, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[2, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.subplots_adjust(bottom=0.04, right=0.8, top=0.92)
    plt.show()
if __name__ == '__main__':

    models = ["/media/arvc/Extreme SSD/importante/Voxel0.05/BestModel0_th_0.4205591072142124voxel_size0.05_0.9269229441412247.pth",
              "/media/arvc/Extreme SSD/importante/Voxel0.1/BestModel0_th_0.477454120144248voxel_size0.1_0.9225966460906108.pth",
              "/media/arvc/Extreme SSD/importante/Voxel0.2/BestModel2_th_0.6042602644860744voxel_size0.2_0.9015045953476892.pth",
              "/media/arvc/Extreme SSD/importante/Voxel0.35/BestModel11_th_0.8108800373971462voxel_size0.35_0.8924573941853222.pth",
              "/media/arvc/Extreme SSD/importante/Voxel0.5/BestModel7_th_0.7333426631987094voxel_size0.5_0.8951478036268408.pth"]

    voxel = [0.05, 0.1, 0.2, 0.35, 0.5]
    ths = [0.4205591072142124, 0.477454120144248, 0.6042602644860744, 0.8108800373971462, 0.7333426631987094]
    final_tp = []
    final_fp = []
    final_tn = []
    final_fn = []
    distance=np.arange(0,(120),1)
    final_tp.append(distance)
    final_fp.append(distance)
    final_tn.append(distance)
    final_fn.append(distance)
    device = torch.device('cpu')

    for l, m in enumerate(models):
        model = MinkUNet34C(2, 1).to(device)
        model.load_state_dict(torch.load(m,map_location=torch.device('cpu')))
        criterion = nn.BCELoss()
        optimizer = SGD(model.parameters(), lr=1e-1)
        n = 0
        tp = np.zeros((120))
        tn = np.zeros((120))
        fp = np.zeros((120))
        fn = np.zeros((120))
        euclidean = []
        centroids = []
        test_dataset = preprocessingDataset2f(root_data="/media/arvc/Extreme SSD/importante/estudio/clouds", mode="valid", voxel=voxel[l])
        test_data = DataLoader(test_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate, num_workers=1)

        for i, data in enumerate(tqdm(test_data)):
            optimizer.zero_grad()
            coords, features, label = data
            print(coords.size())
            print(features.size())

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
            pred = np.where(pred_raw > ths[l], 1, 0)
            # visualize_each_cloud(pred, test_dataset.pcds[i])

            cloud = o3d.io.read_point_cloud(test_dataset.pcds[i])
            points_orig=np.asarray(cloud.points)
            pcd = cloud.voxel_down_sample_and_trace(1, cloud.get_min_bound(), cloud.get_max_bound(),approximate_class=True)
            centroids=np.asarray(pcd[0].points)
            dist_centroids=np.zeros((10000),dtype=int)
            for k, t in enumerate(pcd[2]):  # points in voxels
                dist_centroids[k]=np.linalg.norm(centroids[k])

            for k, t in enumerate(pcd[2]):  # points in voxels
                    for x in np.array(t):
                        euc=np.linalg.norm(points_orig[x])
                        euclidean.append(euc)
                        if test_label_gt[x] == 1 and pred[x] == 1:
                            tp[int(euc)] = tp[int(euc)] + 1
                        elif test_label_gt[x]==1 and pred[x] ==0:
                            fn[int(euc)] = fn[int(euc)] + 1
                        elif test_label_gt[x] == 0 and pred[x] == 1:
                            fp[int(euc)] = fp[int(euc)] + 1
                        else:
                            tn[int(euc)]=tn[int(euc)]+1
            if i ==13:
                break
        total=tp+tn+fp+fn #Divides toodos los puntos del voxel
        tp = np.divide(tp,total,where=total!=0)
        final_tp.append(tp)
        fp = np.divide(fp,total,where=total!=0)
        final_fp.append(fp)
        tn = np.divide(tn, total, where=total != 0)
        final_tn.append(tn)
        fn = np.divide(fn,total,where=total!=0)
        final_fn.append(fn)
        ylim=np.max(np.asarray(final_tp).T)
    #REPRESENTACION DE LOS DATOS EN PANDAS PARA PASALOR A SEABORN
    df_tp = pd.DataFrame(np.asarray(final_tp).T, columns=['distance','voxel005', 'voxel01', 'voxel02', 'voxel035', 'voxel05'])
    df_fp = pd.DataFrame(np.asarray(final_fp).T, columns=['distance','voxel005', 'voxel01', 'voxel02', 'voxel035', 'voxel05'])
    df_tn = pd.DataFrame(np.asarray(final_tn).T, columns=['distance','voxel005', 'voxel01', 'voxel02', 'voxel035', 'voxel05'])
    df_fn = pd.DataFrame(np.asarray(final_fn).T, columns=['distance','voxel005', 'voxel01', 'voxel02', 'voxel035', 'voxel05'])
    plot_rate(df_tp,ylim)
    plot_rate(df_fp,ylim)
    plot_rate(df_tn,ylim)
    plot_rate(df_fn,ylim)

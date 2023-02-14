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

def plot_count(distance, tp,tn,fp,fn, euclidean):
    d = {'distance': distance, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, }
    df = pd.DataFrame(data=d)
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 5))
    axes[0, 0].set_title('Points per distance')
    axes[1, 0].set_title('True Positives-distance')
    axes[1, 1].set_title('False Positives-distance')
    axes[2, 0].set_title('False Negatives-distance')
    axes[2, 1].set_title('True Negatives-distance')
    sb.histplot(ax=axes[0, 0],data=euclidean,kde=True)
    fig.delaxes(axes[0, 1])
    sb.barplot(ax=axes[1, 0], x="distance", y="tp", data=df)
    sb.barplot(ax=axes[1, 1], x="distance", y="fp", data=df)
    sb.barplot(ax=axes[2, 0], x="distance", y="fn", data=df)
    sb.barplot(ax=axes[2, 1], x="distance", y="tn", data=df)
    plt.show()

def plot_rate(distance, tp,tn,fp,fn, euclidean):
    all=tp+tn+fp+fn

    tpr = np.divide(tp,all, where=all!=0)
    fpr = np.divide(fp, all, where=all!=0)
    fnr = np.divide(fn, all, where=all!=0)
    tnr = np.divide(tn, all, where=all!=0)

    d = {'distance': distance, 'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr}
    df = pd.DataFrame(data=d)
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 5))
    axes[0,0].set_title('Points per distance')
    axes[1, 0].set_title('True Positives Rate-distance')
    axes[1, 1].set_title('False Positives Rate-distance')
    axes[2, 0].set_title('False Negatives Rate-distance')
    axes[2, 1].set_title('True Negatives Rate-distance')
    sb.histplot(ax=axes[0, 0], data=euclidean, kde=True)
    fig.delaxes(axes[0, 1])
    sb.barplot(ax=axes[1, 0], x="distance", y="tpr", data=df,palette="CMRmap_r")
    sb.barplot(ax=axes[1, 1], x="distance", y="fpr", data=df,palette="CMRmap_r")
    sb.barplot(ax=axes[2, 0], x="distance", y="fnr", data=df,palette="CMRmap_r")
    sb.barplot(ax=axes[2, 1], x="distance", y="tnr", data=df,palette="CMRmap_r")
    plt.show()

if __name__ == '__main__':

    device = torch.device('cpu')
    # device = torch.device('cpu')
    model = MinkUNet34C(2, 1).to(device)
    model.load_state_dict(torch.load(
        '/home/arvc/virtual_envs/traversability_analysis/src/models/models_minkunet/all_cloud/24_enero_model_znormal_coordz_0_th_0.3339944563806057_voxel_0.2_0.9188524377540336.pth',map_location=torch.device('cpu')))

    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=1e-1)
    th = 0.3339944563806057
    all_tp = []
    all_tf = []
    all_fp = []
    all_fn = []
    centroids = []
    final1 = []
    final2 = []
    final3 = []
    final4 = []
    test_dataset = preprocessingDataset2f(mode="valid", voxel=0.2)
    test_data = DataLoader(test_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate, num_workers=1)

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
        pred = np.where(pred_raw > th, 1, 0)
        cloud = o3d.io.read_point_cloud(test_dataset.pcds[i])
        points_orig=np.asarray(cloud.points)
        pcd = cloud.voxel_down_sample_and_trace(1, cloud.get_min_bound(), cloud.get_max_bound(),approximate_class=True)
        centroids=np.asarray(pcd[0].points)
        dist_centroids=np.zeros((10000),dtype=int)
        for k, t in enumerate(pcd[2]):  # points in voxels
            dist_centroids[k]=np.linalg.norm(centroids[k])
        distance = np.arange(0,(np.max(dist_centroids)+1),1)

        if i ==0:
            tp=np.zeros((np.max(dist_centroids)+1))
            tn = np.zeros((np.max(dist_centroids)+1))
            fp = np.zeros((np.max(dist_centroids)+1))
            fn = np.zeros((np.max(dist_centroids)+1))
            euclidean=[]


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

        if i ==1:
            plot_rate(distance, tp,tn,fp,fn, euclidean)



"""
- Para verlo todo en una grafica
distance = np.arange(0,(np.max(dist_centroids)+1)*4,1)
        for n,j in enumerate(np.where(distance%4==0)[0]):
            distance[j:j+4]=n
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
                        
all = np.arange(0, (np.max(dist_centroids) + 1) * 4, 1)
all[0:-1:4]=tp
all[1:-1:4] = fp
all[2:-1:4] = fn
all[3:all.shape[0]+1:4] = tn
classes= ["tp","fp", "fn", "tn"]*80
d = {'distance':distance, 'factor':all, 'classes':classes }
df = pd.DataFrame(data=d)
print(df)
sb.barplot(x="distance", y="factor", hue="classes",data=df)
plt.show()
"""
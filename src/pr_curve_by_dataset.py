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
import pandas as pd

if __name__ == '__main__':
    label_concat = []
    pred_raw_concat = []
    gradient = None
    optimal_th_list = []
    names=["Voxel 0.05","Voxel 0.1","Voxel 0.2","Voxel 0.35","Voxel 0.5"]
    root="plots/rellis"
    directories = sorted(glob.glob('{}/*'.format(root)))
    print(directories)
    for t, i in enumerate(tqdm(directories)):
        df=pd.read_csv(i)
        data=df.to_numpy()
        precision, recall, thresholds = metrics.precision_recall_curve(data[:,1], data[:,2])
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        label_concat=data[:,1]
        pred = np.where(data[:,2] > thresholds[ix], 1, 0)
        plt.plot(recall, precision,label=names[t])
        plt.scatter(metrics.recall_score(label_concat, pred), metrics.precision_score(label_concat, pred),c=["orange"])  # optimo de esta nube
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
    plt.show()
    # plt.savefig('pr_curve.png')

    #     pred = np.where(pred_raw > th, 1, 0)
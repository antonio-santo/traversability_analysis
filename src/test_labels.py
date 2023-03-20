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
from dataloader4f import preprocessingDataset4f
import pandas as pd
import visualize_trav as vis

if __name__ == '__main__':

    device = torch.device('cpu')
    datasets2evaluate=["/media/arvc/Extreme SSD/dataset_noRecortado_normalesHibridas/valid/rellis/",
                       "/media/arvc/Extreme SSD/dataset_noRecortado_normalesHibridas/valid/kitti/"]
    models2evaluate=[
        "models_def/modelos_normales_vanilla_new/Voxel0.05/BestModel6_th_0.16232705751433968voxel_size0.05_0.9310462879029038.pth",
        "models_def/modelos_normales_vanilla_new/Voxel0.1/BestModel3_th_0.277520599886775voxel_size0.1_0.9289806581258391.pth",
        "models_def/modelos_normales_vanilla_new/Voxel0.2/BestModel0_th_0.2926777680963278voxel_size0.2_0.9157265003338941.pth",
        "models_def/modelos_normales_vanilla_new/Voxel0.35/BestModel2_th_0.28900030519813297voxel_size0.35_0.9026341045830466.pth",
        "models_def/modelos_normales_vanilla_new/Voxel0.5/BestModel0_th_0.2485628306493163voxel_size0.5_0.9078739783473674.pth"]

    th = [0.16232705751433968,0.277520599886775, 0.2926777680963278, 0.28900030519813297, 0.2485628306493163]
    voxels=[0.05,0.1,0.2,0.35,0.5]

    dataset = []
    model_name = []
    accuracy_final = []
    F1_final = []
    recall_final = []
    precision_final = []
    MIOU_final = []

    for x, l in enumerate(models2evaluate):
        model = MinkUNet34C(2, 1).to(device)
        model.load_state_dict(torch.load(l))
        criterion = nn.BCELoss()
        optimizer = SGD(model.parameters(), lr=1e-1)

        for ey in datasets2evaluate:
            test_dataset = preprocessingDataset2f(root_data=ey, mode="valid",voxel=voxels[x])  # este valid es un test en realidad
            test_data = DataLoader(test_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate, num_workers=1)

            all_accuracy = []
            all_f1score = []
            all_recall = []
            all_precision = []
            all_miou = []
            gradient = None

            for i, data in enumerate(tqdm(test_data)):
                optimizer.zero_grad()
                coords, features, label = data
                coords=coords.to(device)
                features = features.to(device)
                label = label.to(device)

                test_in_field = ME.TensorField((features).to(dtype=torch.float32),
                                               coordinates=(coords),
                                               quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                               minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                               device=device)

                test_output = model(test_in_field.sparse())
                logit = test_output.slice(test_in_field)
                val_loss = criterion(logit.F, label.unsqueeze(1).float())
                test_label_gt = label.cpu().numpy()
                pred_raw = logit.F.detach().cpu().numpy()
                pred = np.where(pred_raw > th[x], 1, 0)

                # Pintar las nubes con las inferencias
                if gradient == 0:
                    vis.visualize_each_cloud_gradient(pred, test_dataset.pcds[i], gradient)
                if gradient == 1:
                    vis.visualize_each_cloud_gradient(pred_raw, test_dataset.pcds[i], gradient)

                all_accuracy.append(metrics.accuracy_score(test_label_gt, pred))
                all_f1score.append(metrics.f1_score(test_label_gt, pred))
                all_recall.append(metrics.recall_score(test_label_gt, pred))
                all_precision.append(metrics.precision_score(test_label_gt, pred))
                all_miou.append(metrics.jaccard_score(test_label_gt, pred))

            dataset.append(ey)
            model_name.append(l)

            accuracy_final.append(sum(all_accuracy) / len(all_accuracy))
            F1_final.append(sum(all_f1score) / len(all_f1score))
            recall_final.append(sum(all_recall) / len(all_recall))
            precision_final.append(sum(all_precision) / len(all_precision))
            MIOU_final.append(sum(all_miou) / len(all_miou))


    df = pd.DataFrame(list(zip(dataset, model_name,precision_final,recall_final,F1_final,accuracy_final,MIOU_final)),
               columns =["Dataset", "model", "precision", "recall", "F1","accuracy","MIOU"])
    print(df)
    df.to_csv("models_def/modelos_normales_vanilla_new/metrics_.csv")

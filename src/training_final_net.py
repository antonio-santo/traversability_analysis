import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
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


def compute_th(net, data, device):
    print("Calculando threshold")
    net.eval()
    optimal_th_list = []
    with torch.no_grad():
        for i, cloud in enumerate(tqdm(data)):
            test_coords, test_feats, test_label = cloud
            test_in_field = ME.TensorField(test_feats.to(dtype=torch.float32),
                                      coordinates=test_coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

            test_label = test_label.to(device)
            test_output = net(test_in_field.sparse())
            logit = test_output.slice(test_in_field)
            test_label_gt = test_label.cpu().numpy()
            precision, recall, thresholds = metrics.precision_recall_curve(test_label_gt, logit.F.cpu().numpy())
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
            optimal_th_list.append(thresholds[ix])
        return sum(optimal_th_list)/len(optimal_th_list)


def test(net, data, device, validation_loss, mean_th):
    net.eval()
    all_accuracy = []
    all_recall = []
    all_precision = []
    print("Calculando métricas")
    with torch.no_grad():
        for i, cloud in enumerate(tqdm(data)):
            test_coords, test_feats, test_label = cloud
            test_in_field = ME.TensorField(test_feats.to(dtype=torch.float32),
                                      coordinates=test_coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

            test_label = test_label.to(device)
            test_output = net(test_in_field.sparse())
            logit = test_output.slice(test_in_field)
            val_loss = criterion(logit.F, test_label.unsqueeze(1).float())
            test_label_gt = test_label.cpu().numpy()
            pred=np.where(logit.F.cpu().numpy() > mean_th, 1, 0)
            all_accuracy.append(metrics.accuracy_score(pred, test_label_gt))
            all_recall.append(metrics.recall_score(test_label_gt, pred))
            all_precision.append(metrics.precision_score(test_label_gt, pred))
            validation_loss.append(val_loss.item())
            print('\t\t Loss:', val_loss.item())


        mean_acc=sum(all_accuracy) / len(all_accuracy)
        mean_r=sum(all_recall) / len(all_recall)
        mean_p=sum(all_precision) / len(all_precision)
        print('Mean Accuracy all batches of validation:',mean_acc , '\t Threshold:', mean_th)
        print('Mean Precision all batches of validation:', mean_p, '\t Threshold:', mean_th)
        print('Mean Recall all batches of validation:', mean_r, '\t Threshold:', mean_th)


    return mean_acc, validation_loss,mean_r,mean_p


def loss_curve(training_loss, validation_loss, epochs,t):
    f1 = plt.figure()
    plt.plot(np.linspace(0,epochs, num=len(training_loss)), training_loss, 'g', label='Training loss')
    plt.plot(np.linspace(0, epochs, num=len(validation_loss)), validation_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("trainingVSvalidation/voxel_size_training_"+str(t)+".png")
    plt.clf()
    return 0

def load_data(voxel_size):

    full_dataset_rellis = preprocessingDataset2f("/home/arvc/DATASETS/Rellis3dxdd",mode="train", voxel=voxel_size)
    full_dataset_kitti = preprocessingDataset2f( "/home/arvc/DATASETS/kitti_mine/train/sequences",mode="train", voxel=voxel_size,)
    datasets_train = ConcatDataset([full_dataset_rellis, full_dataset_kitti])
    train_data_final = DataLoader(datasets_train, batch_size=5, collate_fn=ME.utils.batch_sparse_collate,
                                  num_workers=15,shuffle=True)

    valid_data_kitti_rellis = preprocessingDataset2f( "root valid xd",mode="valid", voxel=voxel_size,) #1000 nubes de cada dataset de valid secuencias 8 y 0
    valid_data_final = DataLoader(valid_data_kitti_rellis, batch_size=50, collate_fn=ME.utils.batch_sparse_collate,
                                  num_workers=15,shuffle=True)

    return train_data_final, valid_data_final



if __name__ == '__main__':

    #PARAMETROS DE ENTRENAMIENTO normales y z sin normalizar

    voxel_sizes = [0.05, 0.1, 0.2, 0.35]
    training_loss = []
    validation_loss = []

    for t in range(len(voxel_sizes)):
        isExist = os.path.exists("Voxel"+str(voxel_sizes[t]))
        if not isExist:
            os.mkdir("Voxel"+str(voxel_sizes[t]))
        epochs = 1
        best_metric = 0
        best_threshold = []
        train_data_final, valid_data_final = load_data(voxel_sizes[t])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.BCELoss()
        model = MinkUNet34C(2, 1).to(device)
        optimizer = SGD(model.parameters(), lr=1e-1)

        for epoch in range(epochs):
            model.train()
            print("Estoy en la epoca:", epoch)
            for i,data in enumerate(tqdm(train_data_final)):
                coords, feats, label= data
                in_field = ME.TensorField(feats.to(dtype=torch.float32),coordinates=coords,quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                          minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,device=device)
                # Forward
                input = in_field.sparse()
                output = model(input)
                out_field = output.slice(in_field)
                # Loss
                loss = criterion(out_field.F, label.to(device).unsqueeze(1).float()) #.F son las features, esta funcion es una clase abstracta para calcular el gradiente
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Iteration batch: ', i, ', Loss: ', loss.item())
                training_loss.append(loss.item())
            #VALIDATION EACH EPOCH
            th_opt = compute_th(model, valid_data_final, device)
            mean_acc, validation_loss,mean_r,mean_p=test(model, valid_data_final, device, validation_loss, th_opt)

            if best_metric < mean_p:
                print("Se guarda el modelo de la epoca:", epoch)
                best_metric = mean_p
                torch.save(model.state_dict(),
                'Voxel'+str(voxel_sizes[t])+'/BestModel'+str(epoch)+'_th_'+str(th_opt)+"voxel_size"+
                str(voxel_sizes[t])+'_'+str(mean_p)+'.pth')
                print("------------------")
            else:
                print("No se guarda este modelo, ya que no mejora lo anterior", epoch)
                print("------------------")

        # When training is finished plot the curve
        loss_curve(training_loss, validation_loss, epochs,voxel_sizes[t])
        training_loss.clear()
        validation_loss.clear()

"""
# coords, feats from a data loader
print(len(coords))  # 227742
tfield = ME.TensorField(coordinates=coords, features=feats, quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
print(len(tfield))  # 227742
sinput = tfield.sparse() # 161890 quantization results in fewer voxels
soutput = MinkUNet(sinput)
print(len(soutput))  # 161890 Output with the same resolution
ofield = soutput.slice(tfield)
assert isinstance(ofield, ME.TensorField)
len(ofield) == len(coords)  # recovers the original ordering and length
assert isinstance(ofield.F, torch.Tensor)  # .F returns the features
"""

# VALIDATION EACH 5 BATCHES
# if i != 0 and i % 5 == 0:
#     accuracy_valid, validation_loss = test(model, valid_data, "valid", device,validation_loss)
#     model.train()
#     if best_metric < accuracy_valid:
#         best_metric = accuracy_valid
#         torch.save(model.state_dict(), 'models/model.pth')
#
#     print(f"Validation accuracy: {accuracy_valid}. Best accuracy: {best_metric}")
# Usar el 90% y 10% del resto en valid
# full_dataset_rellis = preprocessingDataset()
# train_size = int(0.9 * len(full_dataset_rellis))
# valid_size = len(full_dataset_rellis) - train_size
# train_data_rellis, valid_data_rellis = torch.utils.data.random_split(full_dataset_rellis, [train_size, valid_size])
#
# full_dataset_kitti = preprocessingDataset("/home/arvc/Antonio/kitti_mine/train/sequences")
# train_size2 = int(0.9 * len(full_dataset_kitti))
# valid_size2 = len(full_dataset_kitti) - train_size2
# train_data_kitti, valid_data_kitti = torch.utils.data.random_split(full_dataset_kitti, [train_size2, valid_size2])
#
# datasets_train = ConcatDataset([train_data_rellis, train_data_kitti])
# datasets_valid = ConcatDataset([valid_data_rellis, valid_data_kitti])
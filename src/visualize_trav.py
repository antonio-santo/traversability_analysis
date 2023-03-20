import numpy as np
import open3d as o3d

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


def visualize_each_cloud_gradient(pred, pcds, gradient):
    cloud = o3d.io.read_point_cloud(pcds)
    points = np.asarray(cloud.points)
    # distances = np.linalg.norm(points, axis=1)
    # umbral_dis_vis = np.where(distances >= 15)
    # new_points = np.delete(points, umbral_dis_vis[0], axis=0)
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
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    return True



def visualize4(pred, pcds, labels):
    cloud = o3d.io.read_point_cloud(pcds)
    points = np.asarray(cloud.points)
    color = []
    for x,i in enumerate(pred):
        if i == 1 and i ==labels[x]:
            color.append([95, 158, 160]) #tp verde
        elif i==1 and i!=labels[x]:
            color.append([199,0,57]) #fp rojo
        elif i==0 and i==labels[x]:
            color.append([106, 90, 205]) #tn violeta
        elif i==0 and i!=labels[x]:
            color.append([255,195,0]) #fn naranja
        else:
            pass

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(color).astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    return True
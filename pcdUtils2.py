import numpy as np
import open3d as o3d
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.neighbors import radius_neighbors_graph

def normalize_tensor(points):
    points = torch.tensor(points)
    points = (points - points.min()) / (points.max() - points.min()) * 2 - 1
    points = points.detach().numpy()
    return points


def ransac(xyz, threshold, iterations=70000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, plane_cloud = pcd.segment_plane(distance_threshold = threshold, ransac_n = 3,num_iterations = iterations)
    inlier_pcd = pcd.select_by_index(plane_cloud)
    outlier_pcd = pcd.select_by_index(plane_cloud, invert=True)
    inliers = np.asarray(inlier_pcd.points)
    outliers = np.asarray(outlier_pcd.points)

    return inliers, outliers

def normal_split(pcd_raw, xyz, threshold):
    pcd_raw.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # normal threshold
    thresh = 0.95
    # get min max along 3 axes
    min_arr = np.min(xyz, axis=0)
    max_arr = np.max(xyz, axis=0)
    # get spitting list
    int_size = []
    interval_points = []
    for i in range(3):
        ax_range = max_arr[i] - min_arr[i]
        num_intervals = int(ax_range // threshold)
        size_interval = ax_range / num_intervals
        points_interval = [[] for _ in range(num_intervals + 1)]

        int_size.append(size_interval)
        interval_points.append(points_interval)

    
    for j in range(len(pcd_raw.points)):# tqdm(range(len(pcd_raw.points)), desc="normal_split: "):
        normal_vec = pcd_raw.normals[j]
        cos_z = np.abs(normal_vec[2])
        cos_y = np.abs(normal_vec[1])
        # cos_x = np.abs(normal_vec[0])
        curr_point  = pcd_raw.points[j].tolist()
        if (cos_z > thresh) or (cos_y > thresh):
            if cos_z > thresh:
                z_int_idx = int((curr_point[2] - min_arr[2]) / int_size[2])
                interval_points[2][z_int_idx].append(curr_point)
            if cos_y > thresh:
                y_int_idx = int((curr_point[1] - min_arr[1]) / int_size[1])
                interval_points[1][y_int_idx].append(curr_point)
        else:
            x_int_idx = int((curr_point[0] - min_arr[0]) / int_size[0])
            interval_points[0][x_int_idx].append(curr_point)

    return interval_points


def find_node(xyz, method="knn"):
    
    # Step 1: Point Cloud to Mesh
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(xyz)
    pcd_raw.estimate_normals()

    #radius determination
    distances = pcd_raw.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    factor = 1
    if method=="knn":
        arr = np.asarray(pcd_raw.points)
        A_sparse = radius_neighbors_graph(arr, avg_dist*factor, mode='connectivity', include_self=False)
        e1 = A_sparse.nonzero()[0].reshape(-1, 1)
        e2 = A_sparse.nonzero()[1].reshape(-1, 1)
        edges = np.concatenate((e1, e2), axis=1)
        
    arr = np.hstack((np.asarray(pcd_raw.points), np.asarray(pcd_raw.normals)))
    nodes = torch.tensor(arr, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
      
    data1 = Data(x=nodes, edge_index=edge_index)

    return data1

def predict_GCN(pcd, req_label, sequence_length, batch_size, torch_model, device, shuffle_data = True):
    n = len(pcd)
    xyz = normalize_tensor(pcd[:, :3])  # normalize the data
    if shuffle_data:
        randomize = np.arange(n)
        np.random.shuffle(randomize)
        pcd = pcd[randomize]
        xyz = xyz[randomize]
    
    Tbatch = n // (batch_size*sequence_length)
    xyz1 = xyz[: Tbatch * batch_size * sequence_length]

    split_xyz1 = np.split(xyz1, Tbatch)
    train_list = []
    for i in range(Tbatch):
        xyz11 = split_xyz1[i]
        data1 = find_node(xyz11[:, :3], method="knn")
        train_list.append(data1)

    data_loader =  DataLoader(train_list, batch_size=1, shuffle=False)

    pred_label = []
    for pred_data in data_loader:
        pred_data.to(device)
        
        predicted_labels = torch_model(pred_data.x, pred_data.edge_index)
        predicted_labels = torch.argmax(predicted_labels, dim=1, keepdim=True)
        pred_label.append(predicted_labels.detach().cpu().numpy())

    pred_label = np.concatenate(pred_label)
    sequence = pcd[: Tbatch * batch_size * sequence_length]
    rem_sequence = pcd[Tbatch * batch_size * sequence_length: ]

    new_xyz = sequence[np.where(pred_label == req_label)[0]]
    no_new_xyz = sequence[np.where(pred_label != req_label)[0]]
    if len(no_new_xyz) > 0:
        if len(rem_sequence) > 0:
            no_new_xyz = np.vstack((no_new_xyz, rem_sequence))

    return new_xyz, no_new_xyz




class part_seg2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(part_seg2, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        # CNN layers
        self.cnn1 = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.cnn2 = nn.Conv1d(32, 64, kernel_size=1)
        self.cnn3 = nn.Conv1d(64, 128, kernel_size=1)
        
        # GCN layers
        self.gcn1 = GCNConv(128, 256)
        self.gcn2 = GCNConv(256, 512)
        self.gcn3 = GCNConv(512, output_dim)
        
    def forward(self, x, edge_index=None):
        # CNN forward pass
        x = x.view(1, x.size(0), x.size(1))
        x = x.transpose(2,1)  # Reshape input to (batch_size, 6, n)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        
        # GCN forward pass
        x = x.transpose(2,1)  # Reshape back to (batch_size, n, 128)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        # x = F.relu(self.gcn3(x, edge_index))
        x = self.gcn3(x, edge_index)

        x = torch.softmax(x.view(-1, self.output_dim), dim=1)
        
        return x
    


def visualization_seg(segment_plane):

    color_data = pd.read_excel("static\colorFilePCD.xlsm")
    color_name = color_data["color_name"].tolist()
    color_code = color_data["rgb_color"].tolist()

    seg_color_name = []
    seg_color_code = []
    color_idx = 0
    
    for i in range(len(segment_plane)):

        if color_idx > len(color_code)-1:
            color_idx = 0

        paint_color = [float(j) for j in color_code[color_idx].split(",")]
        curr_seg_color_name = f"{i+1}.{color_name[color_idx]}"

        seg_color_name.append(curr_seg_color_name)
        seg_color_code.append(paint_color)
        color_idx += 1


    return seg_color_name, seg_color_code
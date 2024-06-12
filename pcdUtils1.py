import torch
import numpy as np
import open3d as o3d
from detect_delimiter import detect
from pcdUtils2 import ransac, normal_split, part_seg2, predict_GCN, visualization_seg


def read_pcd_file(filepath, downsample=True):

    if filepath.endswith('.ply'):
        pcd_raw = o3d.io.read_point_cloud(filepath)

    elif filepath.endswith('.xyz') or filepath.endswith('.txt'):
        #detect delimiter
        f = open(filepath)
        for _ in range(3):
            line = f.readline()
        delimiters = detect(line)

        pcd = np.loadtxt(filepath, skiprows=1, delimiter=delimiters)
        xyz = pcd[:, :3]

        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(xyz)
    
    xyz = np.asarray(pcd_raw.points)
    # print(f"Original point cloud shape : {xyz.shape}")

    # if len(xyz) > 30000000 and downsample:
    #     pcd_raw = pcd_raw.voxel_down_sample(0.05)
    #     xyz = np.asarray(pcd_raw.points)
    #     print(f"(Uniformly) Downsampled point cloud shape : {xyz.shape}")

    if len(xyz) > 30000000 and downsample:
        k_points = int(np.round(len(xyz) / 25000000))
        pcd_raw = pcd_raw.uniform_down_sample(every_k_points = k_points)
        xyz = np.asarray(pcd_raw.points)
        # print(f"(Uniformly) Downsampled point cloud shape : {xyz.shape}")

    distances = pcd_raw.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    thresh_distance = avg_dist * 10
    return pcd_raw, xyz, thresh_distance

def method1(pcd_raw, xyz, threshold_org, iterations=70000):

    segment_planes = []
    total_planes = 20
    req_xyz = xyz

    for j in range(total_planes):
        
        if len(req_xyz) < 1024:
            break

        inlier2, outlier2 = ransac(req_xyz, threshold_org, iterations)

        segment_planes.append(inlier2)
        req_xyz = outlier2

    rem_pcd = o3d.geometry.PointCloud()
    rem_pcd.points = o3d.utility.Vector3dVector(req_xyz)

    return segment_planes, rem_pcd

def method2(pcd_raw, xyz, threshold_org, iterations=70000):

    segment_planes = []
    still_not_used = []
    interval_points = normal_split(pcd_raw, xyz, threshold_org*12)

    for j in range(3):
        num = 2-j
        split_xyz = interval_points[num]

        for i in range(len(split_xyz)):
            req_xyz = np.array(split_xyz[i])
            if len(req_xyz) < 1024:
                continue
            inlier2, outlier2 = ransac(req_xyz, threshold_org, iterations)

            if len(inlier2) > int(len(req_xyz) * (2/3)):
                segment_planes.append(inlier2)
                still_not_used.append(outlier2)
            else:
                still_not_used.append(req_xyz)
    
    not_used_pcd = []
    for k in still_not_used:
        if len(k) > 0:
            not_used_pcd.append(k)
    rem_pcd = o3d.geometry.PointCloud()
    if len(not_used_pcd) > 0:
        not_used_pcd = np.vstack(not_used_pcd)
        rem_pcd.points = o3d.utility.Vector3dVector(not_used_pcd)

    return segment_planes, rem_pcd


def method3(pcd_raw, xyz, threshold_org, iterations=70000):

    sequence_length = 1024
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "gcn"
    torch_model = part_seg2(6, 3)
    model_path = f"{model_name}_PCD.pt"
    checkpoint = torch.load(model_path, map_location=device)
    torch_model.load_state_dict(checkpoint['state_dict'])
    torch_model.to(device)
    torch_model.eval()


    interval_points = normal_split(pcd_raw, xyz, threshold_org*12)

    segment_planes = []
    still_not_used = []
    for j in range(3):
        num = 2-j
        req_label = 1
        if num == 2:
            req_label = 0
        split_xyz = interval_points[num]

        for i in range(len(split_xyz)):
            req_xyz = np.array(split_xyz[i])
            if len(req_xyz) < 1024:
                continue
            inlier2, outlier2 = ransac(req_xyz, threshold_org, iterations)

            if len(inlier2) > batch_size*sequence_length:
                # perform GNN
                inlier, outlier = predict_GCN(inlier2, req_label, sequence_length, batch_size, torch_model, device, shuffle_data = True)
                
                if len(inlier) > int(len(inlier2) * (2/3)):
                    segment_planes.append(inlier)
                    still_not_used.append(outlier)
                    # still_not_used.append(outlier1)
                    still_not_used.append(outlier2)
                else:
                    still_not_used.append(req_xyz)

            else:
                still_not_used.append(req_xyz)
            
                
    not_used_pcd = []
    for k in still_not_used:
        if len(k) > 0:
            not_used_pcd.append(k)
    rem_pcd = o3d.geometry.PointCloud()
    if len(not_used_pcd) > 0:
        not_used_pcd = np.vstack(not_used_pcd)
        rem_pcd.points = o3d.utility.Vector3dVector(not_used_pcd)

    return segment_planes, rem_pcd



def main(method, pcd_raw, xyz, threshold_org, iterations=70000):

    if method == 'option1':
        segment_planes, rem_pcd = method1(pcd_raw, xyz, threshold_org, iterations)
    elif method == 'option2':
        segment_planes, rem_pcd = method2(pcd_raw, xyz, threshold_org, iterations)
    elif method == 'option3':
        segment_planes, rem_pcd = method3(pcd_raw, xyz, threshold_org, iterations)

    
    seg_color_name, seg_color_code = visualization_seg(segment_planes)

    segment_planes.append(rem_pcd.points)
    seg_color_name.append('RestPCD')
    seg_color_code.append([0.6, 0.6, 0.6])

    return segment_planes, seg_color_name, seg_color_code

    
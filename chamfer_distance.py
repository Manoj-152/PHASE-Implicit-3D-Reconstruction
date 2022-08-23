import open3d as o3d
import torch
import numpy as np
from chamferdist import ChamferDistance

if __name__ == '__main__':
    mesh_target = o3d.io.read_triangle_mesh("Preimage_Implicit_DLTaskData/meshes/bunny.obj")
    mesh_result = o3d.io.read_triangle_mesh("Result_Meshes/bunny_50000.ply")

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = mesh_target.vertices
    pcd_target = torch.Tensor(np.array(pcd_target.points)).cuda().unsqueeze(0)

    pcd_result = o3d.geometry.PointCloud()
    pcd_result.points = mesh_result.vertices
    pcd_result = torch.Tensor(np.array(pcd_result.points)).cuda().unsqueeze(0)

    chamferDist = ChamferDistance().cuda()
    dist_forward = chamferDist(pcd_result, pcd_target)
    print(dist_forward.detach().cpu().item())
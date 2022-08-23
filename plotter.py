import torch
import numpy as np
from tqdm import tqdm
import trimesh
from skimage import measure
import pyvista


# Function to obtain the vertices in a cuboidic meshgrid
def get_grid(points, resolution):
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()
    bbox = input_max - input_min
    shortest_axis = np.argmin(bbox)

    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_surface(points, decoder, savename, resolution, mc_value):
    decoder.eval()
    meshexport = None
    grid = get_grid(points.detach()[:,:3], resolution)
    print('Finished grid')
    z = []

    total_iters = len(torch.split(grid['grid_points'], 100000, dim=0))
    pbar = tqdm(total=total_iters)

    with torch.no_grad():
        for i, pnts in enumerate(torch.split(grid["grid_points"], 100000, dim=0)):
            z.append(decoder(pnts).detach().cpu().numpy())
            pbar.update(1)
    z = np.concatenate(z, axis=0)
    print(np.min(z))
    print(np.max(z))
    print(z.shape)

    if not (np.min(z) > mc_value or np.max(z) < mc_value):          # Check to find if points on surface (or close to) are present
        print("Proceeding")
        z = z.astype(np.float64)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))
        
        print(normals.shape)

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values)
        meshexport.export(savename, 'ply')

        mesh = pyvista.read(savename)
        mesh.plot()
    
    else:
        print("Not Proceeding")
import torch
import torch.nn as nn
import numpy as np
import os
import open3d as o3d
import random
from torch.autograd import grad
from plotter import plot_surface


################################################################################
############################# Some Utils Functions #############################
def sample_ball_points(point, sigma, n_per_ball):
    ball_points = np.random.normal(point, sigma, size=(n_per_ball, point.shape[0]))
    return ball_points

def sample_from_omega(bbox, num_points):
    box_corners = np.asarray(bbox.get_box_points())

    min_x, max_x = np.min(box_corners[:, 0], axis=0), np.max(box_corners[:, 0], axis=0)
    min_y, max_y = np.min(box_corners[:, 1], axis=0), np.max(box_corners[:, 1], axis=0)
    min_z, max_z = np.min(box_corners[:, 2], axis=0), np.max(box_corners[:, 2], axis=0)

    sample = [np.array([random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.uniform(min_z, max_z)]) for i in range(num_points)]
    return torch.tensor(np.asarray(sample)).float()

def gradient(inputs, outputs):
    grad_outs = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    points_grad = points_grad[0][:, -3:]
    #print(points_grad.shape)
    return points_grad

def w_function(s):
    return (s ** 2) - 2 * (s.abs()) + 1

################################################################################
############################## Defining the model ##############################
class Fourier(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super(Fourier, self).__init__()
        vec = torch.randn(in_dim, out_dim)*k
        self.register_buffer("vec", vec)

    def forward(self, x):
        x_proj = torch.matmul(2*np.pi*x, self.vec)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


class INR_Model(nn.Module):
    def __init__(self, fourier, k, beta=100, geometric_init = True):
        super(INR_Model, self).__init__()

        self.fourier = fourier
        self.k = k
        self.num_layers = 9
        self.in_dim = 3
        if self.fourier:
            self.ff = Fourier(in_dim = 3, out_dim = 256, k = self.k)
            self.layer0 = nn.Linear(512, 512)
        else:
            self.layer0 = nn.Linear(self.in_dim, 512)
        
        for i in range(1, self.num_layers):
            if i == 3:
                out_dim = 512 - self.in_dim
            elif i == self.num_layers - 1:
                out_dim = 1
            else:
                out_dim = 512

            layer = nn.Linear(512, out_dim)

            if geometric_init:
                if i == self.num_layers - 2:
                    torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(512), std=0.00001)
                    torch.nn.init.constant_(layer.bias, -1)
                else:
                    torch.nn.init.constant_(layer.bias, 0.0)

                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "layer"+str(i), layer)
        
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, input):
        if self.fourier:
            x = self.ff(input)
        else:
            x = input

        for i in range(self.num_layers):
            layer = getattr(self, "layer"+str(i))
            x = layer(x)
            if i == 3:
                x = torch.cat([x,input], -1)

            if i < self.num_layers - 2:
                x = self.activation(x)

        return x


################################################################################
########################### Preparation for Training ###########################
# Setting up Parameters and loading in the train data

base_dir = "Preimage_Implicit_DLTaskData"
filename = "bunny_10000.xyz"
bbox_scale = 1.5
lmbda = 1
mu = 0.2
epsilon = 1e-2
with_fourier = True
k = 4
with_normals = False
points_batch = 3000
num_epochs = 25000
network = INR_Model(with_fourier, k)

optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

pcd_data = np.genfromtxt(os.path.join(base_dir, filename), delimiter=' ')
pcd_data = torch.tensor(pcd_data).float()
balls_data = np.asarray([sample_ball_points(point, sigma=1e-3, n_per_ball=24) for point in np.array(pcd_data[:,:3])])
#print(balls_data.shape)
#print(pcd_data.size())

point_set = o3d.utility.Vector3dVector(np.array(pcd_data[:,:3]))
axisAlignedBox = o3d.geometry.AxisAlignedBoundingBox().create_from_points(point_set)
omega = axisAlignedBox.scale(bbox_scale, axisAlignedBox.get_center())

'''ckpt = torch.load('ckpt_24999.pth')
network.load_state_dict(ckpt['network'])
optimizer.load_state_dict(ckpt['optimizer'])
lr_scheduler.load_state_dict(ckpt['lr_scheduler'])'''

# Visualizing
#pcd = o3d.geometry.PointCloud()
#pcd.points = point_set
#o3d.visualization.draw_geometries([pcd, axisAlignedBox], window_name='input_test')

################################################################################
############################### Start of Training ##############################
print('Starting to train')
pcd_data = pcd_data.cuda()
pcd_data.requires_grad_()
network = network.cuda()

#for epoch in range(num_epochs):
for epoch in range(25000,25001):
    indices = torch.tensor(np.random.choice(pcd_data.shape[0], points_batch, replace=False))
    cur_data = pcd_data[indices]
    cur_balls_data = torch.tensor(balls_data[indices]).float().cuda()
    omega_points = sample_from_omega(omega, points_batch*cur_balls_data.shape[1]).cuda().requires_grad_()
    ball_points = cur_balls_data.reshape(-1, cur_balls_data.shape[-1])

    print(cur_data.shape)
    print(omega_points.shape)
    print(ball_points.shape)

    network.train()
    optimizer.zero_grad()
    # Calculating Reconstruction Loss
    pred_recon = network(ball_points)
    pred_recon = pred_recon.reshape((cur_balls_data.shape[0], cur_balls_data.shape[1]))
    num_balls = pred_recon.shape[1]
    recon_loss = (pred_recon.sum(axis=1) / num_balls).abs().mean()

    # Calculating Energy Loss
    pred_W = network(omega_points)
    grad_u = gradient(omega_points, pred_W)
    W_u = w_function(pred_W.reshape(-1))
    W_loss = (epsilon * grad_u.norm(2, dim=-1) ** 2 + W_u).mean(dim=0)

    # Calculating Normal Loss
    u_x = network(cur_data[:, :3]).reshape(-1, 1)
    del_w_x = (epsilon ** 0.5) * u_x
    if with_normals:
        normals = cur_data[:, -3:]
        normals_loss = (normals - del_w_x).norm(1, dim=1).mean()
    else:
        normals_loss = (1. - del_w_x.norm(2, dim=1)).reshape(-1, 1).norm(2, dim=1).mean()

    loss = lmbda*recon_loss + W_loss + mu*normals_loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    
    if (epoch%100 == 0):
        print("Epoch ", epoch, ": ", loss.item())
        print("Reconstruction Loss: ", np.round(recon_loss.item(),4), ", W Loss: ", np.round(W_loss.item(),4), ", Normal_loss: ", np.round(normals_loss.item()))

    os.makedirs("model_ckpts", exist_ok=True)
    if (epoch%250 == 0 or epoch == num_epochs-1): torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()},
                                                                            "model_ckpts/ckpt_"+str(epoch)+'.pth')


os.makedirs('Result_Meshes', exist_ok=True)
if with_fourier:
    savename = os.path.join('Result_Meshes', filename[:-4]+'_fourier.ply')
else:
    savename = os.path.join('Result_Meshes', filename[:-4]+'.ply')
plot_surface(pcd_data.cpu(), network, savename, resolution=256, mc_value=0.)
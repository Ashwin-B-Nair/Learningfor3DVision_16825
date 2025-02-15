import torch
import pytorch3d as p3d
import torch.nn.functional as F


# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
 
	# Flatten grids for BCE computation
    voxel_src_flat = voxel_src.view(-1)   
    voxel_tgt_flat = voxel_tgt.view(-1)
	# voxel_src_flat = voxel_src.view(voxel_src.size(0),-1)
	# voxel_tgt_flat = voxel_tgt.view(voxel_tgt.size(0),-1)

	loss1 = torch.nn.BCEWithLogitsLoss()
	loss = loss1(voxel_src_flat, voxel_tgt_flat)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	return loss_laplacian
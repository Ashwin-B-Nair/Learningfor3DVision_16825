import torch
import pytorch3d as p3d
import torch.nn.functional as F
from pytorch3d.loss import mesh_laplacian_smoothing

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

	bce_loss = torch.nn.BCELoss()
	loss = bce_loss(voxel_src_flat, voxel_tgt_flat)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	dists_src_to_tgt, _, _ = p3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1)
	dists_tgt_to_src, _, _ = p3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1)
 
	loss_src_to_tgt = torch.mean(dists_src_to_tgt)  # Average over all points in src
	loss_tgt_to_src = torch.mean(dists_tgt_to_src)  # Average over all points in tgt
    
	loss_chamfer = loss_src_to_tgt + loss_tgt_to_src
 
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	# laplacian_matrix, _ = p3d.ops.cot_laplacian(mesh_src.verts_packed(), mesh_src.edges_packed())
	# laplacian_differences = laplacian_matrix @ mesh_src.verts_packed()
    
    # # Compute squared norm of Laplacian differences
	# loss_laplacian = torch.sum(laplacian_differences.pow(2), dim=1).mean()
	
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian
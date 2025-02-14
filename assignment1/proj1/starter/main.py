# %%
import sys
sys.path.append('C:/Users/ashwi/Documents/CMU/Spring_25/Learning_3D_Vision/assignment1/assignment1-master/abijunai_code_proj1/starter')

import dolly_zoom 
import utils 
import pytorch3d
from pytorch3d.utils import ico_sphere
import pytorch3d.renderer
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import camera_transforms
import render_generic
import mcubes

device = torch.device("cpu")

# %%
#-----------------------------------------------------------------
# 1. Practicing with Cameras
#-----------------------------------------------------------------
# 1.1. 360-degree Renders (5 points)

vertices, faces = utils.load_cow_mesh(path="data/cow.obj")
vertices = vertices.unsqueeze(0)
faces = faces.unsqueeze(0)
textures = torch.ones_like(vertices)  # (1, N_v, 3)
textures = textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
mesh = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=pytorch3d.renderer.TexturesVertex(textures),
).to(device)

number_views = 10
azimuth = np.linspace(-180, 180, num=number_views)  

R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 30, azim =azimuth)
cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

image_size = 256
renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
my_images = renderer(mesh.extend(number_views), cameras= cameras, lights= lights)
my_images = my_images.cpu().numpy()[..., :3]
my_images = (my_images * 255).clip(0, 255).astype(np.uint8)
imageio.mimsave("results/360_cow.gif", my_images, fps=4, loop = 0)

print("End of 1.1.")


# # 1.2 Re-creating the Dolly Zoom (10 points)
# dolly_zoom.dolly_zoom(num_frames=30, output_file="results/dolly_zoom.gif")

# print("End of 1.2")
# #-----------------------------------------------------------------

# # %%

# #2. Practicing with Meshes
# #-----------------------------------------------------------------
# ##2.1 Constructing a Tetrahedron (5 points)

# t_vertices = torch.tensor([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0.5, 0.866, 0],
#         [0.5, 0.289, 0.816]
#     ])

# t_faces = torch.tensor([
#     [0, 1, 2],  
#     [0, 2, 3],  
#     [0, 3, 1],  
#     [1, 3, 2]  
# ])

# t_vertices = t_vertices.unsqueeze(0)
# t_faces = t_faces.unsqueeze(0)
# t_textures = torch.ones_like(t_vertices)  # (1, N_v, 3)
# t_textures = t_textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
# mesh = pytorch3d.structures.Meshes(
#     verts=t_vertices,
#     faces=t_faces,
#     textures=pytorch3d.renderer.TexturesVertex(t_textures),
# ).to(device)

# R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 15, azim =azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# image_size = 256
# renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
# lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

# my_images = renderer(mesh.extend(number_views), cameras= cameras, lights= lights)
# my_images = my_images.cpu().numpy()[..., :3]
# my_images = (my_images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/360_tetrahedron.gif", my_images, fps=12, loop = 0)

# print("End of 2.1.")

# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# ##2.2. Constructing a Cube (5 points)

# c_vertices = torch.tensor([
#         [0, 0, 0],
#         [1, 0, 0],
#         [1, 1, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 0, 1],
#         [1, 1, 1],
#         [0, 1, 1]
#     ])
# c_vertices = c_vertices.float()

# c_faces = torch.tensor([
#         [0, 1, 2],
#         [0, 2, 3],
#         [4, 6, 5],
#         [4, 7, 6],
#         [0, 5, 1],
#         [0, 4, 5],
#         [1, 6, 2],
#         [1, 5, 6],
#         [2, 7, 3],
#         [2, 6, 7],
#         [3, 4, 0],
#         [3, 7, 4]
#     ])

# c_vertices = c_vertices.unsqueeze(0)
# c_faces = c_faces.unsqueeze(0)
# c_textures = torch.ones_like(c_vertices)  # (1, N_v, 3)
# c_textures = c_textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
# mesh = pytorch3d.structures.Meshes(
#     verts=c_vertices,
#     faces=c_faces,
#     textures=pytorch3d.renderer.TexturesVertex(c_textures),
# ).to(device)

# number_views = 20
# image_size = 256
# azimuth = np.linspace(-180, 180, num=number_views)  
# R, T = pytorch3d.renderer.look_at_view_transform(dist = 5, elev = 30, azim =azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
# lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

# my_images = renderer(mesh.extend(number_views), cameras= cameras, lights= lights)
# my_images = my_images.cpu().numpy()[..., :3]
# my_images = (my_images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/360_cube.gif", my_images, fps=12, loop = 0)

# print("End of 2.2.")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # 3. Re-texturing a mesh (10 points)
# vertices, faces = utils.load_cow_mesh(path="data/cow.obj")

# textures = torch.ones_like(vertices)

# z_min = torch.min(vertices[:, 2])
# z_max = torch.max(vertices[:, 2])
# color1 = torch.tensor([0, 1 , 0.5])
# color2 = torch.tensor([0, 0, 1])

# for i in range(len(vertices)):
#     alpha = (vertices[i,2] - z_min) / (z_max - z_min)
#     color = alpha * (color2) + (1 - alpha) * (color1)
#     textures[i]= color

# vertices = vertices.unsqueeze(0)
# faces = faces.unsqueeze(0)    
# textures = textures.unsqueeze(0)
# retexture = pytorch3d.renderer.TexturesVertex(textures.to(device))

# re_meshes = pytorch3d.structures.Meshes(
#     verts=vertices,
#     faces=faces,
#     textures=retexture,
# ).to(device)

# number_views = 20
# azimuth = np.linspace(-180, 180, num=number_views)
# R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 30, azim =azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# image_size = 256
# renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
# lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

# images = renderer(re_meshes.extend(number_views), cameras= cameras, lights= lights)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/retexture.gif", images, fps=5, loop = 0)

# print("End of 3.")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # 4. Camera Transformations (10 points)

# # Image 1
# R_relative_1 = torch.tensor([
#         [0, 1, 0],
#         [-1, 0, 0],
#         [0, 0, 1]
#     ])
# T_relative_1 = torch.tensor([0, 0, 0]) 

# image1 = camera_transforms.render_textured_cow(R_relative = R_relative_1, T_relative=T_relative_1)
# plt.imsave("results/cow_trans_1_Q4.jpg", image1)

# # Image 2
# R_relative_2 = torch.tensor([
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]
#     ])
# T_relative_2 = torch.tensor([0, 0, 3]) 

# image2 = camera_transforms.render_textured_cow(R_relative = R_relative_2, T_relative=T_relative_2)
# plt.imsave("results/cow_trans_2_Q4.jpg", image2)

# # Image 3
# R_relative_3 = torch.tensor([
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]
#     ])
# T_relative_3 = torch.tensor([0.5, -0.5, 0]) 

# image3 = camera_transforms.render_textured_cow(R_relative = R_relative_3, T_relative=T_relative_3)
# plt.imsave("results/cow_trans_3_Q4.jpg", image3)

# # Image 4
# R_relative_4 = torch.tensor([
#         [0, 0, 1 ],
#         [0, 1, 0],
#         [-1, 0, 0]
#     ])
# T_relative_4 = torch.tensor([-3, 0, 3]) 
# image4 = camera_transforms.render_textured_cow(R_relative = R_relative_4, T_relative=T_relative_4)
# plt.imsave("results/cow_trans_4_Q4.jpg", image4)

# print("End of 4.")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # 5. Rendering Generic 3D Representations

# # 5.1 Rendering Point Clouds from RGB-D Images (10 points)
# data = render_generic.load_rgbd_data()

# points_1, colors_1 = utils.unproject_depth_image(torch.Tensor(data["rgb1"]), 
#                                               torch.Tensor(data["mask1"]),
#                                               torch.Tensor(data["depth1"]),
#                                               data["cameras1"])

# point_cloud_1 = pytorch3d.structures.Pointclouds(points=points_1.unsqueeze(0), features=colors_1.unsqueeze(0)).to(device)
# number_views = 10
# azimuth = np.linspace(-180, 180, num=number_views)
# image_size = 256

# R, T = pytorch3d.renderer.look_at_view_transform(dist = 10, elev = 0, azim =azimuth)

# #flip point cloud
# R = torch.tensor([
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, 0, 1]
#     ]).float() @ R

# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
# renderer = utils.get_points_renderer(image_size=image_size, device=device)

# images = renderer(point_cloud_1.extend(number_views), cameras= cameras, lights= lights)
# images = images.cpu().numpy()[..., :3] 
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/point_cloud_1.gif", images, fps=4, loop = 0)
# print("End of 5.1- point cloud 1")
# #-----------------------------------------------------------------

# # %%
# points_2, colors_2 = utils.unproject_depth_image(torch.Tensor(data["rgb2"]), 
#                                               torch.Tensor(data["mask2"]),
#                                               torch.Tensor(data["depth2"]),
#                                               data["cameras2"])
# point_cloud_2 = pytorch3d.structures.Pointclouds(points=points_2.unsqueeze(0), features=colors_2.unsqueeze(0))

# images = renderer(point_cloud_2.extend(number_views), cameras= cameras, lights= lights)
# images = images.cpu().numpy()[..., :3] 
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/point_cloud_2.gif", images, fps=4, loop = 0)
# print("End of 5.1- point cloud 2")
# #-----------------------------------------------------------------

# # %%

# point_cloud_3 = pc3 = pytorch3d.structures.Pointclouds(points=torch.cat((points_1,points_2), 0).unsqueeze(0),
#     features=torch.cat((colors_1,colors_2), 0).unsqueeze(0),).to(device)

# images = renderer(point_cloud_3.extend(number_views), cameras= cameras, lights= lights)
# images = images.cpu().numpy()[..., :3] 
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/point_cloud_3.gif", images, fps=5, loop = 0)
# print("End of 5.1- point cloud 3")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # 5.2 Parametric Functions (10 + 5 points)

# phi = torch.linspace(0, 2 * np.pi, 100)
# theta = torch.linspace(0, 2 * np.pi, 100)
# phi, theta = torch.meshgrid(phi, theta)

# x = (3 + torch.cos(theta)) * torch.cos(phi)
# y = (3 + torch.cos(theta)) * torch.sin(phi)
# z = torch.sin(theta)

# points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1).unsqueeze(0)
# color = (points - points.min()) / (points.max() - points.min()).unsqueeze(0)

# torus_point_cloud = pytorch3d.structures.Pointclouds(points=points, features=color)
# R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=0, azim=azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# renderer = utils.get_points_renderer(image_size=image_size, device=device)
# images = renderer(torus_point_cloud.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/torus.gif", images, fps=12, loop=0)

# print("End of 5.2- torus")
# #-----------------------------------------------------------------

# # %%
# # Define parameters for the catenoid
# a = 1.0  
# u = torch.linspace(0, 2 * np.pi, 100)  
# v = torch.linspace(-2, 2, 100)  
# u, v = torch.meshgrid(u, v)

# # Parametric equations for the catenoid
# x = a * torch.cosh(v/a) * torch.cos(u)
# y = a * torch.cosh(v/a) * torch.sin(u)
# z = v

# points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1).unsqueeze(0)
# color = (points - points.min()) / (points.max() - points.min())

# catenoid_point_cloud = pytorch3d.structures.Pointclouds(points=points, features=color)
# R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=0, azim=azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# renderer = utils.get_points_renderer(image_size=image_size, device=device)
# images = renderer(catenoid_point_cloud.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/catenoid.gif", images, fps=12, loop=0)

# print("End of 5.2- new object")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # 5.3 Implicit Surfaces (15 + 5 points)
# device = utils.get_device()
# image_size = 256 
# min_value = -3.1
# max_value = 3.1
# voxel_size = 64
# X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

# voxels = (X ** 2 + Y ** 2 + Z ** 2 + 2 ** 2 - 1 ** 2) ** 2 - 4 * 2 ** 2 * (X ** 2 + Y ** 2)
# vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
# vertices = torch.tensor(vertices).float()
# faces = torch.tensor(faces.astype(int))
# # Vertex coordinates are indexed by array position, so we need to
# # renormalize the coordinate system.
# vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
# textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
# textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

# mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

# number_views = 20
# azimuth = np.linspace(-180, 180, num=number_views)

# renderer = utils.get_mesh_renderer(image_size=image_size, device=device)

# lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)

# R, T = pytorch3d.renderer.look_at_view_transform(dist=15, elev=0, azim=azimuth)

# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

# images = renderer(mesh.extend(number_views), cameras=cameras, lights = lights)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/torus_implicit.gif", images, fps=12, loop=0)
# print("End of 5.3")
# #-----------------------------------------------------------------

# # %%

# device = utils.get_device()
# image_size = 256 
# min_value = -3.1
# max_value = 3.1
# voxel_size = 64

# X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)


# voxels = torch.cos(X)*torch.sin(Y) + torch.cos(Y)*torch.sin(Z) + torch.cos(Z)*torch.sin(X)

# vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
# vertices = torch.tensor(vertices).float()
# faces = torch.tensor(faces.astype(int))
# # Vertex coordinates are indexed by array position, so we need to
# # renormalize the coordinate system.
# vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
# textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
# textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

# mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

# number_views = 20
# azimuth = np.linspace(-180, 180, num=number_views)

# lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)

# renderer = utils.get_mesh_renderer(image_size=image_size, device=device)

# R, T = pytorch3d.renderer.look_at_view_transform(dist=15, elev=0, azim=azimuth)

# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

# images = renderer(mesh.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/gyroid_implicit.gif", images, fps=12, loop=0)

# print("End of 5.3- new object")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # 6. Do Something Fun (10 points)
# vertices, faces = utils.load_cow_mesh(path="data/cow.obj")
# vertices = vertices+torch.tensor([1.2, 3.5, 1])   #offest to set the cow on top of the cube
# vertices = vertices.unsqueeze(0)
# faces = faces.unsqueeze(0)
# textures = torch.ones_like(vertices)  
# textures = textures * torch.tensor([1, 0, 1])  

# cow_mesh = pytorch3d.structures.Meshes(
#     verts=vertices,
#     faces=faces,
#     textures=pytorch3d.renderer.TexturesVertex(textures),
# ).to(device)

# c_vertices = torch.tensor([
#         [0, 0, 0],
#         [3, 0, 0],
#         [3, 3, 0],
#         [0, 3, 0],
#         [0, 0, 3],
#         [3, 0, 3],
#         [3, 3, 3],
#         [0, 3, 3]
#     ])
# c_vertices = c_vertices.float()
# c_vertices = c_vertices.unsqueeze(0)
# c_textures = c_textures * torch.tensor([0.7, 0.7, 1]) 
# cube_mesh = pytorch3d.structures.Meshes(
#     verts=c_vertices,
#     faces=c_faces,
#     textures=pytorch3d.renderer.TexturesVertex(c_textures),
# ).to(device)

# mesh = pytorch3d.structures.meshes.join_meshes_as_scene([cow_mesh, cube_mesh])

# number_views = 50
# dist = 10 + 4 * np.sin(np.linspace(0, 2 * np.pi, num=number_views))  #function to create zooming in and out effect

# azimuth = np.linspace(-180, 180, num=number_views) 
# R, T = pytorch3d.renderer.look_at_view_transform(dist = dist, elev = 30, azim =azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# image_size = 256
# renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
# lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
# images = renderer(mesh.extend(number_views), cameras= cameras, lights= lights)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/Q6.gif", images, fps=12, loop = 0)

# print("End of 6")
# #-----------------------------------------------------------------

# # %%
# #-----------------------------------------------------------------
# # (Extra Credit) 7. Sampling Points on Meshes (10 points
# def sample_mesh_to_point_cloud(mesh_path, num_samples):

#     vertices, faces, _ = pytorch3d.io.load_obj(mesh_path)
    
#     #Finding the vertices of each face
#     face_vertices = vertices[faces.verts_idx]
#     vertices_0 = face_vertices[:, 0]
#     vertices_1 = face_vertices[:, 1]
#     vertices_2 = face_vertices[:, 2]

#     face_areas = 0.5 * torch.norm(torch.cross(vertices_1 - vertices_0, vertices_2 - vertices_0), dim=1)
#     area_weights = face_areas / face_areas.sum()
#     sample_idx = torch.multinomial(area_weights, num_samples, replacement=True)
   
#     u, v = torch.rand(2, num_samples)
#     w0 = 1 - torch.sqrt(u)
#     w1 = torch.sqrt(u) * (1 - v)
#     w2 = torch.sqrt(u) * v
#     weights = torch.stack([w0, w1, w2], dim=1)
    
#     sampled_faces = face_vertices[sample_idx]
#     samples = (weights[:, :, None] * sampled_faces).sum(dim=1)
#     color = (samples - samples.min()) / (samples.max() - samples.min())
#     # point_cloud = pytorch3d.structures.Pointclouds(points=[samples], features=[color])
#     point_cloud = pytorch3d.structures.Pointclouds(points=samples.unsqueeze(0), features=color.unsqueeze(0))
    
#     return point_cloud

# number_views = 20

# mesh_path = "data/cow.obj"

# point_cloud = sample_mesh_to_point_cloud(mesh_path, num_samples=10)

# number_views = 10
# azimuth = np.linspace(-180, 180, num=number_views)
# image_size = 256
# R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 30, azim =azimuth)
# cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
# lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
# renderer = utils.get_points_renderer(image_size=image_size, device=device)

# images = renderer(point_cloud.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/cow_10.gif", images, fps=4, loop=0)


# point_cloud = sample_mesh_to_point_cloud(mesh_path, num_samples=100)
# images = renderer(point_cloud.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/cow_100.gif", images, fps=4, loop=0)

# point_cloud = sample_mesh_to_point_cloud(mesh_path, num_samples=1000)
# images = renderer(point_cloud.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/cow_1000.gif", images, fps=4, loop=0)

# point_cloud = sample_mesh_to_point_cloud(mesh_path, num_samples=10000)
# images = renderer(point_cloud.extend(number_views), cameras=cameras)
# images = images.cpu().numpy()[..., :3]
# images = (images * 255).clip(0, 255).astype(np.uint8)
# imageio.mimsave("results/cow_10000.gif", images, fps=4, loop=0)

# print("End of 7.")
# #-----------------------------------------------------------------



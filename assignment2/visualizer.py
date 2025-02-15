import torch 
import pytorch3d
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from utils import get_device, get_mesh_renderer, get_points_renderer

def voxel_visualizer(voxels, output_path = 'voxel_visualize.gif', textures= None,
                    threshold = 0.5, number_views= 20, image_size=256, distance= 3, 
                    fov =60, fps=12, elev=1):
    device = get_device()
    mesh = pytorch3d.ops.cubify(voxels, thresh=threshold).to(device)
    mesh_visualizer(mesh, output_path,textures= None,number_views= 20, 
                    image_size=256, distance= 3, fov =60, fps=12, elev=1)
    
    return 

def mesh_visualizer(mesh, output_path = 'mesh_visualize.gif', textures= None,
                    number_views= 20, image_size=256, distance= 3, fov =60, 
                    fps=12, elev=1):
    device = get_device()
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)     # (N_f, 3) -> (1, N_f, 3)
    
    if textures is None:
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    
    render_mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)
    
    azimuth = np.linspace(-180, 180, num=number_views)
    R, T = pytorch3d.renderer.look_at_view_transform(dist = distance, elev = elev, 
                                                     azim =azimuth)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov= fov, device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, 3]], device=device)
    
    images = renderer(render_mesh.extend(number_views), cameras= cameras, lights= lights)
    images = images.cpu().numpy()[..., :3]
    images = (images * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, images, fps=fps, format='gif', loop=0)
    
    return
# 16-825 Assignment 1: Rendering Basics with PyTorch3D 

The code to be run is available in  `./starter/main.py` . 

The helper functions such as `utils.py`, `dolly_zoom` etc. has been modified slightly as follows:

In `dolly_zoom.py`, the line `# from starter.utils import get_device, get_mesh_renderer` has been modified to `from utils import get_device, get_mesh_renderer` to avoid an ModuleNotFound error. This similar change has been done across other scripts to avoid the error. 


The outputs are all stored in `./results/` folder. 
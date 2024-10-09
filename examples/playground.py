import math
import torch
from torch import Tensor
from gsplat.cuda._wrapper import _FullyFusedProjectionRade, _make_lazy_cuda_func, _RasterizeToPixelsRADE, isect_tiles, \
    isect_offset_encode
from torch.autograd import Function
from torch.autograd import gradcheck
from typing import Tuple

from gsplat.rendering import rasterization_rade_inria_wrapper

torch.use_deterministic_algorithms(True)
import sys

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the single splat in the center of the scene
means = torch.load("means.pt", weights_only=True).to(device=device)
quats = torch.load("quats.pt", weights_only=True).to(device=device)
scales = torch.load("scales.pt", weights_only=True).to(device=device)
opacities = torch.load("opacities.pt", weights_only=True).to(device=device)
colors = torch.load("colors.pt", weights_only=True).to(device=device)
viewmats = torch.load("viewmats.pt", weights_only=True).to(device=device)
Ks = torch.load("Ks.pt", weights_only=True).to(device=device)

width = 765
height = 572

render_colors = None
render_alphas = None
render_depths = None
render_normals = None


fully_fused_projection = _FullyFusedProjectionRade.apply
rasterize_to_pixels = _RasterizeToPixelsRADE.apply

# Forward pass
radii, means2d, conics, depths, ts, normals, camera_planes, ray_planes = fully_fused_projection(means, quats, scales, viewmats, opacities, Ks, width, height)

tile_size=16

C = viewmats.shape[0]

# Identify intersecting tiles
tile_width = math.ceil(width / float(tile_size))
tile_height = math.ceil(height / float(tile_size))
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d,
    radii,
    depths,
    tile_size,
    tile_width,
    tile_height,
    packed=False,
    n_cameras=C,
    camera_ids=None,
    gaussian_ids=None,
)
# print("rank", world_rank, "Before isect_offset_encode")
isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)


ret = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            camera_planes,
            ray_planes,
            normals,
            ts,
            Ks[0],
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=None,
            packed=False,
            absgrad=False,
        )



























double_means = means.double().requires_grad_()
double_quats = quats.double()#.requires_grad_()
double_scales = scales.double()#.requires_grad_()
double_viewmats = viewmats.double()#.requires_grad_()
double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
double_Ks = Ks.double()#.requires_grad_()































test = _FullyFusedProjectionRadePlayground.apply
input = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks)

outputs = test(*input)
#print(outputs)


#gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4, nondet_tol=1e-5)
#print("GRADCHECK 2", gradcheck_result)
#
#if not gradcheck_result:
#    raise Exception("Gradcheck failed")


n_start = 89
n_end = 90

#n_start = 0
#n_end = 50

means = torch.load("means.pt").to(device=device)[n_start:n_end]
quats = torch.load("quats.pt").to(device=device)[n_start:n_end]
scales = torch.load("scales.pt").to(device=device)[n_start:n_end]
viewmats = torch.load("viewmats.pt").to(device=device)
Ks = torch.load("Ks.pt").to(device=device)


width=764
height=572
eps2d=0.3
near_plane=0.01
far_plane=10000000000.0
radius_clip=0.0
calc_compensations=False
ortho=False

double_means = means.double().requires_grad_()
double_quats = quats.double()#.requires_grad_()
double_scales = scales.double()#.requires_grad_()
double_viewmats = viewmats.double()#.requires_grad_()
double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
double_Ks = Ks.double()#.requires_grad_()

input = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks, width, height)

outputs = test(*input)

gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4)
print("GRADCHECK", gradcheck_result)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

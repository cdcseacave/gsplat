import math
import torch
from torch import Tensor
from gsplat.cuda._wrapper import _FullyFusedProjectionRade, _make_lazy_cuda_func, _RasterizeToPixelsRADE, isect_tiles, \
    isect_offset_encode, _FullyFusedProjection
from torch.autograd import Function
from torch.autograd import gradcheck
from typing import Tuple

from gsplat.rendering import rasterization_rade_inria_wrapper

torch.use_deterministic_algorithms(True)
import sys
















width = 765
height = 572

start = 0
end = 5

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the single splat in the center of the scene
means = torch.load("means.pt", weights_only=True).to(device=device)
quats = torch.load("quats.pt", weights_only=True).to(device=device)
scales = torch.load("scales.pt", weights_only=True).to(device=device)
opacities = torch.load("opacities.pt", weights_only=True).to(device=device).contiguous()
colors = torch.load("colors.pt", weights_only=True).to(device=device)
viewmats = torch.load("viewmats.pt", weights_only=True).to(device=device).contiguous()
Ks = torch.load("Ks.pt", weights_only=True).to(device=device)

if start is not None and end is not None:
    means = means[start:end]
    quats = quats[start:end]
    scales = scales[start:end]
    opacities = opacities[start:end]
    colors = colors[start:end]
    viewmats = viewmats[start:end]
    Ks = Ks[start:end]

means = means.double()
quats = quats.double()
scales = scales.double()
opacities = opacities.double()
colors = colors.double()
viewmats = viewmats.double()
Ks = Ks.double()


fully_fused_projection_rade = _FullyFusedProjectionRade.apply
#
# input_rade = (means, quats, scales, opacities, viewmats, Ks, width, height, 0.3, 0.01, 1e10, 0.0, False, False)
#
# gradcheck_result = gradcheck(fully_fused_projection_rade, input_rade, eps=1e-5)
#
# if not gradcheck_result:
#     raise Exception("Gradcheck failed")
#
# print("Gradcheck passed")
# sys.exit(0)





# Forward pass
radii, means2d, conics, depths, camera_planes, normals, ray_planes, ts = fully_fused_projection_rade(means, quats, scales, opacities, viewmats,
                                                                                                Ks, width, height,
                                                                                                0.3,
                                                                                                0.01,
                                                                                                1e10,
                                                                                                0.0,
                                                                                                False,
                                                                                                False,
                                                                                                )

means2d = means2d.double()
conics = conics.double()
depths = depths.double()
camera_planes = camera_planes.double()
ray_planes = ray_planes.double()
normals = normals.double()
ts = ts.double()



tile_size=16

C = viewmats.shape[0]

# Identify intersecting tiles
tile_width = math.ceil(width / float(tile_size))
tile_height = math.ceil(height / float(tile_size))
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
    means2d,
    radii.int(),
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


rasterize_to_pixels = _RasterizeToPixelsRADE.apply

input = (means2d,
         conics,
         colors,
         opacities,
         camera_planes,
         ray_planes,
         normals,
         ts,
         Ks[0].double(),
         None,
         None,
         width,
         height,
         tile_size,
         isect_offsets,
         flatten_ids,
         False)



gradcheck_result = gradcheck(rasterize_to_pixels, input, eps=1e-5)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

print("Gradcheck passed")
sys.exit(0)























double_means = means.double().requires_grad_()
double_quats = quats.double()#.requires_grad_()
double_scales = scales.double()#.requires_grad_()
double_viewmats = viewmats.double()#.requires_grad_()
double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
double_Ks = Ks.double()#.requires_grad_()































test = _FullyFusedProjectionRade.apply
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



input_rade = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks, width, height)
input_3dgs = (double_means, None, double_quats, double_scales, double_viewmats, double_Ks, width, height)

outputs = test(*input)

gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4)
print("GRADCHECK", gradcheck_result)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

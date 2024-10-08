import torch
from gsplat.rendering import rasterization_radegs
import tifffile


n_start = 0
n_end = -1

device = torch.device("cuda:0")

means = torch.load("means.pt", weights_only=True).to(device=device)[n_start:n_end]
quats = torch.load("quats.pt", weights_only=True).to(device=device)[n_start:n_end]
scales = torch.load("scales.pt", weights_only=True).to(device=device)[n_start:n_end]
viewmats = torch.load("viewmats.pt", weights_only=True).to(device=device)
Ks = torch.load("Ks.pt", weights_only=True).to(device=device)
colors = torch.rand((means.shape[0], 3), device=device)
opacities = torch.ones(means.shape[0], dtype=torch.float32).to(device=device)


width=764
height=572
eps2d=0.3
near_plane=0.01
far_plane=10000000000.0
radius_clip=0.0
calc_compensations=False
ortho=False


print("means", means.shape)
print("quats", quats.shape)
print("scales", scales.shape)
print("viewmats", viewmats.shape)
print("Ks", Ks.shape)
print("colors", colors.shape)

N = means.shape[0]
C = viewmats.shape[0]

print("N=", N)
print("C=", C)

print("colors dim", colors.dim())
print("colors shape", colors.shape[:2], "C=", C, "N=", N)

assert (colors.dim() == 2 and colors.shape[0] == N) or (
    colors.dim() == 3 and colors.shape[:2] == (C, N)
), colors.shape

#double_means = means.double().requires_grad_()
#double_quats = quats.double()#.requires_grad_()
#double_scales = scales.double()#.requires_grad_()
#double_viewmats = viewmats.double()#.requires_grad_()
#double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
#double_Ks = Ks.double()#.requires_grad_()
#
#input = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks, width, height)

rasterization_out = rasterization_radegs(means,
                                  quats,
                                 scales,
                              opacities,
                                 colors,
                               viewmats,
                                     Ks,
                                         width,
                                         height)


(render_colors, render_alphas, render_depths, render_normals, meta) = rasterization_out

print("render_colors", render_colors.shape)
print("render_alphas", render_alphas.shape)
print("render_depths", render_depths.shape)
print("render_normals", render_normals.shape)

tifffile.imwrite("render_colors.tiff", render_colors.cpu().detach().numpy())
tifffile.imwrite("render_alphas.tiff", render_alphas.cpu().detach().numpy())
tifffile.imwrite("render_depths.tiff", render_depths.cpu().detach().numpy())
tifffile.imwrite("render_normals.tiff", render_normals.cpu().detach().numpy())


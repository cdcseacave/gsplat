import torch
from gsplat.rendering import rasterization
import tifffile

width = 763
height = 571


def check(input, gt, name):
    diff = (input - gt).abs()
    if (diff.max() > 1e-6):
        print("Diff", name, ": ", diff.max(), diff.mean())
        print("    Max @ ", torch.unravel_index(diff.argmax(), diff.shape))
        print("    Value @ ", diff[torch.unravel_index(diff.argmax(), diff.shape)])
        print("    Value @ ", diff[torch.unravel_index(diff.argmax(), diff.shape)[0],:])
        print("    Value @ ", input[torch.unravel_index(diff.argmax(), diff.shape)[0],:])
        print("    Value @ ", gt[torch.unravel_index(diff.argmax(), diff.shape)[0],:])
    else:
        print("[OK] ", name)


def get_normal_test(means, quats, scales, viewmat):
    normal_global = get_smallest_axis(means, quats, scales, viewmat)


    gt_normal_global = torch.load("/home/paul/vision/PGSR/normal_global.pth", weights_only=True).to(device="cuda")
    check(normal_global, gt_normal_global, "normal_global")


    camera_center = -viewmat[:3, :3].T @ viewmat[:3, 3]
    gaussian_to_cam_global = camera_center - means
    # gaussian_to_cam_global = viewmat[:3,3] - means
    neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
    normal_global[neg_mask] = -normal_global[neg_mask]



    gt_gaussian_to_cam_global = torch.load("/home/paul/vision/PGSR/gaussian_to_cam_global.pth", weights_only=True).to(device="cuda")
    # gt_neg_mask = torch.load("/home/paul/vision/PGSR/neg_mask.pth", weights_only=True).to(device="cuda")

    check(gaussian_to_cam_global, gt_gaussian_to_cam_global, "gaussian_to_cam_global")
    # check(neg_mask, gt_neg_mask, "neg_mask")


    return normal_global


def compute_all_map_with_test(means, quats, scales, viewmat):
    global_normal = get_normal_test(means, quats, scales, viewmat)
    local_normal = global_normal @ viewmat[:3,:3].T
    pts_in_cam = means @ viewmat[:3, :3].T + viewmat[:3,3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance


    gt_global_normal = torch.load("/home/paul/vision/PGSR/global_normal.pth", weights_only=True).to(device="cuda")
    gt_local_normal = torch.load("/home/paul/vision/PGSR/local_normal.pth", weights_only=True).to(device="cuda")
    gt_pts_in_cam = torch.load("/home/paul/vision/PGSR/pts_in_cam.pth", weights_only=True).to(device="cuda")
    gt_depth_z = torch.load("/home/paul/vision/PGSR/depth_z.pth", weights_only=True).to(device="cuda")
    gt_local_distance = torch.load("/home/paul/vision/PGSR/local_distance.pth", weights_only=True).to(device="cuda")
    gt_input_all_map = torch.load("/home/paul/vision/PGSR/input_all_map.pth", weights_only=True).to(device="cuda")


    check(global_normal, gt_global_normal, "global_normal in compute all map")
    check(local_normal, gt_local_normal, "local_normal")
    check(pts_in_cam, gt_pts_in_cam, "pts_in_cam")
    check(depth_z, gt_depth_z, "depth_z")
    check(local_distance, gt_local_distance, "local_distance")
    check(input_all_map, gt_input_all_map, "input_all_map")


    return input_all_map




def load_initial_params(path = "/home/paul/vision/PGSR", prefix="render_"):
        means = torch.load(f"{path}/{prefix}means.pth", weights_only=True).to(device="cuda")
        features_dc = torch.load(f"{path}/{prefix}features_dc.pth", weights_only=True).to(device="cuda")
        features_rest = torch.load(f"{path}/{prefix}features_rest.pth", weights_only=True).to(device="cuda")
        opacity = torch.load(f"{path}/{prefix}opacity.pth", weights_only=True).to(device="cuda")
        rotation = torch.load(f"{path}/{prefix}rotation.pth", weights_only=True).to(device="cuda")
        scaling = torch.load(f"{path}/{prefix}scaling.pth", weights_only=True).to(device="cuda")

        return means, features_dc, features_rest, opacity, rotation, scaling

(means, features_dc, features_rest, opacity, rotation, scaling) = load_initial_params()

#
#
# viewmat = torch.tensor([[ 0.97184986, -0.07632249,  0.22289620,  0.00000000],
#                         [ 0.10195698,  0.98914152, -0.10584807,  0.00000000],
#                         [-0.21239731,  0.12559426,  0.96907866,  0.00000000],
#                         [-0.07890126, -2.22838664,  2.71024823,  1.00000000]]).to(device="cuda")
#
#
#
# viewmat = torch.tensor([[ 0.90824819, -0.19012550,  0.37274322,  0.00000000],
#         [ 0.23217703,  0.97008431, -0.07092445,  0.00000000],
#         [-0.34810778,  0.15095940,  0.92522007,  0.00000000],
#         [ 0.10334630, -2.25264215,  2.61506438,  1.00000000]]).to(device="cuda")



viewmat = torch.load("/home/paul/vision/PGSR/world_view_transform.pth", weights_only=True).to(device="cuda")

viewmat[:3, :3] = viewmat[:3, :3].T
viewmat[:3, 3] = viewmat[3, :3]
viewmat[3, :3] = 0

# viewmat = viewmat.T

# print(viewmat)
#
# input_all_map = compute_all_map_with_test(means, rotation, scaling, viewmat)
#
# input_all_map_gt = torch.load(f"/home/paul/vision/PGSR/input_all_map.pth", weights_only=True).to(device="cuda")
#
# input_all_map = compute_all_map(means, rotation, scaling, viewmat)
# check(input_all_map, input_all_map_gt, "input_all_map without test")
#
K = torch.eye(3).to(device="cuda")
K[0, 0] = 630.4090
K[1, 1] = 629.9240
K[0, 2] = width / 2
K[1, 2] = height / 2
Ks = K.unsqueeze(0)

colors = torch.zeros_like(means)

scales = torch.exp(scaling)  # [N, 3]
opacities = torch.sigmoid(opacity)  # [N,]

render_colors, render_alphas, info = rasterization(
    means=means,
    quats=rotation,
    scales=scales,
    opacities=opacities.squeeze(),
    colors=colors,
    viewmats=viewmat.unsqueeze(0),  # [C, 4, 4]
    Ks=Ks,  # [C, 3, 3]
    width=width,
    height=height,
    packed=False,
    absgrad=False,
    sparse_grad=False,
    rasterize_mode='RGB+ED',
    distributed=False,
    camera_model="pinhole",
)

pgsr_rendered_normal = torch.load("/home/paul/vision/PGSR/rendered_normal.pth", weights_only=True).to(device="cuda")
pgsr_rendered_alpha = torch.load("/home/paul/vision/PGSR/rendered_alpha.pth", weights_only=True).to(device="cuda")
pgsr_rendered_distance = torch.load("/home/paul/vision/PGSR/rendered_distance.pth", weights_only=True).to(device="cuda")
pgsr_plane_depth = torch.load("/home/paul/vision/PGSR/plane_depth.pth", weights_only=True).to(device="cuda")
pgsr_depth_normal = torch.load("/home/paul/vision/PGSR/depth_normal.pth", weights_only=True).to(device="cuda")

tifffile.imwrite("original_gsplat_alpha.tiff", render_alphas[0].detach().cpu().numpy())
tifffile.imwrite("reproduce_pgsr_normals.tiff", pgsr_rendered_normal.detach().cpu().numpy())
tifffile.imwrite("reproduce_gsplat_normals.tiff", info['render_normals'][0].detach().cpu().numpy())
tifffile.imwrite("reproduce_gsplat_depths.tiff", info['render_depths'][0].detach().cpu().numpy())
tifffile.imwrite("reproduce_depth_normal.tiff", info['depth_normals'][0].detach().cpu().numpy())
tifffile.imwrite("reproduce_pgsr_depth.tiff", pgsr_plane_depth.detach().cpu().numpy())
tifffile.imwrite("reproduce_pgsr_depth_normal.tiff", pgsr_depth_normal.detach().cpu().numpy())

normal_diff = (pgsr_rendered_normal - info['render_normals'][0].permute(2,0,1))
print("Normal diff: ", normal_diff.abs().max(), normal_diff.abs().mean())

alpha_diff = (pgsr_rendered_alpha - render_alphas[0].permute(2,0,1))
print("Alpha diff: ", alpha_diff.abs().max(), alpha_diff.abs().mean())

depth_diff = (pgsr_plane_depth - info['render_depths'][0].permute(2,0,1))
print("Depth diff: ", depth_diff.abs().max(), depth_diff.abs().mean())

depth_normal_diff = (pgsr_depth_normal - info['depth_normals'][0].permute(2,0,1))
print("Depth normal diff: ", depth_normal_diff.abs().max(), depth_normal_diff.abs().mean())
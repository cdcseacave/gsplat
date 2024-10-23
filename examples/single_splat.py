import enum
import math

import torch

from gsplat.rendering import rasterization, rasterization_radegs, rasterization_rade_inria_wrapper

import tifffile
import tyro

test_grad = False

class Rasterization(enum.Enum):
    GS3D = "gs3d"
    GS2D = "gs2d"
    RADE = "rade"
    RADE_INRIA = "rade_inria"

def make_rotation_quat(angle):
    # Define the angle in radians
    angle = math.radians(angle)

    # Calculate the quaternion components
    qw = math.cos(angle / 2)
    qx = 0.0
    qy = math.sin(angle / 2)
    qz = 0.0

    # Return the quaternion
    return [qw, qx, qy, qz]

def make_rotation_matrix(angles):
    # Convert angles from degrees to radians
    angles = [math.radians(angle) for angle in angles]

    # Calculate rotation matrices for each axis
    cx, cy, cz = [math.cos(angle) for angle in angles]
    sx, sy, sz = [math.sin(angle) for angle in angles]

    # Rotation matrix around x-axis
    Rx = torch.tensor([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=torch.float32)

    # Rotation matrix around y-axis
    Ry = torch.tensor([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=torch.float32)

    # Rotation matrix around z-axis
    Rz = torch.tensor([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Combined rotation matrix
    R = torch.mm(Rz, torch.mm(Ry, Rx))

    return R

def look_at(eye, target, up=[0.0, 1.0, 0.0], device=None):
    up = torch.tensor(up, dtype=torch.float32).to(device)

    # Compute forward vector
    forward = target - eye
    forward = forward / torch.norm(forward)

    # Compute right vector
    right = torch.cross(up, forward)
    right = right / torch.norm(right)

    # Recompute up vector
    up = torch.cross(forward, right)

    # Create rotation matrix
    rotation_matrix = torch.stack([right, up, forward], dim=-1)

    return rotation_matrix

def generate_splats():
    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the single splat in the center of the scene
    means = torch.tensor([[2.0, 0.1, 5.0], [0.2, 2.1, 5.5]], dtype=torch.float32, requires_grad=test_grad).to(device)  # [N, 3]
    quats = torch.tensor([make_rotation_quat(-45), make_rotation_quat(30)], dtype=torch.float32, requires_grad=test_grad).to(device)  # [N, 4]
    scales = torch.tensor([[1.0, 0.6, 0.1], [0.8, 0.1, 0.7]], dtype=torch.float32, requires_grad=test_grad).to(device)  # [N, 3]
    opacities = torch.tensor([1.0, 0.9], dtype=torch.float32, requires_grad=test_grad).to(device)  # [N]
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=test_grad).to(device)  # [N, 3]

    # Define the camera parameters
    C = torch.tensor([0.0, 0.5, 10.0], dtype=torch.float32).to(device)  # [C, 3]
    R = look_at(C, means[0][0], device=device)  # [C, 3, 3]
    # Compose view matrix as the transform that contains R and C
    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(device)  # [C, 4, 4]
    viewmats[0, :3, :3] = R
    viewmats[0, :3, 3] = -torch.matmul(R, C)

    # Image dimensions
    if test_grad:
        width = 10
        height = 10
    else:
        width = 300
        height = 200

    Ks = torch.tensor([[[float(width), 0.0, float(width)/2.0],
                        [0.0, float(width), float(height)/2.0],
                        [0.0, 0.0, 1.0]]],
                        dtype=torch.float32).to(device)  # [C, 3, 3]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(means), 1.6e-4),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("colors", torch.nn.Parameter(colors), 2.5e-3),
    ]
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    return splats, viewmats, Ks, width, height

def rasterize_splats(splats, viewmats, Ks, width, height, rasterization_type: Rasterization):
    render_colors = None
    render_alphas = None
    render_depths = None
    render_normals = None

    if rasterization_type == Rasterization.GS3D:
        # Call the rasterization function
        render_colors, render_alphas, meta = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            eps2d=0.3,
            sh_degree=None,
            packed=False,
            tile_size=16,
            backgrounds=None,
            render_mode="RGB+ED",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            channel_chunk=32,
            distributed=False,
            covars=None,
            render_geo=True,
        )
        render_depths = meta["render_depths"]
        render_normals = meta["render_normals"]
    elif rasterization_type == Rasterization.GS2D:
        raise NotImplementedError("GS2D rasterization is not implemented yet.")
    elif rasterization_type == Rasterization.RADE:
        render_colors, render_alphas, render_depths, render_normals, meta = rasterization_radegs(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            eps2d=0.3,
            sh_degree=None,
            packed=False,
            tile_size=16,
            backgrounds=None,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            channel_chunk=32,
            distributed=False,
            ortho=False,
            covars=None,
        )
    elif rasterization_type == Rasterization.RADE_INRIA:
        (render_colors, render_alphas), meta = rasterization_rade_inria_wrapper(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=0.0,
            eps2d=0.3,
            sh_degree=None,
            packed=False,
            tile_size=16,
            backgrounds=None,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            channel_chunk=32,
            distributed=False,
            ortho=False,
            covars=None,
        )
        render_depths = meta["depth"]
        render_normals = meta["normals_rend"]

    return render_colors, render_alphas, render_depths, render_normals

def run(rasterization_type: Rasterization):
    # Generate splats
    splats, viewmats, Ks, width, height = generate_splats()

    if test_grad:
        # Check gradients correctness using torch.autograd.gradcheck
        gradcheck_inputs = tuple([splats, viewmats, Ks])
        gradcheck_fn = lambda *inputs: rasterize_splats(
            splats=inputs[0],
            viewmats=inputs[1],
            Ks=inputs[2],
            width=width,
            height=height,
            rasterization_type=rasterization_type,
        )[0]
        valid = torch.autograd.gradcheck(gradcheck_fn, gradcheck_inputs, eps=1e-4, atol=1e-2)
        print("Gradient check: ", valid)

    # Rasterize splats
    render_colors, render_alphas, render_depths, render_normals = rasterize_splats(splats, viewmats, Ks, width, height, rasterization_type)

    # Save the rendered image
    if render_colors is not None:
        tifffile.imwrite("render_colors_test.tiff", render_colors[0].detach().cpu().numpy())
    if render_alphas is not None:
        tifffile.imwrite("render_alphas_test.tiff", render_alphas[0].detach().cpu().numpy())
    if render_depths is not None:
        tifffile.imwrite("render_depths_test.tiff", render_depths[0].detach().cpu().numpy())
    if render_normals is not None:
        tifffile.imwrite("render_normals_test.tiff", render_normals[0].detach().cpu().numpy())

if __name__ == "__main__":
    tyro.cli(run)
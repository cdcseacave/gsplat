import math
import torch
from torch import Tensor
from gsplat.cuda._wrapper import _FullyFusedProjectionRade, _make_lazy_cuda_func
from torch.autograd import Function
from torch.autograd import gradcheck
from typing import Tuple

from gsplat.rendering import rasterization_rade_inria_wrapper

torch.use_deterministic_algorithms(True)
import sys

class _FullyFusedProjectionRadePlayground(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        viewmats: Tensor,
        opacities: Tensor,
        Ks: Tensor,
        image_width = 300,
        image_height = 200,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        (radii, means2d, depths, conics, compensations,
         camera_plane, normals, ray_plane, coef, invraycov3d, ts) = _make_lazy_cuda_func(
            "fully_fused_projection_fwd_radegs"
        )(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            0,
            0.01,
            100,
            0,
            False,
            False,
        )
        ctx.save_for_backward(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            radii,
            conics,
            depths,
            camera_plane,
            normals,
            ray_plane,
            ts
        )
        ctx.width = width
        ctx.height = height

        means2d = torch.nan_to_num(means2d, nan=0.0, posinf=0.0, neginf=0.0)
        conics = torch.nan_to_num(conics, nan=0.0, posinf=0.0, neginf=0.0)

        print("------------------------------------")
        print("FORWARD IN")
        print("Opacities", opacities)
        print("FORWARD OUT")
        print("Means2d", means2d)
        print("Conics", conics)

        return means2d, conics#, depths#, camera_plane#, normals, ray_plane, ts

    @staticmethod
    def backward(ctx, v_means2d, v_conics):#, v_depths):#, v_camera_plane): #, v_normals, v_ray_plane, v_ts):

        #print("BACKWARD")
        (
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            radii,
            conics,
            depths,
            camera_plane,
            normals,
            ray_plane,
            ts
        ) = ctx.saved_tensors

#        print("------------------------------------")
#
#        print("Input grads of sizes")
#        print("   ", v_means2d.shape)
#        print("   ", v_conics.shape)
#        print("   ", v_depths.shape)
        print("Input grads")
        print("   ", v_means2d)
        print("   ", v_conics)

        width = ctx.width
        height = ctx.height
        grads = _make_lazy_cuda_func(
            "fully_fused_projection_bwd_radegs"
        )(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            radii,
            conics,
            v_means2d.contiguous(),
            torch.zeros_like(depths), # v_depths.contiguous(),
            v_conics.contiguous(),
            torch.zeros_like(camera_plane), # v_camera_plane.contiguous(),
            torch.zeros_like(ray_plane),  # v_ray_plane.contiguous(),
            torch.zeros_like(normals),  # v_normals.contiguous(),
            torch.zeros_like(ts),  # v_ts.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
        )

        v_means, v_quats, v_scales, v_viewmats = grads

        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_quats = None
        if not ctx.needs_input_grad[2]:
            v_scales = None
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        print("Returning grads")
        print("   ", v_means)
        print("   ", v_quats)
        print("   ", v_scales)
        print("   ", v_viewmats)

        # v_quats[0, 0] = 1234.0

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CustomFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save context for backward pass
        ctx.save_for_backward(a, b)
        # Perform the forward computation
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        a, b = ctx.saved_tensors
        # Compute the gradient of the input
        grad_a = grad_output
        grad_b = grad_output
        return grad_a, grad_b


def make_rotation_quat(angle):
    # Define the angle in radians
    angle = math.radians(angle)

    # Calculate the quaternion components
    qw = math.cos(angle / 2)
    qx = 0.0
    qy = math.sin(angle / 2)
    qz = 0.0

    # Create the quaternion tensor
    quaternion = torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32)

    return quaternion

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

quats = make_rotation_quat(45)

# Define the single splat in the center of the scene
means = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to(device)  # [N, 3]
quats = torch.tensor(quats, dtype=torch.float32).to(device)  # [N, 4]
scales = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32).to(device)  # [N, 3]
opacities = torch.tensor([1.0], dtype=torch.float32).to(device)  # [N]
colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(device)  # [N, 3]

# Define the camera parameters
viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(device)  # [C, 4, 4]
viewmats[0, 2, 3] = 5.0  # Move the camera back along the z-axis

Ks = torch.tensor([[[300.0, 0.0, 150.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]]], dtype=torch.float32).to(device)  # [C, 3, 3]

# Image dimensions
width = 300
height = 200



render_colors = None
render_alphas = None
render_depths = None
render_normals = None

double_means = means.double().requires_grad_()
double_quats = quats.double()#.requires_grad_()
double_scales = scales.double()#.requires_grad_()
double_viewmats = viewmats.double()#.requires_grad_()
double_opacities = torch.ones(means.shape[0], dtype=torch.float64).to(device=device)#.requires_grad_()
double_Ks = Ks.double()#.requires_grad_()

print(double_means.requires_grad)
print(double_quats.requires_grad)
print(double_scales.requires_grad)
print(double_viewmats.requires_grad)
print(double_Ks.requires_grad)

test = CustomFunction.apply
gradcheck_result = gradcheck(test, (double_means,double_means), eps=1e-6, atol=1e-4)
print("GRADCHECK 1 (test)", gradcheck_result)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

sys.exit(1)



test = _FullyFusedProjectionRadePlayground.apply
input = (double_means, double_quats, double_scales, double_viewmats, double_opacities, double_Ks)

outputs = test(*input)
#print(outputs)


#gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4, nondet_tol=1e-5)
#print("GRADCHECK 2", gradcheck_result)
#
#if not gradcheck_result:
#    raise Exception("Gradcheck failed")


n_start = 82
n_end = 83

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
print(outputs)

gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4)
print("GRADCHECK", gradcheck_result)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

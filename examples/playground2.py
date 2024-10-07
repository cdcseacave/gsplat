import torch
from tensorboard.compat.tensorflow_stub.dtypes import float64
from torch.autograd import Function, gradcheck

torch.autograd.set_detect_anomaly(True)

def quat_to_rotmat(quat):
    a, b, c, d = quat
    # normalize
    norm = torch.sqrt(a**2 + b**2 + c**2 + d**2)
    x = a / norm
    y = b / norm
    z = c / norm
    w = d / norm
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return torch.tensor([
        [1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)],
        [2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)],
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)]
    ], dtype=quat.dtype).to(device=quat.device)


def quat_scale_to_covar_preci(quat, scale, with_covar=None, with_preci=None):
    R = quat_to_rotmat(quat)
    covar, preci = None, None
    if with_covar is not None:
        # C = R * S * S * Rt
        S = torch.diag(scale)
        M = torch.matmul(R, S)
        covar = torch.matmul(M, M.t())
    if with_preci is not None:
        # P = R * S^-1 * S^-1 * Rt
        S_inv = torch.diag(1.0 / scale)
        M = torch.matmul(R, S_inv)
        preci = torch.matmul(M, M.t())
    return covar, preci



def covar_world_to_cam(R, covar):
    covar_c = torch.matmul(R, torch.matmul(covar, R.t()))
    return covar_c


class ComputeOpacity(Function):
    @staticmethod
    def forward(ctx, quat, R, scale):

        covar, _ = quat_scale_to_covar_preci(quat, scale, with_covar=True)
        cov = covar_world_to_cam(R, covar)

        kernel_size = 1

        det_0 = torch.max(torch.Tensor([1e-6]).to('cuda').double(), cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
        det_1 = torch.max(torch.Tensor([1e-6]).to('cuda').double(), (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);

        ctx.save_for_backward(quat, R, covar, cov, det_0, det_1)

        coef = torch.sqrt(det_0 / (det_1 + 1e-6) + 1e-6)
        if det_0 <= 1e-6 or det_1 <= 1e-6:
            coef = torch.tensor(0.0, dtype=torch.float64)
        return coef


    @staticmethod
    def backward(ctx, grad_output):
        quat, R, covar, cov, det_0, det_1 = ctx.saved_tensors

        # Compute gradients with respect to quat, R, and scale
        grad_quat = torch.zeros_like(quat)
        grad_R = torch.zeros_like(R)
        grad_scale = torch.zeros_like(covar.diag())

        # Compute the gradient of coef with respect to det_0 and det_1
        grad_det_0 = grad_output * (0.5 / torch.sqrt(det_0 / (det_1 + 1e-6) + 1e-6)) * (1 / (det_1 + 1e-6))
        grad_det_1 = grad_output * (0.5 / torch.sqrt(det_0 / (det_1 + 1e-6) + 1e-6)) * (-det_0 / ((det_1 + 1e-6) ** 2))

        # Compute the gradient of det_0 and det_1 with respect to cov
        grad_cov = torch.zeros_like(cov)
        grad_cov[0, 0] = grad_det_0 * cov[1, 1] - grad_det_1 * (cov[1, 1] + 1)
        grad_cov[1, 1] = grad_det_0 * cov[0, 0] - grad_det_1 * (cov[0, 0] + 1)
        grad_cov[0, 1] = grad_cov[1, 0] = -grad_det_0 * cov[0, 1] + grad_det_1 * cov[0, 1]

        # Compute the gradient of cov with respect to R and covar
        grad_R += torch.matmul(grad_cov, covar.t())
        grad_covar = torch.matmul(R.t(), torch.matmul(grad_cov, R))

        # Compute the gradient of covar with respect to quat and scale
        grad_quat += torch.autograd.grad(outputs=covar, inputs=quat, grad_outputs=grad_covar, retain_graph=True)[0]
        grad_scale += torch.autograd.grad(outputs=covar, inputs=covar.diag(), grad_outputs=grad_covar.diag(), retain_graph=True)[0]

        return grad_quat, grad_R, grad_scale




n_start = 0
n_end = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
means = torch.load("means.pt").to(device=device)[n_start:n_end].double()
quats = torch.load("quats.pt").to(device=device)[n_start:n_end].double()
scales = torch.load("scales.pt").to(device=device)[n_start:n_end].double()
viewmats = torch.load("viewmats.pt").to(device=device).double()
Ks = torch.load("Ks.pt").to(device=device).double()


quat = quats[0]
R = viewmats[0, :3, :3]
scale = scales[0]

quat.

# Create input tensor
input = (quat, R, scale)

# Perform gradcheck
test = ComputeOpacity.apply
gradcheck_result = gradcheck(test, input, eps=1e-6, atol=1e-4)
print("GRADCHECK", gradcheck_result)


output = ComputeOpacity.apply(input)
print(output)
# output[0].backward()
output[1].backward()


import torch
from torch.autograd import Function
from torch.autograd import gradcheck

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

test = CustomFunction.apply

input1 = torch.randn(2, 2, requires_grad=True).double()
input2 = torch.randn(2, 2, requires_grad=True).double()

gradcheck_result = gradcheck(test, (input1, input2), eps=1e-6, atol=1e-4)

if not gradcheck_result:
    raise Exception("Gradcheck failed")

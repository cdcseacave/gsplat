import torch
from torch.autograd import gradcheck

from gsplat import rasterize_to_pixels
from gsplat.cuda._wrapper import _RasterizeToPixels

inputs = torch.load("args.pt", weights_only=True)
inputs = list(inputs)

print("Loaded args.pt. Means2d shape:", inputs[0].shape)


for idx in range(len(inputs)):
    if torch.is_tensor(inputs[idx]):
        inputs[idx] = inputs[idx].cuda().requires_grad_(False)
    if torch.is_tensor(inputs[idx]) and inputs[idx].dtype == torch.float32:
        inputs[idx] = inputs[idx].double()

for idx in range(1,2):
    inputs[idx].requires_grad_(True)

for idx in range(len(inputs)):
    if torch.is_tensor(inputs[idx]):
        print("Input", idx, "shape:", inputs[idx].shape)
    else:
        print("Input", idx, "is not a tensor", inputs[idx])


print("Evaluating")
out = _RasterizeToPixels.apply(*inputs)
print("Done evaluating")
loss = out[0].sum()
print("loss is", loss)
loss.backward()

print("Done example loss")


result = gradcheck(_RasterizeToPixels.apply, inputs,
                   eps=1e-8, atol=1e-6, rtol=1e-6, fast_mode=True, nondet_tol=1e-8)

print("Gradcheck result:", result)
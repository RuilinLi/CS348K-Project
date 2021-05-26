import sys
import torch
import conv_cuda
import torch.nn.functional as F

batch_size = 128

if len(sys.argv) < 2:
    blockIdx = 2
else:
    blockIdx = int(sys.argv[1])
    assert blockIdx >= 1 and blockIdx <= 4
    print(blockIdx)

device=torch.device("cuda")

all_size = [(batch_size, 16, 16, 64), (batch_size, 8, 8, 128), (batch_size, 4, 4, 256), (batch_size, 2, 2, 512)]

ind = blockIdx
use_size = all_size[ind - 1]

Activation = torch.rand(use_size, dtype=torch.float16, device=device)

W1 = torch.randn((use_size[-1], use_size[-1]//16), dtype=torch.float16, device=device)
b1 = torch.randn((use_size[-1]//16), dtype=torch.float16, device=device)
W2 = torch.randn((use_size[-1]//16, use_size[-1]), dtype=torch.float16, device=device)
b2 = torch.randn((use_size[-1]), dtype=torch.float16, device=device)
fixup_bias = torch.randn(1, dtype=torch.float16, device=device)
result = torch.zeros(use_size, dtype=torch.float16, device=device)
if blockIdx == 1:
    conv_cuda.SE1(Activation, W1, b1, W2, b2, fixup_bias, result)
elif blockIdx == 2:
    conv_cuda.SE2(Activation, W1, b1, W2, b2, fixup_bias,result)
elif blockIdx == 3:
    conv_cuda.SE3(Activation, W1, b1, W2, b2, fixup_bias,result)
elif blockIdx == 4:
    fixup_bias = torch.zeros(1, dtype=torch.float16, device=device)
    conv_cuda.SE4(Activation, W1, b1, W2, b2, fixup_bias,result)



reduced = torch.sum(torch.sum(Activation, 1), 1) / (use_size[1] * use_size[2])
ref = F.relu(torch.matmul(reduced, W1) + b1)


ref = torch.matmul(ref, W2) + b2
ref = torch.sigmoid(ref)
ref = Activation * ref.unsqueeze(1).unsqueeze(1) + fixup_bias


print((result - ref).max())
print((result - ref).min())
print((result - ref).abs().mean())


rel_err = (result - ref).abs()/(1 + ref.abs())
print(rel_err.max())
print(rel_err.mean())
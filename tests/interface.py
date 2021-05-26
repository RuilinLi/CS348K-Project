import sys
import torch
import conv_cuda
import torch.nn.functional as F

device=torch.device("cuda")

batch_size = 128
if len(sys.argv) < 2:
    blockIdx = 2
else:
    blockIdx = int(sys.argv[1])
    assert blockIdx >= 1 and blockIdx <= 4


activation_sizes = [(batch_size, 16, 16, 64), (batch_size, 16, 16, 64), (batch_size, 8, 8, 128), (batch_size, 4, 4, 256)]
out_sizes = [(batch_size, 16, 16, 64), (batch_size, 8, 8, 128), (batch_size, 4, 4, 256), (batch_size, 2, 2, 512)]

activation_use = activation_sizes[blockIdx - 1]
out_use = out_sizes[blockIdx - 1]
in_channel = activation_use[3]
out_channel = out_use[3]
stride = 1 if blockIdx == 1 else 2
out_plane_size = out_use[1]


activations = torch.rand(activation_use, dtype=torch.float16, device=device) 
filter1 = torch.randn((out_channel, 3,3, in_channel), dtype=torch.float16, device=device)
filter2 = torch.randn((out_channel, 3,3, out_channel), dtype=torch.float16, device=device)

bias1b = torch.randn((1), dtype=torch.float16, device=device)
bias2a = torch.randn((1), dtype=torch.float16, device=device)
bias2b = torch.randn((1), dtype=torch.float16, device=device)
fescale = torch.randn((1), dtype=torch.float16, device=device)

Param1 = conv_cuda.NetParameters(filter1, filter2, bias1b, bias2a, bias2b, fescale)
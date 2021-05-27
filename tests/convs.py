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


activations = torch.empty(activation_use, dtype=torch.float16, device=device)
torch.bernoulli(torch.ones_like(activations) * 0.5, out = activations)
activations = activations - 0.5
filter1 = torch.empty((out_channel, 3,3, in_channel), dtype=torch.float16, device=device)
filter2 = torch.empty((out_channel, 3,3, out_channel), dtype=torch.float16, device=device)
torch.bernoulli(torch.ones_like(filter1) * 0.5, out = filter1)
torch.bernoulli(torch.ones_like(filter2) * 0.5, out = filter2)
filter1 = filter1 - 0.5
filter2 = filter2 - 0.5

bias1b = torch.empty((1), dtype=torch.float16, device=device)
bias2a = torch.empty((1), dtype=torch.float16, device=device)
bias2b = torch.empty((1), dtype=torch.float16, device=device)
fescale = torch.empty((1), dtype=torch.float16, device=device)

torch.bernoulli(torch.ones_like(bias1b) * 0.5, out = bias1b)
torch.bernoulli(torch.ones_like(bias2a) * 0.5, out = bias2a)
torch.bernoulli(torch.ones_like(bias2b) * 0.5, out = bias2b)
torch.bernoulli(torch.ones_like(fescale) * 0.5, out = fescale)


out1 = torch.zeros((batch_size, out_plane_size, out_plane_size, out_channel), dtype=torch.float16, device=device)
result = torch.zeros((batch_size, out_plane_size, out_plane_size, out_channel), dtype=torch.float16, device=device)

if blockIdx == 1:    
    conv_cuda.Conv2Block1(activations, filter1, out1, filter2, result, bias1b, bias2a, bias2b, fescale)
elif blockIdx == 2:
    conv_cuda.Conv2Block2(activations, filter1, out1, filter2, result, bias1b, bias2a, bias2b, fescale)
elif blockIdx == 3:
    conv_cuda.Conv2Block3(activations, filter1, out1, filter2, result, bias1b, bias2a, bias2b, fescale)
elif blockIdx == 4:
    conv_cuda.Conv2Block4(activations, filter1, out1, filter2, result, bias1b, bias2a, bias2b, fescale)


reference = torch.relu(F.conv2d(activations.permute(0,3,1,2), filter1.permute(0,3,1,2), padding=1, stride=stride) + bias1b) + bias2a
reference = F.conv2d(reference, filter2.permute(0,3,1,2), padding=1, stride=1) * fescale + bias2b
reference = reference.permute(0, 2, 3, 1)

rel_err = (result - reference).abs()/(1 + reference.abs())
print(rel_err.max())
print(rel_err.mean())
# linear_ind = torch.argmax(rel_err).item()
# n = linear_ind // (out_use[1] * out_use[2] * out_use[3])
# linear_ind = linear_ind % (out_use[1] * out_use[2] * out_use[3])
# h = linear_ind// (out_use[2] * out_use[3])
# linear_ind = linear_ind % (out_use[2] * out_use[3])
# w = linear_ind // out_use[3]
# c = linear_ind % out_use[3]
# print((n,h,w,c))
# print(reference[n,h,w,c])
# print(result[n,h,w,c])

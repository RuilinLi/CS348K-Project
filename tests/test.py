import torch
import conv_cuda
import math
import time
import torch.nn.functional as F



device=torch.device("cuda")

# Test 2 back-to-back conv
# a = torch.ones((128, 16, 16, 64), dtype=torch.float16, device=device)/100
# fil = torch.ones((64, 3,3, 64), dtype=torch.float16, device=device)/100
# fil2 = torch.ones((64, 3,3, 64), dtype=torch.float16, device=device)/100

# d2 = torch.ones((1), dtype=torch.float16, device=device) * 0.123
# d3 = torch.ones((1), dtype=torch.float16, device=device) * 0.234
# d4 = torch.ones((1), dtype=torch.float16, device=device) + 1.0 
# d5 = torch.ones((1), dtype=torch.float16, device=device) * 1.5


# out1 = torch.zeros((128, 16, 16, 64), dtype=torch.float16, device=device)
# out2 = torch.zeros((128, 16, 16, 64), dtype=torch.float16, device=device)
# my_obj = conv_cuda.Conv128x16x16x64NHWC3x3x64NHWC(fil, out1, fil2, out2, d2, d3, d4, d5)
# my_obj.run(a)
# out2 = out2 + 1
# print(out2[0,0,0])


############# Test SE
if False:
    all_size = [(128, 16, 16, 64), (128, 8, 8, 128)]
    ind = 1
    use_size = all_size[ind]
    Activation = torch.rand(use_size, dtype=torch.float16, device=device)
    # Squeezed = torch.zeros((128, 64), dtype=torch.float16, device=device)
    W1 = torch.randn((use_size[-1], use_size[-1]//16), dtype=torch.float16, device=device)
    b1 = torch.randn((use_size[-1]//16), dtype=torch.float16, device=device)
    W2 = torch.randn((use_size[-1]//16, use_size[-1]), dtype=torch.float16, device=device)
    b2 = torch.zeros((use_size[-1]), dtype=torch.float16, device=device)
    result = torch.zeros(use_size,dtype=torch.float16, device=device)
    obj = conv_cuda.MyCudaSE(Activation, W1, b1, W2, b2, result)
    obj.run()
    # obj.run()
    # obj.run()
    # print(result.max())
    # print(result.min())
    # obj = conv_cuda.ResnetSE(Activation, Squeezed, W1, b1, W2, b2, result)
    # obj.run()
    reduced = torch.sum(torch.sum(Activation, 1), 1) / (use_size[1] * use_size[2])
    ref = F.relu(torch.matmul(reduced, W1) + b1)
    ref = torch.matmul(ref, W2) + b2
    ref = torch.sigmoid(ref)
    ref = Activation * ref.unsqueeze(1).unsqueeze(1)

    print((result - ref).max())
    print((result - ref).min())
    print((result - ref).abs().mean())

################## Larger SE
if True:
    print(0)
# #all_size = [(128, 4, 4, 256), (128, 2, 2, 512)]
# all_size = [(1024, 4, 4, 256), (1024, 2, 2, 512)]

# ind = 0
# use_size = all_size[ind]
# #Activation = torch.ones(use_size, dtype=torch.float16, device=device) * 0.5
# Activation = torch.rand(use_size, dtype=torch.float16, device=device)

# W1 = torch.randn((use_size[-1], use_size[-1]//16), dtype=torch.float16, device=device)
# b1 = torch.randn((use_size[-1]//16), dtype=torch.float16, device=device)
# W2 = torch.randn((use_size[-1]//16, use_size[-1]), dtype=torch.float16, device=device)
# b2 = torch.randn((use_size[-1]), dtype=torch.float16, device=device)
# result = torch.zeros(use_size,dtype=torch.float16, device=device)
# obj = conv_cuda.MyCudaSE2(Activation, W1, b1, W2, b2, result)
# obj.run()


# reduced = torch.sum(torch.sum(Activation, 1), 1) / (use_size[1] * use_size[2])
# ref = F.relu(torch.matmul(reduced, W1) + b1)



# ref = torch.matmul(ref, W2) + b2
# ref = torch.sigmoid(ref)
# ref = Activation * ref.unsqueeze(1).unsqueeze(1)

# print((result - ref).max())
# print((result - ref).min())
# print((result - ref).abs().mean())

########### TEST Stem (GEMM + relu)
all_size = [(128, 16, 16, 64), (1024, 16, 16, 48)]
use_size = all_size[0]
Activation = torch.rand(use_size, dtype=torch.float16, device=device) * 0.2
Filter = torch.randn((64, 64), dtype=torch.float16, device=device)

# Filtertensor = torch.unsqueeze(torch.unsqueeze(Filter, 2), 3)
# ref = F.relu(F.conv2d(Activation.permute(0, 3, 1, 2) , Filtertensor)).permute(0, 2, 3, 1)

result = torch.zeros((use_size[0], 16, 16, 64), dtype=torch.float16, device=device)
conv_cuda.StemOp48_64(Activation, Filter, result)

act2 = Activation.view(128*16*16, 64)
# act2 = F.relu(torch.matmul(act2, torch.transpose(Filter, 0, 1)))
act2 = torch.matmul(act2, Filter)
print((result.view(128*16*16, 64) -act2).max())

# res_local = result[512:(5*128)]
# ref_local = ref[512:(5*128)]
# torch.argmax(res_local - ref_local)
# torch.argmin(ref_local - res_local)
# print((res_local - ref_local).max())
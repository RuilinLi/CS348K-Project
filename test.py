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

Activation = torch.ones((128, 16, 16, 64), dtype=torch.float16, device=device) * 0.1
# Squeezed = torch.zeros((128, 64), dtype=torch.float16, device=device)
W1 = torch.ones((64, 4), dtype=torch.float16, device=device)* 0.1
b1 = torch.ones((4), dtype=torch.float16, device=device)* 0.1
W2 = torch.ones((4, 64), dtype=torch.float16, device=device)* 0.1
b2 = torch.zeros((64), dtype=torch.float16, device=device) * 0.1
result = torch.zeros((128, 16, 16, 64),dtype=torch.float16, device=device)
obj = conv_cuda.MyCudaSE(Activation, W1, b1, W2, b2, result)
obj.run()
print(result.max())
print(result.min())
# obj = conv_cuda.ResnetSE(Activation, Squeezed, W1, b1, W2, b2, result)
# obj.run()


reduced = torch.sum(torch.sum(Activation, 1), 1)
ref = Activation * reduced.unsqueeze(1).unsqueeze(1)
# ref = F.relu(torch.matmul(reduced, W1) + b1)
# ref = torch.matmul(ref, W2) + b2
# ref = torch.sigmoid(ref)
# ref = Activation * ref.unsqueeze(1).unsqueeze(1) + 0.1


# out3 = torch.zeros((128, 64), dtype=torch.float16, device=device)
# obj2 = conv_cuda.MyResnetSE(out2, out3)
# obj2.reduce()

# ref = out2.sum(1).sum(1)


# ref = F.conv2d(a.permute((0,3,1,2)), fil.permute((0, 3, 1, 2)), padding=1).permute(0, 2, 3, 1)

# print((ref - c).max())
# print((ref - c).min())
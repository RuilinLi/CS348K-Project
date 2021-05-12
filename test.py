import torch
import conv_cuda
import math
import time
import torch.nn.functional as F



device=torch.device("cuda")

a = torch.ones((128, 16, 16, 64), dtype=torch.float16, device=device)/100
fil = torch.ones((64, 3,3, 64), dtype=torch.float16, device=device)/100
fil2 = torch.ones((64, 3,3, 64), dtype=torch.float16, device=device)/100

d1 = torch.ones((1), dtype=torch.float16, device=device)
d2 = torch.ones((1), dtype=torch.float16, device=device) * 0.123
d3 = torch.ones((1), dtype=torch.float16, device=device) * 0.234
d4 = torch.ones((1), dtype=torch.float16, device=device)
d5 = torch.ones((1), dtype=torch.float16, device=device)


out1 = torch.zeros((128, 16, 16, 64), dtype=torch.float16, device=device)
out2 = torch.zeros((128, 16, 16, 64), dtype=torch.float16, device=device)
my_obj = conv_cuda.Conv128x16x16x64NHWC3x3x64NHWC(fil, out1, fil2, out2, d1, d2, d3, d4, d5)
my_obj.run(a)

out3 = torch.zeros((128, 64), dtype=torch.float16, device=device)
obj2 = conv_cuda.MyResnetSE(out2, out3)
obj2.reduce()

ref = out2.sum(1).sum(1)


# ref = F.conv2d(a.permute((0,3,1,2)), fil.permute((0, 3, 1, 2)), padding=1).permute(0, 2, 3, 1)

# print((ref - c).max())
# print((ref - c).min())
import torch
import conv_cuda
import math
import time
import torch.nn.functional as F



device=torch.device("cuda")

a = torch.randn((128, 16, 16, 64), dtype=torch.float16, device=device)
fil = torch.randn((64, 3,3, 64), dtype=torch.float16, device=device)
c = torch.zeros((128, 16, 16, 64), dtype=torch.float16, device=device)
my_obj = conv_cuda.Conv128x16x16x64NHWC3x3x64NHWC(a, fil, c)
my_obj.run()

ref = F.conv2d(a.permute((0,3,1,2)), fil.permute((0, 3, 1, 2)), padding=1).permute(0, 2, 3, 1)

print((ref - c).max())
print((ref - c).min())
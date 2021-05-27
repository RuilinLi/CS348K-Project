from resnet import se_resnet9_fixup, my_se_resnet9_fixup
import torch
import torch.autograd.profiler as profiler
import conv_cuda
import torch.nn.functional as F

device = torch.device("cuda")
# fp16_NHWC_weights = OrderedDict()
# with torch.cuda.amp.autocast():
net = torch.jit.script(se_resnet9_fixup(3, 64, -99).to(device).half())
# for key, val in net.state_dict().items():
#     if ("conv" in key) or ("downsample" in key) or ("stem" in key):
#         fp16_NHWC_weights[key] = val.to(torch.float16).permute(0, 2, 3, 1).contiguous()
#     else:
#         fp16_NHWC_weights[key] = val.to(torch.float16)
#     print(key)
#     print(val.size())
# for key, val in net.state_dict().items():
#     print(key)
#     print(val.size())


device = torch.device("cuda")
batch_size = 1024
mymodel = my_se_resnet9_fixup(net, batch_size)
input = torch.rand((batch_size, 3, 64, 64), dtype=torch.float16, device=device)
result = mymodel.RunNCHWInput(input)
torch.cuda.synchronize()
with torch.no_grad():
    ref = net(input)
ref = ref.permute(0, 2, 3, 1)
torch.cuda.synchronize()


print((ref - result).abs().max())
print((ref - result).abs().mean())
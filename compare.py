from resnet import se_resnet9_fixup
import torch
import torch.autograd.profiler as profiler
from collections import OrderedDict

device = torch.device("cuda")
fp16_NHWC_weights = OrderedDict()
# with torch.cuda.amp.autocast():
net = torch.jit.script(se_resnet9_fixup(3, 64, -99).to(device).half())
# for key, val in net.state_dict().items():
#     if ("conv" in key) or ("downsample" in key) or ("stem" in key):
#         fp16_NHWC_weights[key] = val.to(torch.float16).permute(0, 2, 3, 1).contiguous()
#     else:
#         fp16_NHWC_weights[key] = val.to(torch.float16)
#     print(key)
#     print(val.size())
for key, val in net.state_dict().items():
    print(val.dtype)
from resnet import se_resnet9_fixup
import torch
import torch.autograd.profiler as profiler

device = torch.device("cuda")
# Third parameter ngroups here not really used
net = torch.jit.script(se_resnet9_fixup(3, 64, -99).to(device))
# se_resnet9_fixup(3, 64, -99).to(device)

inp = torch.rand((2048, 3, 64, 64), dtype=torch.float16, device=device)

# Backward
# with torch.cuda.amp.autocast():
#     out = net(inp)
#     loss = torch.nn.MSELoss()
#     target = torch.rand_like(out, device=device) # fake target
#     output = loss(out, target)
#     output.backward(retain_graph=True)
#     with profiler.profile(record_shapes=True, use_cuda=True) as prof:
#         # out = net(inp)
#         output = loss(out, target)
#         output.backward()

# Forward
with torch.cuda.amp.autocast():
    # warm-up
    out = net(inp)
    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        out = net(inp)
        
print(prof.key_averages().table(sort_by='cuda_time_total'))

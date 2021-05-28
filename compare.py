from resnet import se_resnet9_fixup, my_se_resnet9_fixup
import torch
# import torch.autograd.profiler as profiler
import time
import conv_cuda
import torch.nn.functional as F

device = torch.device("cuda")
# with torch.cuda.amp.autocast():
# net = torch.jit.script(se_resnet9_fixup(3, 64, -99).to(device).half())
net = se_resnet9_fixup(3, 64, -99).to(device).half()

batch_size = 128

device = torch.device("cuda")
mymodel = my_se_resnet9_fixup(net, batch_size)

############# Verify that the results are right ######################3
if 0:
    input = torch.rand((batch_size, 3, 64, 64), dtype=torch.float16, device=device)
    result = mymodel.RunNCHWInput(input)
    with torch.no_grad():
        ref = net(input)
    ref = ref.permute(0, 2, 3, 1)


    print((ref - result).abs().max())
    print((ref - result).abs().mean())


########## Timing #########
large_batch_input =  torch.rand((batch_size * 100, 3, 64, 64), dtype=torch.float16, device=device)
with torch.no_grad():
    tic = time.perf_counter()
    for j in range(1000):
        i = j % 100
        data = large_batch_input[(i * batch_size):(i+1)*batch_size, ]
        #result = net(data)
        result = mymodel.RunNCHWInput(data)
        torch.cuda.synchronize()

    toc = time.perf_counter()
    print(f"Finishhing 128000 inferences in {toc - tic:0.4f} seconds")

# Optimizing Batch Inference and Training of ResNet on GPUs
### Summary 
In this project, I'm going to improve the runtime (latency) of batch inference and training of [this particular ResNet](https://github.com/RuilinLi/CS348K-Project/blob/2c17b63ea251ab943d4566e1b069c83e6c5330ae/resnet.py#L651) (code cpied from [here](https://github.com/shacklettbp/bps-nav/blob/master/bps_nav/rl/ddppo/policy/resnet.py)) on Volta and Ampere architectures. I expect most of the speedup will come from effecitve usage of Tensor cores as well as opertator fusion. My plan is to implement each resnet block (including FixUp and Squeeze-and-Excitation, for both forward and backward) using [CUTLASS](https://github.com/NVIDIA/cutlass). I believe its support for Tensor Core, GEMM epilogue, and fused convolution+convolution features will be helpful. The motivation comes from [this paper](https://arxiv.org/pdf/2103.07013.pdf).
- Inputs: 
  - For inference: a (N, 3, 64, 64) tensor (corresponding to N RGB images) or a (N, 1, 64, 64) tensor (corresponding to N depth map). Here N is the batch size, and is fixed for each GPU type. In the paper above N = 128 on a Tesla V100. In particular I will not change N to improve the throughput of the network.
  - For training: a (16 * N, 3, 64, 64) tensor (16 * N = N agents * 32 rollout length / 2 mini-batch), or a (16 * N, 1, 64, 64) tensor.
- Outputs:
  - For inference it is the output of the network: a tensor of size (N, 512, 2, 2).
  - For training the output are the gradient (or a gradient update?) of all parameters of this network evaluated at the minibatch. I don't think the intermediate values in the layers of the network is cached at inference time, since the GPU resources needs to be used for rendering. Therefore the forward values needs to be computed here, too.
- Task List:
  1. Profile the kernels used in Pytorch to identify the "hotspots" (pretty sure they are the convolutions based on the output of [this script](https://github.com/RuilinLi/CS348K-Project/blob/main/basic_profile.py), ran on my RTX 3070). The Pytorch implementation will be the baseline.
  2. Get familiar with CUTLASS. Implement an unoptimized ResNet block using CUTLASS.
  3. Make the implementation from the last step a Pytorch operator. I will follow [this tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html). Confirm that the results are correct.
  4. Optimize. The CUTLASS profiler might be handy for this step.
  
- Expected deliverables. An optimzied version of teh ResNet, available as a child class of nn.Module. A barplot that demonstrates the runtime reduction of the optimzied implementation against the baseline.

- Dependencies: most of them are mentioned above. To summarize, the dependencies are:
  1. The [started code](https://github.com/RuilinLi/CS348K-Project/blob/main/resnet.py) that defines the network to be optimized.
  2. [CUTLASS](https://github.com/NVIDIA/cutlass) and tutorials on how to use it.
  3. [Tutorial on how to write custom Pytorch operators](https://pytorch.org/tutorials/advanced/cpp_extension.html)

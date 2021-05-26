#pragma once

#include <torch/extension.h>

#include <iostream>
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "linear_combination_relu_fixup.h"
#include "cutlass/reduction/device/tensor_reduce.h"

// These are the same for all operations
using ElementAccumulator = float;  // Data type of accumulator
using ElementComputeEpilogue =
    cutlass::half_t;  // Data type of epilogue computation (alpha, beta)
using ElementConvInputA =
    cutlass::half_t;  // Data type of elements in input tensor
using ElementConvInputB =
    cutlass::half_t;  // Data type of elements in input tensor
using ElementConvOutput =
    cutlass::half_t;  // Data type of elements in output tensor

using LayoutConvInputA = cutlass::layout::TensorNHWC;
using LayoutConvInputB = cutlass::layout::TensorNHWC;
using LayoutConvOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or
// regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
static constexpr int NumStages = 3;

using EpilogueConvOp1 = cutlass::epilogue::thread::LinearCombinationReluFixUp<
    ElementConvOutput,
    128 / cutlass::sizeof_bits<ElementConvOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueConvOp2 = cutlass::epilogue::thread::LinearCombinationReluFixUp<
    ElementConvOutput,
    128 / cutlass::sizeof_bits<ElementConvOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue,
    false  // Don't apply relu
    >;

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
        cutlass::conv::IteratorAlgorithm::kAnalytic;

// This code section describes the size of MMA op, TensorCore instruction shape
using TensorCoreInsructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// These could be different for different blocks
// First block
// Input dimension (NHWC) (batch_size, 16, 16, 64) Conv 3x3, stride 1-> (batch_size, 16, 16, 64) Conv 3x3, stride 1 -> (batch_size, 16, 16, 64)
// GEMM M: batch_size * 16 * 16, N = 64, K = 3 * 3 * 64, same for both convs
using ThreadblockShape_Conv11 =
    cutlass::gemm::GemmShape<64, 64, 32>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape_Conv11 =
    cutlass::gemm::GemmShape<32, 32, 32>;  // Warp tile shape
using Conv2dFpropKernel11 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv11,
    WarpShape_Conv11,
    TensorCoreInsructionShape,
    EpilogueConvOp1,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;
using ImplicitGemm11 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel11>;

using ThreadblockShape_Conv12 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv12 = cutlass::gemm::GemmShape<32, 32, 32>;
using Conv2dFpropKernel12 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv12,
    WarpShape_Conv12,
    TensorCoreInsructionShape,
    EpilogueConvOp2,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;
using ImplicitGemm12 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel12>;


// Second block
// Input dimension (NHWC) (batch_size, 16, 16, 64) Conv 3x3, stride 2-> (batch_size, 8, 8, 128) Conv 3x3, stride 1 -> (batch_size, 8, 8, 128)
// GEMM1 M = batch_size * 8 * 8, N = 128, K = 3 * 3 * 64
// GEMM2 M = batch_size * 8 * 8, N = 128, K = 3 * 3 * 128
using ThreadblockShape_Conv21 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv21 = cutlass::gemm::GemmShape<32, 32, 32>;
using Conv2dFpropKernel21 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv21,
    WarpShape_Conv21,
    TensorCoreInsructionShape,
    EpilogueConvOp1,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;
using ImplicitGemm21 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel21>;

using ThreadblockShape_Conv22 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv22 = cutlass::gemm::GemmShape<32, 32, 32>;

using Conv2dFpropKernel22 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv22,
    WarpShape_Conv22,
    TensorCoreInsructionShape,
    EpilogueConvOp2,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;

using ImplicitGemm22 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel22>;
// Third block
// Input dimension (NHWC) (batch_size, 8, 8, 128) Conv 3x3, stride 2-> (batch_size, 4, 4, 256) Conv 3x3, stride 1 -> (batch_size, 4, 4, 256)
// GEMM1 M = batch_size * 4 * 4, N = 256, K = 3 * 3 * 128
// GEMM2 M = batch_size * 4 * 4, N = 256, K = 3 * 3 * 256
using ThreadblockShape_Conv31 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv31 = cutlass::gemm::GemmShape<32, 32, 32>;

using Conv2dFpropKernel31 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv31,
    WarpShape_Conv31,
    TensorCoreInsructionShape,
    EpilogueConvOp1,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;
using ImplicitGemm31 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel31>;

using ThreadblockShape_Conv32 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv32 = cutlass::gemm::GemmShape<32, 32, 32>;

using Conv2dFpropKernel32 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv32,
    WarpShape_Conv32,
    TensorCoreInsructionShape,
    EpilogueConvOp2,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;

using ImplicitGemm32 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel32>;
// Last block
// Input dimension (NHWC) (batch_size, 4, 4, 256) Conv 3x3, stride 2-> (batch_size, 2, 2, 512) Conv 3x3, stride 1 -> (batch_size, 2, 2, 512)
// GEMM1 M = batch_size * 2 * 2, N = 512, K = 3 * 3 * 256
// GEMM2 M = batch_size * 4 * 4, N = 256, K = 3 * 3 * 512
using ThreadblockShape_Conv41 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv41 = cutlass::gemm::GemmShape<32, 32, 32>;
using Conv2dFpropKernel41 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv41,
    WarpShape_Conv41,
    TensorCoreInsructionShape,
    EpilogueConvOp1,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;
using ImplicitGemm41 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel41>;

using ThreadblockShape_Conv42 = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape_Conv42 = cutlass::gemm::GemmShape<32, 32, 32>;
using Conv2dFpropKernel42 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementConvInputA,
    LayoutConvInputA,
    ElementConvInputB,
    LayoutConvInputB,
    ElementConvOutput,
    LayoutConvOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape_Conv42,
    WarpShape_Conv42,
    TensorCoreInsructionShape,
    EpilogueConvOp2,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm>::Kernel;

using ImplicitGemm42 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel42>;


template <int BlockIdx = 1, typename ImplicitGemm1 = ImplicitGemm11, typename ImplicitGemm2 = ImplicitGemm12>
void ConvBlock(torch::Tensor Activation,
                torch::Tensor Filter1,
                torch::Tensor Output1,
                torch::Tensor Filter2,
                torch::Tensor Output2,
                torch::Tensor fixup_bias1b,
                torch::Tensor fixup_bias2a,
                torch::Tensor fixup_bias2b,
                torch::Tensor fixup_scale) {
    const int batch_size = Activation.size(0);
    // In this case the same layout applies to both inputs
    static_assert(BlockIdx > 0 && BlockIdx < 5, "Block index must be 1, 2, 3, 4");

    constexpr int size_offset = (BlockIdx == 1)?0:(BlockIdx - 2);
    constexpr int in_channel = 64 << size_offset;
    constexpr int in_plane_size = 16 >> size_offset;
    constexpr int out_plane_size = (BlockIdx == 1)?16:(in_plane_size / 2);
    constexpr int out_channel = (BlockIdx == 1)?64:(in_channel * 2);
    constexpr int stride = (BlockIdx == 1)?1:2;
    // printf("batch_size is %d, in_channel is %d, in_plane is %d out plane is %d, inchannel is %d, outchannel is %d", batch_size, in_channel, in_plane_size, out_plane_size, in_channel, out_channel);
    const LayoutConvInputA layoutA = LayoutConvInputA::packed({batch_size, in_plane_size, in_plane_size, in_channel});
    const LayoutConvInputB layoutFilter1 = LayoutConvInputB::packed({out_channel, 3, 3, in_channel});
    const LayoutConvOutput layoutOut1 = LayoutConvOutput::packed({batch_size, out_plane_size, out_plane_size, out_channel});
    const LayoutConvInputB layoutFilter2 = LayoutConvInputB::packed({out_channel, 3, 3, out_channel});
    const LayoutConvOutput layoutResult = LayoutConvOutput::packed({batch_size, out_plane_size, out_plane_size, out_channel});

    const cutlass::conv::Conv2dProblemSize problem_size1(
        {batch_size, in_plane_size, in_plane_size, in_channel},
        {out_channel, 3, 3, in_channel},
        {1, 0, 1, 0},
        {stride, stride},
        {1, 1},
        {batch_size, out_plane_size, out_plane_size, out_channel},
        cutlass::conv::Mode::kCrossCorrelation,
        1);

    const cutlass::conv::Conv2dProblemSize problem_size2(
        {batch_size, out_plane_size, out_plane_size, out_channel},
        {out_channel, 3, 3, out_channel},
        {1, 0, 1, 0},
        {1, 1},
        {1, 1},
        {batch_size, out_plane_size, out_plane_size, out_channel},
        cutlass::conv::Mode::kCrossCorrelation,
        1);

    cutlass::TensorRef<ElementConvInputA, LayoutConvInputA> activation_ref(static_cast<ElementConvInputA*>(Activation.data_ptr()), layoutA);
    cutlass::TensorRef<ElementConvInputB, LayoutConvInputB> filter1_ref(static_cast<ElementConvInputB*>(Filter1.data_ptr()), layoutFilter1);
    cutlass::TensorRef<ElementConvOutput, LayoutConvOutput> out1_ref(static_cast<ElementConvOutput*>(Output1.data_ptr()), layoutOut1);
    cutlass::TensorRef<ElementConvInputB, LayoutConvInputB> filter2_ref(static_cast<ElementConvInputB*>(Filter2.data_ptr()), layoutFilter2);
    cutlass::TensorRef<ElementConvOutput, LayoutConvOutput> result_ref(static_cast<ElementConvOutput*>(Output2.data_ptr()), layoutResult);

    ElementComputeEpilogue* bias1b_ptr =
        static_cast<ElementComputeEpilogue*>(fixup_bias1b.data_ptr());
    ElementComputeEpilogue* bias2a_ptr =
        static_cast<ElementComputeEpilogue*>(fixup_bias2a.data_ptr());
    ElementComputeEpilogue* bias2b_ptr =
        static_cast<ElementComputeEpilogue*>(fixup_bias2b.data_ptr());
    ElementComputeEpilogue* scale_ptr =
        static_cast<ElementComputeEpilogue*>(fixup_scale.data_ptr());

    ImplicitGemm1 implicit_gemm_op1;
    ImplicitGemm2 implicit_gemm_op2;

    const typename ImplicitGemm1::Arguments arguments1{problem_size1, activation_ref, filter1_ref, out1_ref, out1_ref, {bias1b_ptr, bias2a_ptr}};
    cutlass::Status status1 = implicit_gemm_op1.initialize(arguments1);
    status1 = implicit_gemm_op1();

    const typename ImplicitGemm2::Arguments arguments2{problem_size2, out1_ref, filter2_ref, result_ref, result_ref,  {bias2b_ptr, nullptr, scale_ptr}};
    cutlass::Status status2 = implicit_gemm_op2.initialize(arguments2);
    status2 = implicit_gemm_op2();

    if (status1 != cutlass::Status::kSuccess ||
        status2 != cutlass::Status::kSuccess) {
        std::cout << cutlass::cutlassGetStatusString(status1) << std::endl;
        std::cout << cutlass::cutlassGetStatusString(status2) << std::endl;
    } else {
        std::cout << "No Problem!\n";
    }
}
#pragma once

#include <torch/extension.h>

#include <iostream>
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

class Conv128x16x16x64NHWC3x3x64NHWC {
    private:
    using ElementAccumulator = float;  // Data type of accumulator
    using ElementComputeEpilogue =
        float;  // Data type of epilogue computation (alpha, beta)
    using ElementInputA =
        cutlass::half_t;  // Data type of elements in input tensor
    using ElementInputB =
        cutlass::half_t;          // Data type of elements in input tensor
    using ElementOutput = cutlass::half_t;  // Data type of elements in output tensor

    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    // This code section describes whether you want to use tensor cores or
    // regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using ThreadblockShape =
        cutlass::gemm::GemmShape<64, 64, 32>;  // Threadblock tile shape

    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;  // Warp tile shape

    // This code section describes the size of MMA op
    using InstructionShape =
        cutlass::gemm::GemmShape<16, 8, 16>;  // TensorCore instruction shape

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // Number of pipelines you want to use
    static constexpr int NumStages = 3;

    // This code section describe iterator algorithm selected is Analytic or
    // Optimized
    static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
        cutlass::conv::IteratorAlgorithm::kAnalytic;

    // This code section describes the epilogue part of the kernel, we use
    // default value
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,  // Data type of output matrix.
        128 / cutlass::sizeof_bits<ElementOutput>::
                  value,     // The number of elements per vectorized.
                             // memory access. This becomes the vector width of
                             // math instructions in the epilogue too.
        ElementAccumulator,  // Data type of accumulator
        ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                  // combination

    using Conv2dFpropKernel =
        typename cutlass::conv::kernel::DefaultConv2dFprop<
            ElementInputA,
            LayoutInputA,
            ElementInputB,
            LayoutInputB,
            ElementOutput,
            LayoutOutput,
            ElementAccumulator,
            MMAOp,
            SmArch,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOp,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            IteratorAlgorithm>::Kernel;

    using ImplicitGemm =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    static const cutlass::conv::Conv2dProblemSize problem_size;
    static const cutlass::layout::TensorNHWC layout_activation;
    static const cutlass::layout::TensorNHWC layout_filter;
    static const cutlass::layout::TensorNHWC layout_output;

   private:
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> Activation;
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> Filter;
    cutlass::TensorRef<ElementOutput, cutlass::layout::TensorNHWC> Out;
    typename ImplicitGemm::Arguments arguments;

    ImplicitGemm implicit_gemm_op;

   public:
    Conv128x16x16x64NHWC3x3x64NHWC(torch::Tensor Activation,
                                   torch::Tensor Filter,
                                   torch::Tensor output)
        : Activation(static_cast<cutlass::half_t*>(Activation.data_ptr()),
                     layout_activation),
          Filter(static_cast<cutlass::half_t*>(Filter.data_ptr()),
                 layout_filter),
          Out(static_cast<ElementOutput*>(output.data_ptr()), layout_output),
          arguments{
              problem_size, this->Activation, this->Filter, Out, Out, {}} {
        CHECK_INPUT(Activation);
        CHECK_INPUT(Filter);
        CHECK_INPUT(output);
    }

    void run() {
        cutlass::Status status = implicit_gemm_op.initialize(arguments);
        status = implicit_gemm_op();
        if (status != cutlass::Status::kSuccess) {
            std::cout << "something is not working\n";
        } else {
            std::cout << "Done!\n";
        }
    }
};

const cutlass::conv::Conv2dProblemSize
    Conv128x16x16x64NHWC3x3x64NHWC::problem_size(
        {128, 16, 16, 64},
        {64, 3, 3, 64},
        {1, 0, 1, 0},
        {1, 1},
        {1, 1},
        {128, 16, 16, 64},
        cutlass::conv::Mode::kCrossCorrelation,
        1);

const cutlass::layout::TensorNHWC
    Conv128x16x16x64NHWC3x3x64NHWC::layout_activation =
        cutlass::layout::TensorNHWC::packed({128, 16, 16, 64});

const cutlass::layout::TensorNHWC
    Conv128x16x16x64NHWC3x3x64NHWC::layout_filter =
        cutlass::layout::TensorNHWC::packed({64, 3, 3, 64});

const cutlass::layout::TensorNHWC
    Conv128x16x16x64NHWC3x3x64NHWC::layout_output =
        cutlass::layout::TensorNHWC::packed({128, 16, 16, 64});
#pragma once

#include <torch/extension.h>

#include <iostream>
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "linear_combination_relu_fixup.h"

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

class Conv128x16x16x64NHWC3x3x64NHWC {
    private:
    using ElementAccumulator = cutlass::half_t;  // Data type of accumulator
    using ElementComputeEpilogue =
        ElementAccumulator;  // Data type of epilogue computation (alpha, beta)
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

    using EpilogueOp1 = cutlass::epilogue::thread::LinearCombinationReluFixUp<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;

    using EpilogueOp2 = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,  // Data type of output matrix.
        128 / cutlass::sizeof_bits<ElementOutput>::
                  value,     // The number of elements per vectorized.
                             // memory access. This becomes the vector width of
                             // math instructions in the epilogue too.
        ElementAccumulator,  // Data type of accumulator
        ElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                  // combination

    using Conv2dFpropKernel1 =
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
            EpilogueOp1,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            IteratorAlgorithm>::Kernel;

    using Conv2dFpropKernel2 =
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
            EpilogueOp2,
            SwizzleThreadBlock,
            NumStages,
            cutlass::arch::OpMultiplyAdd,
            IteratorAlgorithm>::Kernel;

    using ImplicitGemm1 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel1>;

    using ImplicitGemm2 =
        cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel2>;

    static const cutlass::conv::Conv2dProblemSize problem_size1;
    static const cutlass::conv::Conv2dProblemSize problem_size2;
    static const cutlass::layout::TensorNHWC layout_activation;
    static const cutlass::layout::TensorNHWC layout_filter1;
    static const cutlass::layout::TensorNHWC layout_output1;
    static const cutlass::layout::TensorNHWC layout_filter2;
    static const cutlass::layout::TensorNHWC layout_output2;

   private:
    // cutlass::TensorRef<ElementInputA, cutlass::layout::TensorNHWC> Activation;
    cutlass::TensorRef<ElementInputB, cutlass::layout::TensorNHWC> Filter1;
    cutlass::TensorRef<ElementOutput, cutlass::layout::TensorNHWC> Out1;
    cutlass::TensorRef<ElementInputB, cutlass::layout::TensorNHWC> Filter2;
    cutlass::TensorRef<ElementOutput, cutlass::layout::TensorNHWC> Out2;
    const typename ImplicitGemm2::Arguments arguments2;

    // device pointers
    ElementAccumulator* fixup_bias1a_ptr;
    ElementAccumulator* fixup_bias1b_ptr;
    ElementAccumulator* fixup_bias2a_ptr;
    ElementAccumulator* fixup_bias2b_ptr;
    ElementAccumulator* fixup_scale_ptr;


    // typename ImplicitGemm::Arguments arguments;

    ImplicitGemm1 implicit_gemm_op1;
    ImplicitGemm2 implicit_gemm_op2;

   public:
    // Memory management done in pytorch
    Conv128x16x16x64NHWC3x3x64NHWC(torch::Tensor Filter1,
                                   torch::Tensor Output1,
                                   torch::Tensor Filter2,
                                   torch::Tensor Output2,
                                   torch::Tensor fixup_bias1a,
                                   torch::Tensor fixup_bias1b,
                                   torch::Tensor fixup_bias2a,
                                   torch::Tensor fixup_bias2b,
                                   torch::Tensor fixup_scale)
        : Filter1(static_cast<ElementInputB*>(Filter1.data_ptr()),
                  layout_filter1),
          Out1(static_cast<ElementOutput*>(Output1.data_ptr()), layout_output1),
          Filter2(static_cast<ElementInputB*>(Filter2.data_ptr()),
                  layout_filter2),
          Out2(static_cast<ElementOutput*>(Output2.data_ptr()), layout_output2),
          fixup_bias1a_ptr(
              static_cast<ElementAccumulator*>(fixup_bias1a.data_ptr())),
          fixup_bias1b_ptr(
              static_cast<ElementAccumulator*>(fixup_bias1b.data_ptr())),
          fixup_bias2a_ptr(
              static_cast<ElementAccumulator*>(fixup_bias2a.data_ptr())),
          fixup_bias2b_ptr(
              static_cast<ElementAccumulator*>(fixup_bias2b.data_ptr())),
          fixup_scale_ptr(
              static_cast<ElementAccumulator*>(fixup_scale.data_ptr())),
          arguments2{problem_size2, this->Out1, this->Filter2, Out2, Out2, {}} {
        CHECK_INPUT(Filter1);
        CHECK_INPUT(Output1);
        CHECK_INPUT(Filter2);
        CHECK_INPUT(Output2);
        CHECK_INPUT(fixup_bias1a);
        CHECK_INPUT(fixup_bias1b);
        CHECK_INPUT(fixup_bias2a);
        CHECK_INPUT(fixup_bias2b);
        CHECK_INPUT(fixup_scale);
    }

    void run(torch::Tensor Activation) {
        cutlass::TensorRef<ElementInputA, cutlass::layout::TensorNHWC> Act_ref(
            static_cast<ElementInputA*>(Activation.data_ptr()),
            layout_activation);
        const typename ImplicitGemm1::Arguments arguments1{problem_size1, Act_ref, Filter1, Out1, Out1, {fixup_bias1b_ptr, fixup_bias2a_ptr}};
        cutlass::Status status1 = implicit_gemm_op1.initialize(arguments1);
        status1 = implicit_gemm_op1();

        cutlass::Status status2 = implicit_gemm_op2.initialize(arguments2);
        status2 = implicit_gemm_op2();

        if (status1 != cutlass::Status::kSuccess || status2 != cutlass::Status::kSuccess) {
            std::cout << cutlass::cutlassGetStatusString(status1) << std::endl;
            std::cout << cutlass::cutlassGetStatusString(status2) << std::endl;
        } else {
            std::cout << "Done!\n";
        }
    }
};

const cutlass::conv::Conv2dProblemSize
    Conv128x16x16x64NHWC3x3x64NHWC::problem_size1(
        {128, 16, 16, 64},
        {64, 3, 3, 64},
        {1, 0, 1, 0},
        {1, 1},
        {1, 1},
        {128, 16, 16, 64},
        cutlass::conv::Mode::kCrossCorrelation,
        1);

const cutlass::conv::Conv2dProblemSize
    Conv128x16x16x64NHWC3x3x64NHWC::problem_size2(
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
    Conv128x16x16x64NHWC3x3x64NHWC::layout_filter1 =
        cutlass::layout::TensorNHWC::packed({64, 3, 3, 64});

const cutlass::layout::TensorNHWC
    Conv128x16x16x64NHWC3x3x64NHWC::layout_output1 =
        cutlass::layout::TensorNHWC::packed({128, 16, 16, 64});

const cutlass::layout::TensorNHWC
    Conv128x16x16x64NHWC3x3x64NHWC::layout_filter2 =
        cutlass::layout::TensorNHWC::packed({64, 3, 3, 64});

const cutlass::layout::TensorNHWC
    Conv128x16x16x64NHWC3x3x64NHWC::layout_output2 =
        cutlass::layout::TensorNHWC::packed({128, 16, 16, 64});
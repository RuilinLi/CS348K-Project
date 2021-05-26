#pragma once

#include <torch/extension.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Stem Ops, 1x1 convolution on the input of size N, 16, 16, 48
// output has size N, 16, 16, 64
// GEMM of size (N * 256, 64, 48)
#define myK 64

using StemElementAccumulator = cutlass::half_t;
using StemElementComputeEpilogue = StemElementAccumulator;
using StemElementInputA = cutlass::half_t;
using StemElementInputB = cutlass::half_t;
using StemElementOutput = cutlass::half_t;

using StemLayoutInputA = cutlass::layout::RowMajor;
using StemLayoutInputB = cutlass::layout::RowMajor;
using StemLayoutOutput = cutlass::layout::RowMajor;



// This code section describes the tile size a thread block will compute
using StemShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 64, 16>;
// This code section describes tile size a warp will compute
using StemShapeMMAWarp = cutlass::gemm::GemmShape<64, 32, 16>;
// This code section describes the size of MMA op
using StemShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

// This code section describes how threadblocks are scheduled on GPU
using StemSwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using StemEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    StemElementOutput,  // Data type of output matrix.
    128 / cutlass::sizeof_bits<StemElementOutput>::
              value,         // The number of elements per vectorized.
                             // memory access. This becomes the vector width of
                             // math instructions in the epilogue too.
    StemElementAccumulator,  // Data type of accumulator
    StemElementComputeEpilogue>;  // Data type for alpha/beta in linear
                                  // combination

using Stem1x1Conv =
    cutlass::gemm::device::Gemm<StemElementInputA,
                                StemLayoutInputA,
                                StemElementInputB,
                                StemLayoutInputB,
                                StemElementOutput,
                                StemLayoutOutput,
                                StemElementAccumulator,
                                cutlass::arch::OpClassTensorOp,
                                cutlass::arch::Sm80,
                                StemShapeMMAThreadBlock,
                                StemShapeMMAWarp,
                                StemShapeMMAOp,
                                StemEpilogueOp,
                                StemSwizzleThreadBlock,
                                2>;


// from 48 channels to 64 channesl
void StemOp48_64(torch::Tensor Input, torch::Tensor Filter, torch::Tensor Output){
    const int batch_size = Input.size(0);
    const StemLayoutInputA layoutA = StemLayoutInputA::packed({batch_size*16*16, myK});
    const StemLayoutInputB layoutB = StemLayoutInputB::packed({myK, 64});
    const StemLayoutOutput layoutC = StemLayoutOutput::packed({batch_size*16*16, 64});
    cutlass::TensorRef<StemElementInputA, StemLayoutInputA> A(static_cast<StemElementInputA*>(Input.data_ptr()), layoutA);
    cutlass::TensorRef<StemElementInputB, StemLayoutInputB> B(static_cast<StemElementInputB*>(Filter.data_ptr()), layoutB);
    cutlass::TensorRef<StemElementOutput, StemLayoutOutput> C(static_cast<StemElementOutput*>(Output.data_ptr()), layoutC);
    cutlass::gemm::GemmCoord problem_size(batch_size*16*16, 64, myK);

    StemElementComputeEpilogue alpha = StemElementComputeEpilogue(1);
    StemElementComputeEpilogue beta = StemElementComputeEpilogue(0);
    std::cout << Input.size(0) <<  std::endl;
    std::cout << Input.size(1) <<  std::endl;
    std::cout << Input.size(2) <<  std::endl;
    std::cout << Input.size(3) <<  std::endl;

    typename Stem1x1Conv::Arguments arguments{
        problem_size,  // <- problem size of matrix multiplication
        A,             // <- reference to matrix A on device
        B,             // <- reference to matrix B on device
        C,             // <- reference to matrix C on device
        C,             // <- reference to matrix D on device
        {alpha, beta},
        1};

    Stem1x1Conv GemmOp;

    cutlass::Status status = GemmOp.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cout << cutlass::cutlassGetStatusString(status) << std::endl;
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = GemmOp.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cout << cutlass::cutlassGetStatusString(status) << std::endl;
    }

    // // Launch initialized CUTLASS kernel
    status = GemmOp();
    if (status != cutlass::Status::kSuccess) {
        std::cout << cutlass::cutlassGetStatusString(status) << std::endl;
    }
}


// Ideally this should be fused with the GEMM that comes after
// But I'm running out of time....
// template <int N_sample_per_block = 1, int IMG_SIZE=16, int Channel_count = 64>
// __global__ void avg_pool2d_kernel(const at::Half * __restrict__ activation, at::Half * __restrict__ pooled_result){

// }
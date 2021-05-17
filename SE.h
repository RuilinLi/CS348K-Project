#pragma once

#include <torch/extension.h>

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/reduction/device/tensor_reduce.h"
#include "device/b2b_gemm.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include <cuda_pipeline.h>


#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

template <int N0, int K0=N0*16, int N1=K0, int K1 = N0, int Mtile_size=32>
__global__ void SEb2bGEMMFused_Kernel(
    const cutlass::half_t* __restrict__ Squeezed,
    const cutlass::half_t* __restrict__ W0,
    const cutlass::half_t* __restrict__ bias0,
    const cutlass::half_t* __restrict__ W1,
    const cutlass::half_t* __restrict__ bias1,
    cutlass::half_t* __restrict__ out) {
        


    cutlass::epilogue::thread::Sigmoid<float> sigmoid_op;


    cutlass::half_t out0_tile[N0];
    cutlass::half_t W0_register[N0 * K0];  // load entire W0 to register
    __shared__ cutlass::half_t
        A0_tile[Mtile_size * (K0 + 1)];  // add an offset to avoid bank conflict

#pragma unroll 32
    for (int i = 0; i < K0; i++) {
        A0_tile[threadIdx.x * (K0 + 1) + i] =
            Squeezed[(blockIdx.x * Mtile_size + threadIdx.x) * K0 + i] * static_cast<cutlass::half_t>(0.00390625f); // averaging from the last part
    }

#pragma unroll 32
    for (int i = 0; i < N0 * K0; ++i) {
        W0_register[i] = W0[i];
    }

    __syncthreads();
    // Before starting computation, issue an async memcpy to get a tile from W1,
    // bias0, bias1
    __shared__ cutlass::half_t W1_shared[N1 * K1];
    __shared__ cutlass::half_t bias0_shared[N0];
    __shared__ cutlass::half_t bias1_shared[N1];


    // 256 = 32 threads (1 warp) * 8 half_t per thread
    for (int i = 0; i < N1 * K1 / 256; i++) {
        __pipeline_memcpy_async(&W1_shared[8 * threadIdx.x + i * 256],
                                &W1[8 * threadIdx.x + i * 256],
                                8 * sizeof(cutlass::half_t));
    }

    __pipeline_memcpy_async(&bias1_shared[threadIdx.x * (N1/32)], &bias1[threadIdx.x * (N1/32)],
                            (N1 / 32) * sizeof(cutlass::half_t));

    if (threadIdx.x == 0) {
        __pipeline_memcpy_async(&bias0_shared[0], &bias0[0],
                                N0 * sizeof(cutlass::half_t));
    }


    __pipeline_commit();

    // Now start computation

#pragma unroll N0
    for (int i = 0; i < N0; ++i) {
        out0_tile[i] = cutlass::half_t(0);
    }

    for (int k = 0; k < K0; ++k) {
#pragma unroll N0
        for (int i = 0; i < N0; ++i) {
            out0_tile[i] +=
                A0_tile[threadIdx.x * (K0 + 1) + k] * W0_register[k * N0 + i];
        }
    }

    __pipeline_wait_prior(0);

// relu
#pragma unroll N0
    for (int i = 0; i < N0; ++i) {
        out0_tile[i] = out0_tile[i] + bias0_shared[i];
        if(out0_tile[i] < cutlass::half_t(0)){
            out0_tile[i] = cutlass::half_t(0);
        }
    }


    for (int k = 0; k < N1; ++k) {
        // Use float accumulator because pytorch blocks some of the half operators
        float result = 0.f;

#pragma unroll N0
        for (int i = 0; i < K1; ++i) {
            result += static_cast<float>(
                out0_tile[i] *
                W1_shared[i * N1 + k]);  // Broadcast, no bank conflict
        }
        result += static_cast<float>(bias1_shared[k]);
        result = sigmoid_op(result);
        out[(blockIdx.x * Mtile_size + threadIdx.x) * N1 + k] = static_cast<cutlass::half_t>(result);
    }
}

template <int N0>
struct SEb2bGEMMFused{
    // static int const M = 128;
    // static int const K0 = N0 * 16; // N0 should be 4 or 8, larger problem size will be handled using CUTLASS
    // static int const N1 = K0;
    // static int const K1 = N0;
    // static int const Mtile_size = 32;


    // static cutlass::epilogue::thread::ReLu<cutlass::half_t> const  relu_op;
    // static cutlass::epilogue::thread::Sigmoid<cutlass::half_t> const  sigmoid_op;

    // // using gemm0_shape = cutlass::gemm::GemmShape<128, N0, N0 * 16>; // e.g. <128, 4, 64>
    // // using gemm1_shape = cutlass::gemm::GemmShape<128, N0 * 16, N0>; // e.g. <128, 64, 4>
    // // using threadblock0_shape = cutlass::gemm::GemmShape<32, N0, N0 * 16>; // <32, 4, 64>
    // // using threadblock1_shape = cutlass::gemm::GemmShape<32, N0 * 16, N0>; // <32, 64, 4>

    // // Each thread compute a row in the output, each threadblock only uses one warp
    // __global__ void SEb2bGEMMFused_Kernel(
    //     const cutlass::half_t* __restrict__ Squeezed,
    //     const cutlass::half_t* __restrict__ W0,
    //     const cutlass::half_t* __restrict__ bias0,
    //     const cutlass::half_t* __restrict__ W1,
    //     const cutlass::half_t* __restrict__ bias1,
    //     cutlass::half_t* __restrict__ out){
    //         cutlass::half_t out0_tile[N0];
    //         cutlass::half_t W0_register[N0 * K0]; // load entire W0 to register
    //         __shared__ cutlass::half_t A0_tile[Mtile_size * (K0 + 1)]; // add an offset to avoid bank conflict

    //         #pragma unroll 32
    //         for(int i = 0; i < N0; i++){
    //             A0_tile[threadIdx.x * (K0 + 1) + i] = Squeezed[(blockIdx.x*Mtile_size + threadIdx.x) * K0 + i];
    //         }

    //         #pragma unroll 32
    //         for(int i = 0; i < N0 * K0){
    //             W0_register[i] = W0[i];
    //         }

    //         __syncthreads();
    //         // Before starting computation, issue an async memcpy to get a tile from W1, bias0, bias1
    //         __shared__ cutlass::half_t W1_shared[N1 * K1];
    //         __shared__ cutlass::half_t bias0_shared[N0];
    //         __shared__ cutlass::half_t bias1_shared[N1];

    //         // 256 = 32 threads (1 warp) * 8 half_t per thread
    //         for (int i = 0; i < N1 * K1 / 256; i++) {
    //             __pipeline_memcpy_async(&W1_shared[8 * threadIdx.x + i * 256], &W1[8 * threadIdx.x + i * 256],
    //                                     8 * sizeof(cutlass::half_t));
    //         }

    //         if (threadIdx.x == 0) {
    //             __pipeline_memcpy_async(&bias0_shared[0], &bias0[0],
    //                                     N0 * sizeof(cutlass::half_t));
    //             __pipeline_memcpy_async(&bias1_shared[0], &bias1[0],
    //                                     N1 * sizeof(cutlass::half_t));
    //         }

    //         __pipeline_commit();

    //         // Now start computation

    //         #pragma unroll N0
    //         for(int i = 0; i < N0; ++i){
    //             out0_tile[i] = cutlass::half_t(0);
    //         }

    //         for(int k = 0; k < K0; ++k){
    //             #pragma unroll N0
    //             for(int i = 0; i < N0; ++i){
    //                 out0_tile[i] += A0_tile[threadIdx.x * (K0 + 1) + k] * W0_register[k * N0 + i];
    //             }
    //         }

    //         __pipeline_wait_prior(0);

    //         // relu
    //         #pragma unroll N0
    //         for(int i = 0; i < N0; ++i){
    //             out0_tile[i] = out0_tile[i] + bias0_shared[i];
    //             out0_tile[i] = relu_op(out0_tile[i]);
    //         }

    //         for(int k = 0; k < K1; ++k){
    //             cutlass::half_t result = cutlass::half_t(0);

    //             #pragma unroll N0
    //             for(int i = 0; i < N0; ++i){
    //                 result += out0_tile[i] * W1_shared[i * N1 + k]; // Broadcast, no bank conflict
    //             }
    //             result = sigmoid_op(result + bias1_shared[k]);
    //             out[(blockIdx.x*Mtile_size + threadIdx.x) * K1 + k] = result;
    //         }
    //     }

    static void run(const cutlass::half_t* __restrict__ Squeezed,
                    const cutlass::half_t* __restrict__ W0,
                    const cutlass::half_t* __restrict__ bias0,
                    const cutlass::half_t* __restrict__ W1,
                    const cutlass::half_t* __restrict__ bias1,
                    cutlass::half_t* __restrict__ out) {
        SEb2bGEMMFused_Kernel<N0>
            <<<128 / 32, 32>>>(Squeezed, W0, bias0, W1, bias1, out);
        cudaError_t err = cudaGetLastError();
        std::cout << cudaGetErrorString(err) << std::endl;
    }
};

// To start off assume input has dimension (128, 16, 16, 64)
// To use cutlass reduction, interpret as a (1, 128,  256, 64) tensor
// reduction -> (1, 128, 1, 64) -> row major matrix (128, 64)
// bottle neck (128, 64) -> (128, 4) -> (128, 64):
// let x be the resnet input (possibly downsampled), y be the output
// what needs to be done here are
// relu(y * se(y) + x)

class ResnetSE {
   private:
    using Layout = cutlass::layout::TensorNHWC;
    using ElementOutput = cutlass::half_t;
    using ElementSource = cutlass::half_t;
    using ElementCompute = float;
    using ElementAccumulator = ElementCompute;

    // Define the functor
    using Functor = cutlass::plus<ElementCompute>;

    static int const kV = 1;

    using TensorReduction =
        cutlass::reduction::device::TensorReduction<ElementOutput,
                                                    ElementSource,
                                                    Layout,
                                                    Functor,
                                                    kV,
                                                    ElementCompute>;

    static const cutlass::layout::TensorNHWC input_tensor_layout;
    static const cutlass::layout::TensorNHWC squeezed_tensor_layout;
    static const cutlass::layout::RowMajor squeezed_matrix_layout;
    static const cutlass::layout::RowMajor W1_layout;
    static const cutlass::layout::RowMajor W2_layout;
    static const cutlass::layout::ColumnMajor excited_matrix_layout;

    cutlass::TensorRef<ElementSource, cutlass::layout::TensorNHWC> Activation;
    cutlass::TensorRef<ElementOutput, cutlass::layout::TensorNHWC> Squeezed;
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> W1;
    cutlass::TensorRef<ElementOutput, cutlass::layout::ColumnMajor> bias1;

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> W2;
    cutlass::TensorRef<ElementOutput, cutlass::layout::ColumnMajor> bias2;

    cutlass::TensorRef<ElementOutput, cutlass::layout::ColumnMajor> Excited;

    TensorReduction reduction;

   public:
    ResnetSE(torch::Tensor Activation,
             torch::Tensor Squeezed,
             torch::Tensor W1,
             torch::Tensor bias1,
             torch::Tensor W2,
             torch::Tensor bias2,
             torch::Tensor Excited)
        : Activation(static_cast<ElementSource*>(Activation.data_ptr()),
                     input_tensor_layout),
          Squeezed(static_cast<ElementOutput*>(Squeezed.data_ptr()),
                   squeezed_tensor_layout),
          W1(static_cast<ElementOutput*>(W1.data_ptr()), W1_layout),
          bias1(static_cast<ElementOutput*>(bias1.data_ptr()), {0}),
          W2(static_cast<ElementOutput*>(W2.data_ptr()), W2_layout),
          bias2(static_cast<ElementOutput*>(bias2.data_ptr()), {0}),
          Excited(static_cast<ElementOutput*>(Excited.data_ptr()),
                  excited_matrix_layout),
          reduction({1, 128, 256, 64}, 2) {
        CHECK_INPUT(Activation);
        CHECK_INPUT(Squeezed);
        CHECK_INPUT(W1);
        CHECK_INPUT(bias1);
        CHECK_INPUT(W2);
        CHECK_INPUT(bias2);
        CHECK_INPUT(Excited);
    }

    // This sums over the HW dimension, we want average.
    // Divide the H x W extent in the next GEMM (could this cause problem in
    // precision?)
    void reduce() {
        cutlass::Status status = reduction.reduce(Squeezed, Activation);
        if (status != cutlass::Status::kSuccess) {
            std::cout << cutlass::cutlassGetStatusString(status) << std::endl;
        }


        SEb2bGEMMFused<4> b2bgemm_op;
        b2bgemm_op.run(Squeezed.data(), W1.data(), bias1.data(), W2.data(), bias2.data(),Excited.data());


    }
};

const cutlass::layout::TensorNHWC ResnetSE::input_tensor_layout =
    cutlass::layout::TensorNHWC::packed({1, 128, 256, 64});
const cutlass::layout::TensorNHWC ResnetSE::squeezed_tensor_layout =
    cutlass::layout::TensorNHWC::packed({1, 128, 1, 64});
const cutlass::layout::RowMajor ResnetSE::squeezed_matrix_layout =
    cutlass::layout::RowMajor::packed({128, 64});
const cutlass::layout::RowMajor ResnetSE::W1_layout =
    cutlass::layout::RowMajor::packed({64, 4});
const cutlass::layout::RowMajor ResnetSE::W2_layout =
    cutlass::layout::RowMajor::packed({4, 64});
const cutlass::layout::ColumnMajor ResnetSE::excited_matrix_layout =
    cutlass::layout::ColumnMajor::packed({128, 64});

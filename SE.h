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

template <int N0, int K0=N0*16, int N1=K0, int K1 = N0, int Mtile_size=32, int IMG_SIZE=64/N0>
__global__ void SEb2bGEMMFused_Kernel(
    const cutlass::half_t* __restrict__ activation,
    const cutlass::half_t* __restrict__ W0,
    const cutlass::half_t* __restrict__ bias0,
    const cutlass::half_t* __restrict__ W1,
    const cutlass::half_t* __restrict__ bias1,
    cutlass::half_t* __restrict__ downsampled) {
    // activation is a tensor of size (128, 16, 16, 64) or (128, 8, 8, 128)
        


    cutlass::epilogue::thread::Sigmoid<float> sigmoid_op;

    constexpr int HALF_T_BANK_PADDING = 2;


    cutlass::half_t out0_tile[N0];

    __shared__ cutlass::half_t W0_tile[N0 * K0]; // too large for registers when N0 = 8
    __shared__ cutlass::half_t
        A0_tile[Mtile_size * (K0 + HALF_T_BANK_PADDING)];  // add an offset to avoid bank conflict

    for(int i = 0; i < (N0 * K0 / 256); ++i){
        __pipeline_memcpy_async(&W0_tile[8 * threadIdx.x + i * 256], &W0[8 * threadIdx.x + i * 256], sizeof(cutlass::half_t) * 8);

    }
    __pipeline_commit();
    
    
    // reduction
    for (int k = 0; k < K0; ++k) {
        A0_tile[threadIdx.x * (K0 + HALF_T_BANK_PADDING) + k] =
            cutlass::half_t(0);
    }

    for(int j = 0; j < (IMG_SIZE * IMG_SIZE); ++j){
        #pragma unroll 32
        for(int k = 0; k < K0; ++k){
            const int idx = (blockIdx.x * Mtile_size + threadIdx.x) * (IMG_SIZE * IMG_SIZE * K0) + j * K0 + k;
            A0_tile[threadIdx.x * (K0 + HALF_T_BANK_PADDING) + k] += activation[idx] * static_cast<cutlass::half_t>(1.f/(IMG_SIZE * IMG_SIZE));
        }
    }

    __syncthreads(); // there is only one warp...
    __pipeline_wait_prior(0);
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
                A0_tile[threadIdx.x * (K0 + HALF_T_BANK_PADDING) + k] * W0_tile[k * N0 + i];
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


    cutlass::half_t results[N1];
    for (int k = 0; k < N1; ++k) {
        // Use float accumulator because pytorch blocks some of the half operators
        float result = 0.f;

        #pragma unroll K1
        for (int i = 0; i < K1; ++i) {
            result += static_cast<float>(
                out0_tile[i] *
                W1_shared[i * N1 + k]);  // Broadcast, no bank conflict
        }
        result += static_cast<float>(bias1_shared[k]);
        result = sigmoid_op(result);
        results[k] = static_cast<cutlass::half_t>(result);
    }

    for(int j = 0; j < (IMG_SIZE * IMG_SIZE); ++j){
        #pragma unroll 32
        for(int k = 0; k < N1; ++k){
            int idx = (blockIdx.x * Mtile_size + threadIdx.x) * (IMG_SIZE * IMG_SIZE * N1) + j * N1 + k;
            downsampled[idx] += activation[idx] * results[k];
        }
    }
}

template <int N0, int K0=N0*16, int N1=K0, int K1 = N0, int Mtile_size=32, int IMG_SIZE=64/N0>
__global__ void SEb2bGEMMFused_Kernel2(
    const cutlass::half_t* __restrict__ activation,
    const cutlass::half_t* __restrict__ W0,
    const cutlass::half_t* __restrict__ bias0,
    const cutlass::half_t* __restrict__ W1,
    const cutlass::half_t* __restrict__ bias1,
    cutlass::half_t* __restrict__ downsampled) {
    
    const int reduction_thread_per_channel = blockDim.x / K0;
    const int thread_reduction_width = (IMG_SIZE * IMG_SIZE)/reduction_thread_per_channel;
    __shared__ cutlass::half_t A0_row[K0];
    const int channel_id = threadIdx.x % K0;
    const int start_row_id = (threadIdx.x / K0) * thread_reduction_width;

    cutlass::epilogue::thread::Sigmoid<float> sigmoid_op;

    __shared__ cutlass::half_t W0_shared[N0 * K0];
    __shared__ cutlass::half_t W1_shared[N1 * K1];
    __shared__ cutlass::half_t bias0_shared[N0];
    __shared__ cutlass::half_t bias1_shared[N1];


    // Issue memcpy to get the first weight matrix and bias
    if(threadIdx.x < (N0 * K0)/8){
        __pipeline_memcpy_async(&W0_shared[8 * threadIdx.x],
                                &W0[8 * threadIdx.x],
                                8 * sizeof(cutlass::half_t));
    }

    // N0 = 4 or 8
    if(threadIdx.x == 0){
        __pipeline_memcpy_async(&bias0_shared[0], &bias0[0],
                                N0 * sizeof(cutlass::half_t));
    }

    __pipeline_commit();

    // Each thread computes part of the channel-wise average
    cutlass::half_t tmp(0.f);
    #pragma unroll 16
    for(int i = 0; i < thread_reduction_width; ++i){
        // add (blockIdx.x, start_row_id+i, channel_id)
        tmp += activation[blockIdx.x * (IMG_SIZE * IMG_SIZE * K0) + (start_row_id + i) * K0 + channel_id];
    }

    if(threadIdx.x < K0){
        A0_row[threadIdx.x] = cutlass::half_t(0.f);
    }
    __syncthreads();

    // Put the partial reduction together
    for(int i = 0; i < reduction_thread_per_channel; ++i){
        if((threadIdx.x / K0) == i){
            A0_row[channel_id] += tmp * static_cast<cutlass::half_t>(1.f/(IMG_SIZE * IMG_SIZE));
        }
        __syncthreads();
    }

    // Wait until W0, bias0 arrives
    __pipeline_wait_prior(0);

    // Issue async memcpy to get W1 and bias1
    if(threadIdx.x < (N1 * K1)/8){
        __pipeline_memcpy_async(&W1_shared[8 * threadIdx.x],
                                &W1[8 * threadIdx.x],
                                8 * sizeof(cutlass::half_t));
    }
    

    // Need to confirm this, if send k byte then input must be k-aligned?
    if(threadIdx.x < (N1 / 4)){
        __pipeline_memcpy_async(&bias1_shared[4 * threadIdx.x], &bias1[4 * threadIdx.x],
                                4 * sizeof(cutlass::half_t));
    }



    __pipeline_commit();



    // decompose matrix multiplication to element-wise multiplication and reduction
    for(int i = 0; i < ((K0 * N0) + blockDim.x - 1)/blockDim.x; ++i){
        const int linear_idx = threadIdx.x + i * blockDim.x;
        if(linear_idx < K0 * N0){
            W0_shared[linear_idx] *= A0_row[linear_idx / N0]; //broadcast, no bank conflict
        }
    }
    // if(threadIdx.x < K0 * N0){
    //     W0_shared[threadIdx.x] *= A0_row[threadIdx.x / N0];
    // }

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int col_idx = warp_id % N0;
    const int row_idx = 32 * (warp_id / N0);

    __syncthreads();
    // Reduction
    for(unsigned int s = K0 / 2; s > 0; s>>=1){
        if(row_idx + lane_id < s){
            W0_shared[(row_idx + lane_id) * N0 + col_idx] += W0_shared[(row_idx + lane_id + s) * N0 + col_idx];
        }
        __syncthreads();
    }

    // For some reason this warp level reduction didn't work...
    // if(warp_id < N0){
    //     W0_shared[(lane_id) * N0 + col_idx] += W0_shared[(lane_id + 32) * N0 + col_idx];
    //     W0_shared[(lane_id) * N0 + col_idx] += W0_shared[(lane_id + 8) * N0 + col_idx];
    //     W0_shared[(lane_id) * N0 + col_idx] += W0_shared[(lane_id + 4) * N0 + col_idx];
    //     W0_shared[(lane_id) * N0 + col_idx] += W0_shared[(lane_id + 2) * N0 + col_idx];
    //     W0_shared[(lane_id) * N0 + col_idx] += W0_shared[(lane_id + 1) * N0 + col_idx];
    // }
    // __syncthreads();

    // Now the first N0 term of W0_shared stores the output
    if(threadIdx.x < N0){
        W0_shared[threadIdx.x] += bias0_shared[threadIdx.x];
        if(W0_shared[threadIdx.x] < cutlass::half_t(0)){
            W0_shared[threadIdx.x] = cutlass::half_t(0);
        }
    } // relu
    
    __pipeline_wait_prior(0);
    __syncthreads();
    // if(threadIdx.x < K1 * N1){
    //     W1_shared[threadIdx.x] *= W0_shared[threadIdx.x / N1];
    // }
    for (int i = 0; i < ((K1 * N1) + blockDim.x - 1) / blockDim.x; ++i) {
        const int linear_idx = threadIdx.x + i * blockDim.x;
        if (linear_idx < K1 * N1) {
            W1_shared[linear_idx] *=
                W0_shared[linear_idx / N1];  // broadcast, no bank conflict
        }
    }

    __syncthreads();
    // another small reduction hard code it
    // N1 = 64, K1 = 4, or N1 = 128, K1 = 8
    const int row_idx1 = threadIdx.x / N1;
    const int col_idx1 = threadIdx.x % N1;

    if (N1 == 128) {
        if (row_idx1 < 4) {
            W1_shared[col_idx1 + row_idx1 * N1] +=
                W1_shared[col_idx1 + (row_idx1 + 4) * N1];
        }
        __syncthreads();
    }
    
    if (row_idx1 < 2) {
        W1_shared[col_idx1 + row_idx1 * N1] +=
            W1_shared[col_idx1 + (row_idx1 + 2) * N1];
    }
    __syncthreads();
    if (row_idx1 == 0) {
        W1_shared[col_idx1] += W1_shared[col_idx1 + N1];
    }


    __syncthreads();
    // bias + sigmoid
    if(threadIdx.x < N1){
        float tmp = sigmoid_op(static_cast<float>(W1_shared[threadIdx.x] + bias1_shared[threadIdx.x]));
        W1_shared[threadIdx.x] = static_cast<cutlass::half_t>(tmp);
    }
    __syncthreads();
    // Finally multiply the activation with size (IMG_SIZE, IMG_SIZE, K0)
    // (IMG_SIZE * IMG_SIZE * K0) must be an integer multiple of blockDim.x
    #pragma unroll 8
    for(int i = 0; i < (IMG_SIZE * IMG_SIZE * K0) / blockDim.x; ++i){
        const int inner_idx = threadIdx.x + i * blockDim.x;
        downsampled[blockIdx.x * (IMG_SIZE * IMG_SIZE * K0) + inner_idx] +=
            activation[blockIdx.x * (IMG_SIZE * IMG_SIZE * K0) + inner_idx] *
            W1_shared[inner_idx % K0];
    }
}

template <int N0>
struct SEb2bGEMMFused{
    static void run(const cutlass::half_t* __restrict__ activation,
                    const cutlass::half_t* __restrict__ W0,
                    const cutlass::half_t* __restrict__ bias0,
                    const cutlass::half_t* __restrict__ W1,
                    const cutlass::half_t* __restrict__ bias1,
                    cutlass::half_t* __restrict__ downsampled) {
        SEb2bGEMMFused_Kernel<N0>
            <<<128 / 32, 32>>>(activation, W0, bias0, W1, bias1, downsampled);
        cudaError_t err = cudaGetLastError();
        std::cout << cudaGetErrorString(err) << std::endl;
    }
};


template <int N0>
class MyCudaSE{

    using ElementInput = cutlass::half_t;

    ElementInput* activation;
    ElementInput* W1;
    ElementInput* bias1;
    ElementInput* W2;
    ElementInput* bias2;
    ElementInput* downsampled;

    public:
     MyCudaSE(torch::Tensor Activation,
              torch::Tensor W1,
              torch::Tensor bias1,
              torch::Tensor W2,
              torch::Tensor bias2,
              torch::Tensor downsampled)
         : activation(static_cast<ElementInput*>(Activation.data_ptr())),
           W1(static_cast<ElementInput*>(W1.data_ptr())),
           bias1(static_cast<ElementInput*>(bias1.data_ptr())),
           W2(static_cast<ElementInput*>(W2.data_ptr())),
           bias2(static_cast<ElementInput*>(bias2.data_ptr())),
           downsampled(static_cast<ElementInput*>(downsampled.data_ptr())) {}

    void run() {
        SEb2bGEMMFused_Kernel2<N0>
            <<<128, 512>>>(activation, W1, bias1, W2, bias2, downsampled);
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
        // if (status != cutlass::Status::kSuccess) {
        //     std::cout << cutlass::cutlassGetStatusString(status) << std::endl;
        // }


        // SEb2bGEMMFused<4> b2bgemm_op;
        // b2bgemm_op.run(Squeezed.data(), W1.data(), bias1.data(), W2.data(), bias2.data(),Excited.data());


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

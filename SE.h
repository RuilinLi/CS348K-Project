#pragma once

#include <torch/extension.h>

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/reduction/device/tensor_reduce.h"
#include "device/b2b_gemm.h"
#include "cuda_fp16.h"
#include <mma.h>
#include <cuda_pipeline.h>


#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)


template <int N0, typename HALF_T = at::Half, int K0=N0*16, int N1=K0, int K1 = N0, int IMG_SIZE=64/N0>
__global__ void SEb2bGEMMFused_Kernel(
    const HALF_T* __restrict__ activation,
    const HALF_T* __restrict__ W0,
    const HALF_T* __restrict__ bias0,
    const HALF_T* __restrict__ W1,
    const HALF_T* __restrict__ bias1,
    HALF_T* __restrict__ downsampled) {
    
    const int reduction_thread_per_channel = blockDim.x / K0;
    const int thread_reduction_width = (IMG_SIZE * IMG_SIZE)/reduction_thread_per_channel;
    __shared__ HALF_T A0_row[K0];
    const int channel_id = threadIdx.x % K0;
    const int start_row_id = (threadIdx.x / K0) * thread_reduction_width;

    __shared__ HALF_T W0_shared[N0 * K0];
    __shared__ HALF_T W1_shared[N1 * K1];
    __shared__ HALF_T bias0_shared[N0];
    __shared__ HALF_T bias1_shared[N1];


    // Issue memcpy to get the first weight matrix and bias
    if(threadIdx.x < (N0 * K0)/8){
        __pipeline_memcpy_async(&W0_shared[8 * threadIdx.x],
                                &W0[8 * threadIdx.x],
                                8 * sizeof(HALF_T));
    }

    // N0 = 4 or 8
    if(threadIdx.x == 0){
        __pipeline_memcpy_async(&bias0_shared[0], &bias0[0],
                                N0 * sizeof(HALF_T));
    }

    __pipeline_commit();

    // Each thread computes part of the channel-wise average
    HALF_T tmp(0.f);
    #pragma unroll 16
    for(int i = 0; i < thread_reduction_width; ++i){
        // add (blockIdx.x, start_row_id+i, channel_id)
        tmp += activation[blockIdx.x * (IMG_SIZE * IMG_SIZE * K0) + (start_row_id + i) * K0 + channel_id];
    }

    if(threadIdx.x < K0){
        A0_row[threadIdx.x] = HALF_T(0.f);
    }
    __syncthreads();

    // Put the partial reduction together
    for(int i = 0; i < reduction_thread_per_channel; ++i){
        if((threadIdx.x / K0) == i){
            A0_row[channel_id] += tmp * static_cast<HALF_T>(1.f/(IMG_SIZE * IMG_SIZE));
        }
        __syncthreads();
    }
    // Wait until W0, bias0 arrives
    __pipeline_wait_prior(0);

    // Issue async memcpy to get W1 and bias1
    if(threadIdx.x < (N1 * K1)/8){
        __pipeline_memcpy_async(&W1_shared[8 * threadIdx.x],
                                &W1[8 * threadIdx.x],
                                8 * sizeof(HALF_T));
    }
    

    // Need to confirm this, if send k byte then input must be k-aligned?
    if(threadIdx.x < (N1 / 4)){
        __pipeline_memcpy_async(&bias1_shared[4 * threadIdx.x], &bias1[4 * threadIdx.x],
                                4 * sizeof(HALF_T));
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
    // TODO: it's probably better to map each warp to a 16 x 2 tile in W0_shared to get better bandwidth utilization and remove
    // the bank conflict above. Not sure if this fix will be too important since global memory access latency seems to dominate in this kernel. 
    // Also __shfl_down_sync is probably the best way to do reduction.

    // Now the first N0 term of W0_shared stores the output
    if(threadIdx.x < N0){
        W0_shared[threadIdx.x] += bias0_shared[threadIdx.x];
        if(W0_shared[threadIdx.x] < HALF_T(0)){
            W0_shared[threadIdx.x] = HALF_T(0);
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
        HALF_T tmp = W1_shared[threadIdx.x] + bias1_shared[threadIdx.x];
        W1_shared[threadIdx.x] = 1.0/(1.0 + exp(-tmp));
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

template <int N0,
          typename HALF_T = at::Half,
          int K0 = N0 * 16,
          int N1 = K0,
          int K1 = N0,
          int warp_M = 16,
          int IMG_SIZE = 64 / N0>
__global__ void SEb2bGEMMFused_Tensor_Core_Kernel(
    const HALF_T* __restrict__ activation,
    const HALF_T* __restrict__ W0,
    const HALF_T* __restrict__ bias0,
    const HALF_T* __restrict__ W1,
    const HALF_T* __restrict__ bias1,
    HALF_T* __restrict__ downsampled) {
    // In this kernel N0 is at least 16, we can use Tensor core to compute the two GEMM
    // Each threadblock is responsible for 16 sub-samples
    // Each warp is responsible for a 16x16x16 (or 8x32x16?) tile in the 2 GEMMs
    // This will require loading the two weight matrices entirely to shared memory
    // Another way to do it is to split K dimension, but it will require a few more kernel launches
    using namespace nvcuda;
    // constexpr int warp_M = 16;
    constexpr int warp_N = 256 / warp_M;
    constexpr int warp_K = 16;
    constexpr int warp_size = 32;

    // Must equal to blockDim.x / 32
    constexpr int num_warp = K0/warp_K;
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;

    const int row_start = blockIdx.x * warp_M;
    const int col_start = warp_id * warp_K;
    
    __shared__ HALF_T activation_tile[num_warp][warp_M*warp_K];
    // before reduction first load W0 bias0 async to shared mem
    __shared__ HALF_T W0_shared[N0 * K0];
    __shared__ HALF_T W1_shared[N1 * K1];
    __shared__ HALF_T bias0_shared[N0];
    __shared__ HALF_T bias1_shared[N1];

    for (int i = 0; i < (N0 * K0 + 8 * blockDim.x - 1) / (8 * blockDim.x); ++i) {
        const int linear_idx = (i * blockDim.x + threadIdx.x) * 8;
        if (linear_idx < N0 * K0) {
            __pipeline_memcpy_async(&W0_shared[linear_idx], &W0[linear_idx],
                                    8 * sizeof(HALF_T));
        }
    }

    // N0 = 16 or 32
    if(threadIdx.x < (N0 / 8)){
        __pipeline_memcpy_async(&bias0_shared[8 * threadIdx.x], &bias0[8 * threadIdx.x],
                                8 * sizeof(HALF_T));
    }

    __pipeline_commit();


    // Reduction
    for(int i = 0; i < (warp_M * warp_K) / warp_size; ++i){
        const int n = row_start + ((warp_size * i) + lane_id) / warp_K;
        const int c = col_start + ((warp_size * i) + lane_id) % warp_K;
        HALF_T tmp(0.f);

        #pragma unroll (IMG_SIZE * IMG_SIZE)
        for(int j = 0; j < (IMG_SIZE * IMG_SIZE); ++j){
            // reduction over (n,. ,. , c)
            tmp += activation[n*IMG_SIZE * IMG_SIZE*K0 + j*K0 + c];
        }
        activation_tile[warp_id][(warp_size * i) + lane_id] = tmp * static_cast<HALF_T>(1.f/(IMG_SIZE * IMG_SIZE));
    }
    __syncwarp();

    // First GEMM
    __pipeline_wait_prior(0);
    
    // Before first GEMM, load W1 and bias1
    for (int i = 0; i < (N1 * K1 + 8 * blockDim.x - 1) / (8 * blockDim.x);
         ++i) {
        const int linear_idx = (i * blockDim.x + threadIdx.x) * 8;
        if (linear_idx < N1 * K1) {
            __pipeline_memcpy_async(&W1_shared[linear_idx], &W1[linear_idx],
                                    8 * sizeof(HALF_T));
        }
    }

    // N1 = 256 or 512
    if (threadIdx.x < (N1 / 8)) {
        __pipeline_memcpy_async(&bias1_shared[8 * threadIdx.x],
                                &bias1[8 * threadIdx.x], 8 * sizeof(HALF_T));
    }

    __pipeline_commit();

    // Start first GEMM
    wmma::fragment<wmma::matrix_a, warp_M, warp_N, warp_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, warp_M, warp_N, warp_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, warp_M, warp_N, warp_K, half> acc_frag;

    // wmma ops can only be called on cuda half, but not at::Half...
    half *activation_cuda_half_ptr = reinterpret_cast<half*>(activation_tile[warp_id]);
    half *W0_cuda_half_ptr = reinterpret_cast<half*>(W0_shared + col_start * N0);
    wmma::load_matrix_sync(a_frag, activation_cuda_half_ptr, warp_K);
    wmma::load_matrix_sync(b_frag, W0_cuda_half_ptr, N0);
    const HALF_T zero(0.f);
    wmma::fill_fragment(acc_frag, *reinterpret_cast<const half*>(&zero));
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(W0_cuda_half_ptr, acc_frag, N0, wmma::mem_row_major);

    __syncthreads();
    // Now do a reduction over the tiles in the same block
    for(unsigned int s = num_warp /2; s > 0; s>>=1){
        if(warp_id < s){
            for(int i = 0; i < (warp_M * warp_N)/ warp_size; ++i){
                const int idx = warp_id * warp_M * warp_N + i * warp_size + lane_id;
                W0_shared[idx] += W0_shared[idx + s * (warp_M * warp_N)];
            }
        }
        __syncthreads();
    }
    // Now first warp_M rows of W0_shared contains the result of matrix multiplication
    // Important, the number of threads in a block must be at least warp_M * warp_N
    if(threadIdx.x < warp_M * warp_N){
        W0_shared[threadIdx.x] += bias0_shared[threadIdx.x % warp_N];
        if(W0_shared[threadIdx.x] < HALF_T(0)){
            W0_shared[threadIdx.x] = HALF_T(0);
        }
    } // Relu done
    __syncthreads();
    __pipeline_wait_prior(0);
    
    // Second GEMM col_start in Activation is the same as in W1
    wmma::fill_fragment(acc_frag, *reinterpret_cast<const half*>(&zero));

    // Hard code for each of the two cases
    if(warp_N == 32){
        const int second_half = warp_id / (num_warp / 2);
        const int local_col_start = (warp_id % (num_warp / 2)) * warp_N;
        half *activation1_cuda_half_ptr = reinterpret_cast<half*>(W0_shared + second_half * warp_K);
        half *W1_cuda_half_ptr = reinterpret_cast<half*>(W1_shared + second_half * warp_K * N1 + local_col_start);
        wmma::load_matrix_sync(a_frag, activation1_cuda_half_ptr, K1);
        wmma::load_matrix_sync(b_frag, W1_cuda_half_ptr, N1);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        wmma::store_matrix_sync(W1_cuda_half_ptr, acc_frag, N1, wmma::mem_row_major);

        // need a reduction
        __syncthreads();
        for(int i = 0; i < (N1 * warp_M)/blockDim.x; ++i){
            const int linear_idx = threadIdx.x + i * blockDim.x;
            W1_shared[linear_idx] += W1_shared[linear_idx + warp_K * N1] + bias1_shared[linear_idx % N1];
            // sigmoid
            W1_shared[linear_idx] = 1.0/(1.0 + exp(-W1_shared[linear_idx]));
        }

    }

    if(warp_N == 16){
        half *activation1_cuda_half_ptr = reinterpret_cast<half*>(W0_shared);
        half *W1_cuda_half_ptr = reinterpret_cast<half*>(W1_shared + warp_id * warp_N);
        wmma::load_matrix_sync(a_frag, activation1_cuda_half_ptr, K1);
        wmma::load_matrix_sync(b_frag, W1_cuda_half_ptr, N1);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        wmma::store_matrix_sync(W1_cuda_half_ptr, acc_frag, N1, wmma::mem_row_major);

        __syncthreads();
        for(int i = 0; i < (warp_M * N1)/blockDim.x; ++i){
            const int linear_idx = threadIdx.x + i * blockDim.x;
            W1_shared[linear_idx] += bias1_shared[linear_idx % N1];
            W1_shared[linear_idx] = 1.0/(1.0 + exp(-W1_shared[linear_idx]));
        }
    }

    __syncthreads();

    // Finally multiply the activation with size (IMG_SIZE, IMG_SIZE, K0)
    // (IMG_SIZE * IMG_SIZE * K0) must be an integer multiple of blockDim.x
    #pragma unroll 2
    for(int i = 0; i < (IMG_SIZE * IMG_SIZE * K0) / blockDim.x; ++i){
        const int inner_idx = threadIdx.x + i * blockDim.x;
        downsampled[blockIdx.x * (IMG_SIZE * IMG_SIZE * K0) + inner_idx] +=
            activation[blockIdx.x * (IMG_SIZE * IMG_SIZE * K0) + inner_idx] *
            W1_shared[inner_idx % K0];
    }


}

template <int N0>
class MyCudaSE{

    // using ElementInput = cutlass::half_t;
    using ElementInput = at::Half;
    // using ElementInput = half;

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
           downsampled(static_cast<ElementInput*>(downsampled.data_ptr())) {
               static_assert((N0 == 4) || (N0 == 8));
           }

    void run() {
        SEb2bGEMMFused_Kernel<N0, ElementInput>
            <<<128, 512>>>(activation, W1, bias1, W2, bias2, downsampled);
        cudaError_t err = cudaGetLastError();
        std::cout << cudaGetErrorString(err) << std::endl;
    }
};

template <int N0>
class MyCudaSE2{

    // using ElementInput = cutlass::half_t;
    using ElementInput = at::Half;
    // using ElementInput = half;

    ElementInput* activation;
    ElementInput* W1;
    ElementInput* bias1;
    ElementInput* W2;
    ElementInput* bias2;
    ElementInput* downsampled;

    public:
     MyCudaSE2(torch::Tensor Activation,
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
           downsampled(static_cast<ElementInput*>(downsampled.data_ptr())) {
               static_assert((N0 == 16) || (N0 == 32));
           }

    void run() {
        int warp_M = 16;
        if(N0 == 32){
            warp_M = 8;
        }
        const int n = 128;
        const int channel = 256;
        const int n_threads = (channel/16) * 32;

        SEb2bGEMMFused_Tensor_Core_Kernel<N0, ElementInput>
            <<<n/warp_M, n_threads>>>(activation, W1, bias1, W2, bias2, downsampled);
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

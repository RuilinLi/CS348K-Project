/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include <cutlass/half.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes: D =  op(accumulator * scale + bias1) + bias2 to an array of elements.
///
/// op = relu if ApplyRelu = true, otherwise op = identity map
///
template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  bool ApplyRelu = true,
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationReluFixUp {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {


    ElementCompute const *bias1_ptr;
    ElementCompute const *bias2_ptr;
    ElementCompute const *scale_ptr;
    ElementCompute threshold;
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : bias1_ptr(nullptr),
          bias2_ptr(nullptr),
          scale_ptr(nullptr),
          threshold(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *bias1_ptr, ElementCompute const *bias2_ptr)
        : bias1_ptr(bias1_ptr),
          bias2_ptr(bias2_ptr),
          scale_ptr(nullptr),
          threshold(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *bias1_ptr,
           ElementCompute const *bias2_ptr,
           ElementCompute const *scale_ptr)
        : bias1_ptr(bias1_ptr),
          bias2_ptr(bias2_ptr),
          scale_ptr(scale_ptr),
          threshold(ElementCompute(0)) {}
  };

private:

  //
  // Data members
  //

  ElementCompute bias1_;
  ElementCompute bias2_;
  ElementCompute scale_;
  ElementCompute threshold_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationReluFixUp(Params const &params) {
    bias1_ = (params.bias1_ptr)?(*params.bias1_ptr):ElementCompute(0);
    bias2_ = (params.bias2_ptr)?(*params.bias2_ptr):ElementCompute(0);
    scale_ = (params.scale_ptr)?(*params.scale_ptr):ElementCompute(1);
    threshold_ = params.threshold;

    // printf("Bias1 is %f and bias2 is %f, thresh is %f, scale is %f \n",
    //        static_cast<float>(bias1_), static_cast<float>(bias2_),
    //        static_cast<float>(threshold_), static_cast<float>(scale_));
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return false;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
      if (k_partition) {
          // not really needed becuase
          // in the first K-partition there should be no source
          // in the rest of the partitions bias1_ is not used
          bias1_ = ElementCompute(0);
      }

      if (k_partition != k_partition_count - 1) {
          // set to NaN to make ReLU no-op for all except last k partitions
          int64_t allones = -1;
          threshold_ = reinterpret_cast<ElementCompute const &>(allones);
          // Only add the second bias at the last K-partition
          bias2_ = ElementCompute(0);
      }
  }

  /// Computes: D =  relu(accumulator + bias1) + bias2
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator) const {
    // if(blockIdx.x + threadIdx.x ==0){
    //      printf("Bias1 is %f and bias2 is %f, thresh is  %f \n", static_cast<float>(bias1_), static_cast<float>(bias2_), static_cast<float>(threshold_));
    // }

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    // D = scale * Accum + bias1
    // Should probably just add this to functional.h
    multiply_add<ElementCompute> scalar_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
        intermediate[i] = scalar_op(scale_, converted_accumulator[i], bias1_);
    }

    // Compute Relu optionally
    if (ApplyRelu) {
        ReLu<ComputeFragment> relu;
        intermediate = relu(threshold_, intermediate);
    }
    plus<ComputeFragment> plus_obj;
    intermediate = plus_obj(bias2_, intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    return destination_converter(intermediate);
  }

  // This method is only needed for split-k 
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
      // Convert source to interal compute numeric type
      NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
          source_converter;
      NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
          accumulator_converter;

      ComputeFragment converted_source = source_converter(source);
      ComputeFragment converted_accumulator =
          accumulator_converter(accumulator);

      // Perform binary operations
      ComputeFragment intermediate;

      // D = scale * Accum + bias1
      multiply_add<ElementCompute> scalar_op;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
          intermediate[i] = scalar_op(scale_, converted_accumulator[i], bias1_);
      }
      // Compute Relu optionally
      if (ApplyRelu) {
          ReLu<ComputeFragment> relu;
          intermediate = relu(threshold_, intermediate);
      }
      
      plus<ComputeFragment> plus_obj;
      intermediate = plus_obj(bias2_, intermediate);

      // Convert to destination numeric type
      NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
          destination_converter;

      return destination_converter(intermediate);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

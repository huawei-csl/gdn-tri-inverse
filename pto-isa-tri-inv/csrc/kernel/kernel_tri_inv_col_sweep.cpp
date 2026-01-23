/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

#define MEMORY_BASE

#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

using namespace pto;

constexpr unsigned NUM_BLOCKS = 20;    // number of AIVs
constexpr unsigned UB_SIZE = 0x30000;  // 192KB UB of A2A3

template <typename T, unsigned matrix_size>
AICORE void runTTriInv(__gm__ T* z, __gm__ T* x, uint32_t total_length) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t tile_len = matrix_size * matrix_size;

  // UB zero address
  constexpr unsigned UB_ZERO_ADDR = 0x0;
  constexpr unsigned Z_UB_ADDR = tile_len * sizeof(T);

  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, matrix_size, matrix_size>;
  using StridDim5 = pto::Stride<1, 1, 1, matrix_size, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
  GlobalData xGlobal(x);
  GlobalData zGlobal(z);

  // define TileData on UB buffer with static shape and dynamic mask
  using TileData = Tile<TileType::Vec, T, matrix_size, matrix_size,
                        BLayout::RowMajor, -1, -1>;

  // define ping-pong buffer for related tiles
  TileData xTiles(matrix_size, matrix_size);
  TileData zTiles(matrix_size, matrix_size);

  // assign the UB address for each tile
  TASSIGN(xTiles, UB_ZERO_ADDR);
  TASSIGN(zTiles, Z_UB_ADDR);

  // total number of loops of one vector core
  int32_t loopCount = matrix_size;
  // address offset between vector cores
  unsigned offset = block_idx * tile_len;

  // synchronization operations between hardware pipelines
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  event_t event_flag = static_cast<event_t>(0);
  for (uint32_t i = 0; i < loopCount; i++) {
    unsigned inner_offset = offset + i * matrix_size;
    // Prepare read GM offset
    TASSIGN(xGlobal, x + inner_offset);
    TASSIGN(zGlobal, z + inner_offset);

    wait_flag(PIPE_V, PIPE_MTE2, event_flag);
    // load data from global memory to UB buffer
    TLOAD(xTiles, xGlobal);

    set_flag(PIPE_MTE2, PIPE_V, event_flag);
    wait_flag(PIPE_MTE2, PIPE_V, event_flag);

    wait_flag(PIPE_MTE3, PIPE_V, event_flag);
    // perform elementwise absolute value
    TABS(zTiles, xTiles);
    set_flag(PIPE_V, PIPE_MTE2, event_flag);

    set_flag(PIPE_V, PIPE_MTE3, event_flag);
    wait_flag(PIPE_V, PIPE_MTE3, event_flag);
    // store data from UB buffer to global memory
    TSTORE(zGlobal, zTiles);
    set_flag(PIPE_MTE3, PIPE_V, event_flag);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TASSIGN(zGlobal, z);
  z = zGlobal.data();
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(GM_ADDR x, GM_ADDR z,
                                                          uint32_t in_length) {
  // Define the tile size
  constexpr unsigned martix_size = 64;
  // main kernel, totalLength is dynamic input
  runTTriInv<half, martix_size>((__gm__ half*)z, (__gm__ half*)x, in_length);
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(GM_ADDR x, GM_ADDR z,
                                                          uint32_t in_length) {
  // Define the tile size
  constexpr unsigned martix_size = 64;
  // main kernel, totalLength is dynamic input
  runTTriInv<float, martix_size>((__gm__ float*)z, (__gm__ float*)x, in_length);
}

#endif

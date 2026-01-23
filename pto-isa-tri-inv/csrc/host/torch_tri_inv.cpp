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

#include "aclrtlaunch_triv_inv_col_sweep_fp16.h"
#include "aclrtlaunch_triv_inv_col_sweep_fp32.h"
#include "utils.h"

namespace ascendc_path {

at::Tensor run_tri_inv(const at::Tensor& x) {
  const auto dtype = x.options().dtype();
  at::Tensor z = at::empty_like(x);
  // Define the number of blocks of vector core
  uint32_t blockDim = 20;
  uint32_t totalLength = x.numel();

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp16, blockDim, x, z, totalLength);

  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp32, blockDim, x, z, totalLength);

  } else {
    throw std::runtime_error("Unsupported dtype for add_custom kernel");
  }

  return z;
}
}  // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("tri_inv(Tensor x) -> Tensor"); }
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("tri_inv", TORCH_FN(ascendc_path::run_tri_inv));
}
}  // namespace

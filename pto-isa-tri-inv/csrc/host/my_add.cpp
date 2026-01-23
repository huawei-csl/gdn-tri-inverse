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

#include "aclrtlaunch_add_custom_fp16.h"
#include "aclrtlaunch_add_custom_fp32.h"
#include "utils.h"

namespace ascendc_path {

at::Tensor run_add_custom(const at::Tensor& x, const at::Tensor& y) {
  const auto dtype = x.options().dtype();
  at::Tensor z = at::empty_like(x);
  // Define the number of blocks of vector core
  uint32_t blockDim = 20;
  uint32_t totalLength = x.numel();

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(add_custom_fp16, blockDim, x, y, z, totalLength);

  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(add_custom_fp32, blockDim, x, y, z, totalLength);

  } else {
    throw std::runtime_error("Unsupported dtype for add_custom kernel");
  }

  return z;
}
}  // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m) {
  // Declare the custom operator schema
  m.def("my_add(Tensor x, Tensor y) -> Tensor");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  // Register the custom operator implementation function
  m.impl("my_add", TORCH_FN(ascendc_path::run_add_custom));
}
}  // namespace

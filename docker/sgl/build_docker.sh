#!/bin/bash

export GDN_TRI_INVERSE_LOCAL_PATH="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/"
export GDN_TRI_INVERSE_COMMIT="${GDN_TRI_INVERSE_COMMIT:-"$(git rev-parse --verify HEAD)"}"
export SGL_DOCKER_HOSTNAME="${SGL_DOCKER_HOSTNAME:-}"
export SGL_KERNEL_NPU_BRANCH_OR_TAG="${SGL_KERNEL_NPU_BRANCH_OR_TAG:-2026.6.2}" # 6-triinv-integrate-tri_inv_cube_col_sweep-kernel
export SGL_KERNEL_NPU_HTTPS_GIT_URL="${SGL_KERNEL_NPU_HTTPS_GIT_URL:-https://github.com/sgl-project/sgl-kernel-npu.git}" # https://github.com/gioelegott/sgl-kernel-npu.git
export TILELANG_ASCEND_COMMIT="${TILELANG_ASCEND_COMMIT:-d4736eb}" # Only supported with CANN 8.5.0
export CANN_VERSION="${CANN_VERSION:-9.0.0}" # Supported: 8.5.0 (with tilelang-ascend) and 9.0.0 (w/o tilelang-ascend)
export DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-gdn-tri-inverse:${CANN_VERSION}-25062026}"
export DOCKERFILE="cann-${CANN_VERSION}.Dockerfile"

pushd $GDN_TRI_INVERSE_LOCAL_PATH \
&& docker build --build-arg SGL_DOCKER_HOSTNAME="${SGL_DOCKER_HOSTNAME}" \
    --build-arg SGL_KERNEL_NPU_BRANCH_OR_TAG="${SGL_KERNEL_NPU_BRANCH_OR_TAG}" \
    --build-arg SGL_KERNEL_NPU_HTTPS_GIT_URL="${SGL_KERNEL_NPU_HTTPS_GIT_URL}" \
    --build-arg TILELANG_ASCEND_COMMIT="${TILELANG_ASCEND_COMMIT}" \
    --build-arg GDN_TRI_INVERSE_COMMIT="${GDN_TRI_INVERSE_COMMIT}" \
    -t "${DOCKER_IMAGE_TAG}" \
    -f "${DOCKERFILE}" . \
&& popd

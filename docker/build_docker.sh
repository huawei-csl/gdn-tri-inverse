#!/bin/bash
export GDN_TRI_INVERSE_LOCAL_PATH="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/../"
export SGL_DOCKER_HOSTNAME="${SGL_DOCKER_HOSTNAME:-}"
export SGL_KERNEL_NPU_BRANCH="${SGL_KERNEL_NPU_BRANCH:-6-triinv-integrate-tri_inv_cube_col_sweep-kernel}"
export SGL_KERNEL_NPU_HTTPS_GIT_URL="${SGL_KERNEL_NPU_HTTPS_GIT_URL:-https://github.com/gioelegott/sgl-kernel-npu.git}"
export PTO_KERNELS_BRANCH="${PTO_KERNELS_BRANCH:-main}"
export PTO_KERNELS_HTTPS_GIT_URL="${PTO_KERNELS_HTTPS_GIT_URL:-https://github.com/huawei-csl/pto-kernels.git}"
export TILELANG_ASCEND_COMMIT="${TILELANG_ASCEND_COMMIT:-d4736eb}"
export DOCKER_IMAGE_TAG="gdn-main-cann-8.5.0"

pushd $GDN_TRI_INVERSE_LOCAL_PATH \
&& docker build --build-arg SGL_DOCKER_HOSTNAME="${SGL_DOCKER_HOSTNAME}" \
    --build-arg SGL_KERNEL_NPU_BRANCH="${SGL_KERNEL_NPU_BRANCH}" \
    --build-arg SGL_KERNEL_NPU_HTTPS_GIT_URL="${SGL_KERNEL_NPU_HTTPS_GIT_URL}" \
    --build-arg PTO_KERNELS_BRANCH="${PTO_KERNELS_BRANCH}" \
    --build-arg PTO_KERNELS_HTTPS_GIT_URL="${PTO_KERNELS_HTTPS_GIT_URL}" \
    --build-arg TILELANG_ASCEND_COMMIT="${TILELANG_ASCEND_COMMIT}" \
    -t "${DOCKER_IMAGE_TAG}" \
    -f docker/Dockerfile . \
&& popd
